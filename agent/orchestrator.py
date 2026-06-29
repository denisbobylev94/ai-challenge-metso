"""
Orchestrator — the main agent loop (LLM-driven tool calling).

For every user message:
  1. A cheap classifier labels the intent (product | cost | benchmark | general).
  2. The model is called via the Responses API with the tool set. The LLM decides
     which tool(s) to call and extracts the arguments; tool_choice nudges the first
     call so product questions always retrieve and cost questions always confirm
     equipment first (the two-step cost flow).
  3. Tools run in code and their structured results are fed back; the model
     synthesises the final natural-language answer.

State is threaded server-side via ``previous_response_id``: each call sends only the
NEW items (the user message, or the tool outputs) and references the prior response,
so OpenAI keeps the reasoning / function_call / output chain intact between turns.

Tools are defined in agent/tools.py; the worker functions live in their own files.
"""

import json
import time
from dataclasses import dataclass, field

import config
from agent.prompts import SYSTEM_PROMPT
from agent.tools import DEFINITIONS as TOOL_DEFINITIONS, DISPATCH as TOOL_DISPATCH
from agent.benchmarker import load_data as load_flotation_data


# ── Return type ───────────────────────────────────────────────────────────────

@dataclass
class AgentResponse:
    text: str                          # final answer shown to the user
    tools_used: list[str] = field(default_factory=list)   # which tools fired
    sources: list[dict] = field(default_factory=list)     # passages / cost records


# ── Orchestrator ──────────────────────────────────────────────────────────────

class Orchestrator:
    """Routes each message to the right worker via Responses-API function calling."""

    _MAX_TOOL_TURNS = 4  # cap on model→tool round-trips per user message

    def __init__(self) -> None:
        self._last_response_id: str | None = None  # threads context across turns
        self._equipment_listed = False             # True once list_equipment_services ran

        try:
            load_flotation_data()
        except Exception:
            pass

    # ── Main entry point ──────────────────────────────────────────────────────

    def chat(self, user_message: str) -> AgentResponse:
        intent = self._classify_intent(user_message)
        new_input: list = [{"role": "user", "content": user_message}]

        # Greetings and out-of-scope questions need no tools — answer directly.
        if intent == "general":
            response = self._respond(new_input)
            return AgentResponse(text=response.output_text or "")

        # Nudge the first tool call:
        #   product       → always retrieve from brochures
        #   cost (step 1) → always confirm equipment + services first
        #   else          → let the model choose
        if intent == "product":
            tool_choice: str | dict = {"type": "function", "name": "search_product_brochures"}
        elif intent == "cost" and not self._equipment_listed:
            tool_choice = {"type": "function", "name": "list_equipment_services"}
        else:
            tool_choice = "auto"

        return self._run_tool_loop(new_input, tool_choice)

    def _run_tool_loop(self, new_input: list, first_tool_choice: str | dict) -> AgentResponse:
        """Loop model→tool→model until the model returns a final text answer.

        ``tool_choice`` is nudged only on the first turn; afterwards "auto" lets the
        model chain tools (e.g. list_equipment_services → estimate_service_cost) or stop.
        """
        tools_used: list[str] = []
        sources: list[dict] = []
        tool_choice = first_tool_choice

        for _ in range(self._MAX_TOOL_TURNS):
            response = self._respond(new_input, tools=TOOL_DEFINITIONS, tool_choice=tool_choice)

            function_calls = [item for item in response.output if item.type == "function_call"]
            if not function_calls:
                return AgentResponse(
                    text=response.output_text or "", tools_used=tools_used, sources=sources,
                )

            tool_result_items, turn_tools, turn_sources = self._execute_tools(function_calls)
            tools_used.extend(turn_tools)
            sources.extend(turn_sources)

            new_input = tool_result_items  # next turn submits only the tool outputs
            tool_choice = "auto"           # never re-nudge after the first turn

        # Too many tool turns — ask the model to answer with no further tools.
        response = self._respond(new_input)
        return AgentResponse(text=response.output_text or "", tools_used=tools_used, sources=sources)

    # ── Tool execution ────────────────────────────────────────────────────────

    def _execute_tools(self, function_calls) -> tuple[list, list, list]:
        """Run every function call; return function_call_output items, names, sources."""
        tool_result_items: list = []
        tools_used: list[str] = []
        sources: list[dict] = []

        for call in function_calls:
            name = call.name
            args = json.loads(call.arguments)
            result = self._dispatch(name, args)

            tools_used.append(name)
            sources.extend(self._extract_sources(name, result))
            if name == "list_equipment_services":
                self._equipment_listed = True

            tool_result_items.append({
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": json.dumps(result),
            })

        return tool_result_items, tools_used, sources

    def _dispatch(self, name: str, args: dict) -> dict:
        """Call the named worker, normalising the one argument shape the model varies."""
        worker = TOOL_DISPATCH.get(name)
        if worker is None:
            return {"error": f"Unknown tool: {name}"}
        # The model sometimes passes benchmark metrics as top-level kwargs instead of
        # nesting them inside {"readings": {...}}; normalise before dispatching.
        if name == "benchmark_process" and "readings" not in args:
            args = {"readings": args}
        return worker(**args)

    @staticmethod
    def _extract_sources(tool_name: str, result: dict) -> list[dict]:
        """Pull sidebar-displayable sources out of a tool result."""
        if not result.get("found"):
            return []
        if tool_name == "search_product_brochures":
            return result.get("passages", [])
        if tool_name == "estimate_service_cost":
            return result.get("similar_jobs", [])
        return []

    # ── Intent classification ─────────────────────────────────────────────────

    def _classify_intent(self, user_message: str) -> str:
        """Label the message: product | cost | benchmark | general (defaults to general)."""
        response = config.client.chat.completions.create(
            model=config.CLASSIFICATION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify the user message into exactly one category. "
                        "Reply with only the category word, nothing else.\n\n"
                        "cost — questions about price, cost, quote, or how much something costs, "
                        "even if they mention a specific part or service (e.g. 'impeller "
                        "replacement cost', 'maintenance cost for MD650'). Also classify a bare "
                        "service name as 'cost' if it answers a pending cost question.\n"
                        "product — questions about equipment specs, features, suitability, or "
                        "comparisons that do NOT ask about price.\n"
                        "benchmark — questions about process performance metrics or flotation "
                        "plant data.\n"
                        "general — greetings, clarifications, or anything else."
                    ),
                },
                {"role": "user", "content": user_message},
            ],
            max_completion_tokens=5,
            temperature=0,
        )
        label = (response.choices[0].message.content or "").strip().lower()
        return label if label in {"product", "cost", "benchmark"} else "general"

    # ── Responses API call ────────────────────────────────────────────────────

    def _respond(self, items: list, tools=None, tool_choice: str | dict = "auto", max_retries: int = 3):
        """Call the Responses API with retry, threading state via previous_response_id."""
        kwargs: dict = {
            "model": config.SYNTHESIS_MODEL,
            "instructions": SYSTEM_PROMPT,
            "input": items,
            "store": True,
            "previous_response_id": self._last_response_id,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        for attempt in range(max_retries):
            try:
                response = config.client.responses.create(**kwargs)
                self._last_response_id = response.id
                return response
            except Exception:
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)


# Backwards-compatible alias — app.py imports Orchestrator as Agent
Agent = Orchestrator
