"""
Tool schemas and dispatch table for the Orchestrator.

Each entry in DEFINITIONS uses the Responses API internally-tagged format.
DISPATCH maps the tool name to the callable that actually does the work.

Adding a new tool means:
  1. Define its schema here
  2. Add it to DISPATCH
  3. The Orchestrator picks it up automatically — no other changes needed.
"""

from agent.product_expert import search_product_brochures
from agent.cost_estimator import estimate_service_cost, list_equipment_services
from agent.benchmarker import benchmark_process


DEFINITIONS = [
    {
        "type": "function",
        "name": "search_product_brochures",
        "description": (
            "Search product brochures to answer questions about Metso equipment. "
            "For specific questions (named product + named spec), returns passages "
            "from the single most relevant brochure. "
            "For discovery or comparison questions ('what options exist for X', "
            "'does X work for Y', 'what do we have for Z'), automatically returns "
            "passages from up to three brochures so the answer can cover multiple products."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The question or topic to search for",
                },
            },
            "required": ["query"],
        },
    },
    {
        "type": "function",
        "name": "estimate_service_cost",
        "description": (
            "Estimate field-service cost from historical records. "
            "Returns a cost range with confidence level and similar past jobs."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "service_type": {
                    "type": "string",
                    "description": (
                        "A short snake_case label describing the service the user is asking about. "
                        "Use the most specific term that fits, e.g. 'impeller_replacement', "
                        "'rotor_stator_inspection', 'filter_cloth_replacement', 'seal_replacement', "
                        "'inspection', 'commissioning'. Infer it from the user's wording."
                    ),
                },
                "user_query": {
                    "type": "string",
                    "description": "The user's exact question about service cost, copied verbatim",
                },
                "equipment_model": {
                    "type": "string",
                    "description": (
                        "The equipment the user wants to service, as they described it. "
                        "Use their exact wording — 'ColumnCell', 'Larox filter', "
                        "'MD pump' are all fine."
                    ),
                },
            },
            "required": ["service_type", "user_query", "equipment_model"],
        },
    },
    {
        "type": "function",
        "name": "list_equipment_services",
        "description": (
            "List the service types that have been performed on a given equipment model, "
            "based on historical service records. Use this when the user asks what services "
            "are available or typical for a piece of equipment, or when the service type "
            "is unknown and you want to offer options before estimating cost."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "equipment_model": {
                    "type": "string",
                    "description": "The equipment model to look up services for.",
                },
            },
            "required": ["equipment_model"],
        },
    },
    {
        "type": "function",
        "name": "benchmark_process",
        "description": (
            "Benchmark customer flotation process readings against historical plant data. "
            "Returns percentile ranks, quartile ranges, and top correlated controls."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "readings": {
                    "type": "object",
                    "description": (
                        "Metric name → numeric value. "
                        "Keys: silica_pct, iron_pct, feed_iron_pct, feed_silica_pct"
                    ),
                    "additionalProperties": {"type": "number"},
                },
            },
            "required": ["readings"],
        },
    },
]

DISPATCH: dict[str, callable] = {
    "search_product_brochures":  search_product_brochures,
    "estimate_service_cost":     estimate_service_cost,
    "list_equipment_services":   list_equipment_services,
    "benchmark_process":         benchmark_process,
}
