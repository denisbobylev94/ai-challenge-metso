import os
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Sales Agent", layout="wide")


def _processed_exists() -> bool:
    return Path("./processed/chroma_db").exists() and Path("./processed/service_records.db").exists()


@st.cache_resource
def get_agent():
    from agent.orchestrator import Orchestrator as Agent
    return Agent()


def main():
    if not _processed_exists():
        st.warning(
            "⚠️ Data not ingested yet. Run `python -m ingestion.run_all` first, "
            "then refresh this page."
        )

    # Layout
    chat_col, sidebar_col = st.columns([2, 1])

    # Sidebar
    with sidebar_col:
        if _processed_exists():
            agent = get_agent()

            # Show last action metadata
            if "last_response" in st.session_state:
                resp = st.session_state.last_response
                st.divider()
                st.subheader("Last Action")
                tool_icons = {
                    "search_product_brochures": "🔍 brochures",
                    "estimate_service_cost": "💰 cost",
                    "benchmark_process": "📊 benchmark",
                }
                for t in resp.tools_used:
                    st.write(tool_icons.get(t, t))

                # Sources
                brochure_sources = [s for s in resp.sources if "section" in s]
                cost_sources = [s for s in resp.sources if "cost_source" in s]

                if brochure_sources:
                    st.subheader("Sources")
                    unique_docs = list(dict.fromkeys(s.get("source") for s in brochure_sources if s.get("source")))
                    if unique_docs:
                        st.caption(f"Primary document: {unique_docs[0]}")
                    for s in brochure_sources[:5]:
                        st.caption(f"📄 {s.get('source')} — {s.get('section')}")

                if cost_sources:
                    st.subheader("Cost Evidence")
                    for j in cost_sources[:5]:
                        cost = j.get("cost")
                        currency = j.get("currency") or ""
                        cost_str = f"{currency} {cost:.0f}".strip() if cost else "N/A"
                        source_doc = j.get("source_document") or f"{j.get('id')}.txt"
                        st.caption(
                            f"[{j.get('id')}] {j.get('country')} — {cost_str} "
                            f"({j.get('cost_source')}) — {source_doc}"
                        )

    # Chat
    with chat_col:
        st.title("Sales Agent")

        if "messages" not in st.session_state:
            welcome = (
                "👋 Hello! I'm your technical sales assistant. I can help you with:\n\n"
                "- 🔍 **Product information** — specs and features from product brochures\n"
                "- 💰 **Service cost estimation** — cost ranges from historical service records\n"
                "- 📊 **Process benchmarking** — compare your process data against historical plant data\n\n"
                "How can I help you today?"
            )
            st.session_state.messages = [{"role": "assistant", "content": welcome}]

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        if prompt := st.chat_input("Ask about products, service costs, or process benchmarks..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            if _processed_exists():
                agent = get_agent()
                with st.spinner("Thinking..."):
                    resp = agent.chat(prompt)
                st.session_state.last_response = resp
                st.session_state.messages.append({"role": "assistant", "content": resp.text})
                with st.chat_message("assistant"):
                    st.write(resp.text)
                st.rerun()
            else:
                st.error("Please run ingestion first: `python -m ingestion.run_all`")


if __name__ == "__main__":
    main()
