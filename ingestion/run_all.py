"""
Ingestion entry point — runs brochure and FSR ingestion in sequence.

Usage:
    python -m ingestion.run_all
"""


def main() -> None:
    """Ingest all brochures and field service reports into their respective stores."""
    import shutil
    from pathlib import Path
    import config  # validates API key early
    from ingestion.products.brochures import ingest_brochures
    from ingestion.services.service_records import ingest_service_records

    processed = Path("./processed")
    if processed.exists():
        print("→ Clearing previous processed/ files...")
        shutil.rmtree(processed)
    processed.mkdir(parents=True)

    print("→ Ingesting brochures...")
    n1 = ingest_brochures()
    print(f"  {n1} chunks indexed")

    print("→ Ingesting service records...")
    n2 = ingest_service_records()
    print(f"  {n2} records indexed")

    print("✓ Done. Run: streamlit run app.py")


if __name__ == "__main__":
    main()
