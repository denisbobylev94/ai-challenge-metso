import pytest
from unittest.mock import patch, MagicMock
from ingestion.services.service_records import ServiceRecord


# ── ServiceRecord.has_pricing_signal ──────────────────────────────────────────

def test_ingestion_filter_skips_no_data_records():
    record = ServiceRecord(total_cost=None)
    assert record.has_pricing_signal is False


def test_ingestion_filter_keeps_cost_records():
    record = ServiceRecord(total_cost=1500.0)
    assert record.has_pricing_signal is True


# ── ServiceRecord.cost_source ─────────────────────────────────────────────────

def test_no_cost_records_are_classified_correctly():
    record = ServiceRecord(total_cost=None)
    assert record.cost_source == "no_invoice"


def test_estimate_with_no_vector_results():
    from agent.cost_estimator import estimate_service_cost
    with patch("agent.cost_estimator.VectorStore") as MockVS, \
         patch("agent.cost_estimator.config") as mock_config:
        mock_config.client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=[0.0] * 1536)]
        )
        mock_config.SERVICE_DB_PATH = ":memory:"
        mock_config.OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
        MockVS.return_value.query.return_value = []
        result = estimate_service_cost(
            "filter_cloth_replacement", "how much does a filter cloth replacement cost?", "Larox PF-DS"
        )
        assert "found" in result


def test_estimate_with_no_records():
    from agent.cost_estimator import estimate_service_cost
    with patch("agent.cost_estimator.VectorStore") as MockVS, \
         patch("agent.cost_estimator.config") as mock_config:
        mock_config.client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=[0.0] * 1536)]
        )
        mock_config.SERVICE_DB_PATH = ":memory:"
        mock_config.OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
        MockVS.return_value.query.return_value = []
        result = estimate_service_cost("bearing_service", "how much does bearing service cost?", "MD-650")
        assert result["found"] is False


def test_estimate_returns_range_not_point():
    from agent.cost_estimator import estimate_service_cost
    import sqlite3, tempfile, os

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        tmp_db = f.name

    try:
        conn = sqlite3.connect(tmp_db)
        conn.execute("""CREATE TABLE service_records (
            id TEXT, filename TEXT, service_types TEXT, equipment_model TEXT,
            country TEXT, total_cost REAL,
            currency TEXT, cost_source TEXT,
            work_summary TEXT, raw_text TEXT)""")
        conn.execute("INSERT INTO service_records VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("FSR-001","FSR-001.txt",'["bearing_service"]',"MD-650","Sweden",
             1000.0, "EUR", "actual_recorded", "Test job", "raw"))
        conn.execute("INSERT INTO service_records VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("FSR-002","FSR-002.txt",'["bearing_service"]',"MD-650","Sweden",
             2000.0, "EUR", "actual_recorded", "Test job 2", "raw"))
        conn.commit()
        conn.close()

        with patch("agent.cost_estimator.VectorStore") as MockVS, \
             patch("agent.cost_estimator.config") as mock_config:
            mock_config.client.embeddings.create.return_value = MagicMock(
                data=[MagicMock(embedding=[0.0] * 1536)]
            )
            mock_config.SERVICE_DB_PATH = tmp_db
            mock_config.OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
            MockVS.return_value.query.return_value = [
                {"id": "FSR-001", "distance": 0.3},
                {"id": "FSR-002", "distance": 0.3},
            ]
            result = estimate_service_cost("bearing_service", "bearing service cost estimate", "MD-650")
            assert result["found"] is True
            eur_range = result["estimate"]["ranges_by_currency"]["EUR"]
            assert eur_range["min_cost"] < eur_range["max_cost"]
            assert "actual_records_used" in result["estimate"]
    finally:
        os.unlink(tmp_db)


def test_source_ids_are_present():
    from agent.cost_estimator import estimate_service_cost
    import sqlite3, tempfile, os

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        tmp_db = f.name

    try:
        conn = sqlite3.connect(tmp_db)
        conn.execute("""CREATE TABLE service_records (
            id TEXT, filename TEXT, service_types TEXT, equipment_model TEXT,
            country TEXT, total_cost REAL,
            currency TEXT, cost_source TEXT,
            work_summary TEXT, raw_text TEXT)""")
        conn.execute("INSERT INTO service_records VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("FSR-010","FSR-010.txt",'["inspection"]',"Concorde Cell","Germany",
             500.0, "EUR", "actual_recorded", "Inspection job", "raw"))
        conn.commit()
        conn.close()

        with patch("agent.cost_estimator.VectorStore") as MockVS, \
             patch("agent.cost_estimator.config") as mock_config:
            mock_config.client.embeddings.create.return_value = MagicMock(
                data=[MagicMock(embedding=[0.0] * 1536)]
            )
            mock_config.SERVICE_DB_PATH = tmp_db
            mock_config.OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
            MockVS.return_value.query.return_value = [{"id": "FSR-010", "distance": 0.3}]
            result = estimate_service_cost("inspection", "inspection visit cost", "Concorde Cell")
            assert result["found"] is True
            assert "source_record_ids" in result
            assert len(result["source_record_ids"]) > 0
    finally:
        os.unlink(tmp_db)


def test_same_currency_records_form_one_range():
    from agent.cost_estimator import estimate_service_cost
    import sqlite3, tempfile, os

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        tmp_db = f.name

    try:
        conn = sqlite3.connect(tmp_db)
        conn.execute("""CREATE TABLE service_records (
            id TEXT, filename TEXT, service_types TEXT, equipment_model TEXT,
            country TEXT, total_cost REAL,
            currency TEXT, cost_source TEXT, work_summary TEXT, raw_text TEXT)""")
        for fsr, cost in [("FSR-A", 1000.0), ("FSR-B", 2000.0), ("FSR-C", 1500.0)]:
            conn.execute("INSERT INTO service_records VALUES (?,?,?,?,?,?,?,?,?,?)",
                (fsr, f"{fsr}.txt", '["inspection"]', "ColumnCell", "Sweden",
                 cost, "EUR", "actual_recorded", "Job", "raw"))
        conn.commit()
        conn.close()

        with patch("agent.cost_estimator.VectorStore") as MockVS, \
             patch("agent.cost_estimator.config") as mock_config:
            mock_config.client.embeddings.create.return_value = MagicMock(
                data=[MagicMock(embedding=[0.0] * 1536)]
            )
            mock_config.SERVICE_DB_PATH = tmp_db
            mock_config.OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
            MockVS.return_value.query.return_value = [
                {"id": "FSR-A", "distance": 0.3},
                {"id": "FSR-B", "distance": 0.3},
                {"id": "FSR-C", "distance": 0.3},
            ]
            result = estimate_service_cost("inspection", "cost of inspection visit", "ColumnCell")
            assert result["found"] is True
            assert "EUR" in result["estimate"]["ranges_by_currency"]
            assert len(result["estimate"]["ranges_by_currency"]) == 1
    finally:
        os.unlink(tmp_db)


def test_mixed_currencies_separate_groups():
    from agent.cost_estimator import estimate_service_cost
    import sqlite3, tempfile, os

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        tmp_db = f.name

    try:
        conn = sqlite3.connect(tmp_db)
        conn.execute("""CREATE TABLE service_records (
            id TEXT, filename TEXT, service_types TEXT, equipment_model TEXT,
            country TEXT, total_cost REAL,
            currency TEXT, cost_source TEXT, work_summary TEXT, raw_text TEXT)""")
        conn.execute("INSERT INTO service_records VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("FSR-E", "FSR-E.txt", '["inspection"]', "ColumnCell", "Sweden",
             2000.0, "EUR", "actual_recorded", "EUR job", "raw"))
        conn.execute("INSERT INTO service_records VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("FSR-U", "FSR-U.txt", '["inspection"]', "ColumnCell", "Canada",
             3000.0, "USD", "actual_recorded", "USD job", "raw"))
        conn.commit()
        conn.close()

        with patch("agent.cost_estimator.VectorStore") as MockVS, \
             patch("agent.cost_estimator.config") as mock_config:
            mock_config.client.embeddings.create.return_value = MagicMock(
                data=[MagicMock(embedding=[0.0] * 1536)]
            )
            mock_config.SERVICE_DB_PATH = tmp_db
            mock_config.OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
            MockVS.return_value.query.return_value = [
                {"id": "FSR-E", "distance": 0.3},
                {"id": "FSR-U", "distance": 0.3},
            ]
            result = estimate_service_cost("inspection", "cost of inspection visit", "ColumnCell")
            assert result["found"] is True
            assert len(result["estimate"]["ranges_by_currency"]) == 2
    finally:
        os.unlink(tmp_db)


def test_cost_response_includes_source_documents():
    from agent.cost_estimator import estimate_service_cost
    import sqlite3, tempfile, os

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        tmp_db = f.name

    try:
        conn = sqlite3.connect(tmp_db)
        conn.execute("""CREATE TABLE service_records (
            id TEXT, filename TEXT, service_types TEXT, equipment_model TEXT,
            country TEXT, total_cost REAL,
            currency TEXT, cost_source TEXT, work_summary TEXT, raw_text TEXT)""")
        conn.execute("INSERT INTO service_records VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("FSR-S", "FSR-S.txt", '["inspection"]', "ColumnCell", "Sweden",
             2000.0, "EUR", "actual_recorded", "EUR job", "raw"))
        conn.commit()
        conn.close()

        with patch("agent.cost_estimator.VectorStore") as MockVS, \
             patch("agent.cost_estimator.config") as mock_config:
            mock_config.client.embeddings.create.return_value = MagicMock(
                data=[MagicMock(embedding=[0.0] * 1536)]
            )
            mock_config.SERVICE_DB_PATH = tmp_db
            mock_config.OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
            MockVS.return_value.query.return_value = [{"id": "FSR-S", "distance": 0.3}]
            result = estimate_service_cost("inspection", "cost of inspection visit", "ColumnCell")
            assert result["found"] is True
            assert "source_documents" in result
            assert "FSR-S.txt" in result["source_documents"]
            assert result["similar_jobs"][0]["source_document"] == "FSR-S.txt"
    finally:
        os.unlink(tmp_db)
