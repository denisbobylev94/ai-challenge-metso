from unittest.mock import patch, MagicMock


def test_product_search_returns_one_source_document():
    from agent.product_expert import search_product_brochures

    fake_results = [
        {
            "text": "Chunk A",
            "metadata": {"source": "doc-a.pdf", "section": "Page 1"},
            "score": 0.9,
        },
        {
            "text": "Chunk B",
            "metadata": {"source": "doc-a.pdf", "section": "Page 2"},
            "score": 0.8,
        },
        {
            "text": "Chunk C",
            "metadata": {"source": "doc-b.pdf", "section": "Page 1"},
            "score": 0.2,
        },
    ]

    with patch("agent.product_expert._get_hybrid_search") as mock_hs, \
         patch("agent.product_expert.config") as mock_config:
        mock_config.client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=[0.0] * 1536)]
        )
        mock_config.OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
        mock_hs.return_value.search.return_value = fake_results

        result = search_product_brochures("md pump")
        assert result["found"] is True
        assert result["source_documents"] == ["doc-a.pdf"]
        assert len({p["source"] for p in result["passages"]}) == 1
        assert all(p["source"] == "doc-a.pdf" for p in result["passages"])

