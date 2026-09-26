"""Unit tests for PubMed, Europe PMC, and PubTator 3.0 client."""

from unittest.mock import MagicMock, patch

import pytest

from src.crawler.pubmed import PubMedClient


def test_pubmed_cache(tmp_path):
    cache_file = tmp_path / "test_cache.jsonl"
    client = PubMedClient(cache_file=cache_file)

    sample_articles = [
        {
            "doc_id": "12345",
            "title": "Study on Kidney Stones",
            "text": "Study on Kidney Stones\n\nAbstract text here",
            "lang": "en",
            "source": "pubtator",
        }
    ]
    client._append_cache(sample_articles)

    assert "12345" in client._memory_cache
    assert cache_file.exists()

    # Load cache from new client instance
    new_client = PubMedClient(cache_file=cache_file)
    assert "12345" in new_client._memory_cache
    assert new_client._memory_cache["12345"]["title"] == "Study on Kidney Stones"


def test_search_pubtator_mocked(tmp_path):
    cache_file = tmp_path / "pubtator_cache.jsonl"
    client = PubMedClient(cache_file=cache_file)

    mock_search_json = {
        "results": [
            {"_id": "1001", "pmid": 1001, "title": "PubTator Kidney Paper"},
            {"_id": "1002", "pmid": 1002, "title": "PubTator Lithotripsy Paper"},
        ]
    }

    mock_export_json = {
        "PubTator3": [
            {
                "pmid": 1001,
                "passages": [
                    {"infons": {"type": "title"}, "text": "PubTator Kidney Paper"},
                    {"infons": {"type": "abstract"}, "text": "Detailed abstract about kidney."},
                ],
            },
            {
                "pmid": 1002,
                "passages": [
                    {"infons": {"type": "title"}, "text": "PubTator Lithotripsy Paper"},
                    {
                        "infons": {"type": "abstract"},
                        "text": "Detailed abstract about lithotripsy.",
                    },
                ],
            },
        ]
    }

    def fake_get(url, params=None, **kwargs):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        if "/search/" in str(url):
            resp.json.return_value = mock_search_json
        elif "biocjson" in str(url):
            resp.json.return_value = mock_export_json
        return resp

    with patch("httpx.Client.get", side_effect=fake_get):
        results = client.search_pubtator_sync("kidney stones", page_size=2)
        assert len(results) == 2
        assert results[0]["doc_id"] == "1001"
        assert "Detailed abstract about kidney." in results[0]["text"]
        assert results[0]["source"] == "pubtator"


def test_search_candidate_articles_routing(tmp_path):
    client = PubMedClient(cache_file=tmp_path / "cache.jsonl")

    with patch.object(
        client,
        "search_pubtator_sync",
        return_value=[
            {"doc_id": "999", "title": "T", "text": "A", "lang": "en", "source": "pubtator"}
        ],
    ) as mock_pubtator:
        results = client.search_candidate_articles("kidney", top_k=5, source="pubtator")
        assert len(results) == 1
        assert results[0]["doc_id"] == "999"
        mock_pubtator.assert_called_once()


def test_search_candidate_articles_fallback(tmp_path):
    client = PubMedClient(cache_file=tmp_path / "cache.jsonl")

    # If pubtator returns empty, fallback to europe_pmc or ncbi
    with (
        patch.object(client, "search_pubtator_sync", return_value=[]),
        patch.object(
            client,
            "search_europe_pmc_sync",
            return_value=[
                {
                    "doc_id": "888",
                    "title": "PMC Title",
                    "text": "PMC Text",
                    "lang": "en",
                    "source": "europe_pmc",
                }
            ],
        ),
    ):
        results = client.search_candidate_articles("kidney", top_k=5, source="pubtator")
        assert len(results) == 1
        assert results[0]["doc_id"] == "888"


@pytest.mark.anyio
async def test_search_pubtator_async(tmp_path):
    client = PubMedClient(cache_file=tmp_path / "cache_async.jsonl")

    mock_search_json = {"results": [{"_id": "2001", "pmid": 2001, "title": "Async Title"}]}
    mock_export_json = {
        "PubTator3": [
            {
                "pmid": 2001,
                "passages": [
                    {"infons": {"type": "title"}, "text": "Async Title"},
                    {"infons": {"type": "abstract"}, "text": "Async Abstract"},
                ],
            }
        ]
    }

    def fake_get(url, params=None, **kwargs):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        if "/search/" in str(url):
            resp.json.return_value = mock_search_json
        elif "biocjson" in str(url):
            resp.json.return_value = mock_export_json
        return resp

    with patch("httpx.AsyncClient.get", side_effect=fake_get):
        results = await client.search_pubtator("test query", page_size=1)
        assert len(results) == 1
        assert results[0]["doc_id"] == "2001"
        assert "Async Abstract" in results[0]["text"]
