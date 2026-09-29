import json
from unittest.mock import AsyncMock, patch

import pytest

from src.crawler.url_scraper import ArticleScraper


def test_extract_text_html_boilerplate_removal():
    scraper = ArticleScraper()
    mock_html = """
    <!DOCTYPE html>
    <html>
    <head><title>Bệnh Sỏi Thận - Triệu chứng và Điều trị</title></head>
    <body>
        <nav><a href="/">Trang chủ</a></nav>
        <div class="sidebar advertisement">Quảng cáo thuốc đông y 100% khỏi</div>
        <article>
            <h1>Bệnh Sỏi Thận - Triệu chứng và Điều trị</h1>
            <p>Sỏi thận là hiện tượng lắng đọng chất khoáng trong thận tạo thành các tinh thể rắn.</p>
            <p>Các phương pháp điều trị gồm có tán sỏi ngoài cơ thể và phẫu thuật nội soi.</p>
        </article>
        <footer>Bản quyền thuộc về bệnh viện</footer>
    </body>
    </html>
    """
    res = scraper.extract_text(mock_html, fallback_url="https://example.com/soi-than")
    assert "Sỏi Thận" in res["title"]
    assert "lắng đọng chất khoáng" in res["text"]
    assert "tán sỏi ngoài cơ thể" in res["text"]
    # Check boilerplate cleanup
    assert "Quảng cáo thuốc" not in res["text"]
    assert "Bản quyền thuộc về" not in res["text"]


@pytest.mark.anyio
async def test_scrape_urls_resumable(tmp_path):
    input_file = tmp_path / "urls.jsonl"
    output_file = tmp_path / "crawled.jsonl"

    items = [
        {"id": "doc1", "url": "https://example.com/art1"},
        {"id": "doc2", "url": "https://example.com/art2"},
        {"id": "doc3", "url": "https://example.com/art3"},
    ]
    with open(input_file, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")

    # Pre-populate doc1 as already scraped
    existing = {"doc_id": "doc1", "url": "https://example.com/art1", "title": "Art 1", "text": "Content 1", "lang": "vi", "status": "success"}
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(existing) + "\n")

    scraper = ArticleScraper()

    # Mock fetch_url so it only gets called for doc2 and doc3
    async def mock_fetch_url(client, url):
        return f"<html><head><title>Title for {url}</title></head><body><p>Text content for {url}</p></body></html>"

    with patch.object(scraper, "fetch_url", side_effect=mock_fetch_url) as mock_fetch:
        results = await scraper.scrape_urls_jsonl(input_file, output_file, resume=True)

        # Only doc2 and doc3 should have been fetched
        assert mock_fetch.call_count == 2

    # Verify that all 3 documents exist in output
    assert len(results) == 3
    doc_ids = [r["doc_id"] for r in results]
    assert "doc1" in doc_ids
    assert "doc2" in doc_ids
    assert "doc3" in doc_ids

    # Verify output file has 3 lines
    lines = [line.strip() for line in open(output_file, encoding="utf-8") if line.strip()]
    assert len(lines) == 3
