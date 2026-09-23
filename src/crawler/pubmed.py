"""PubMed client for fetching candidate English medical documents via NCBI Entrez and Europe PMC."""

import xml.etree.ElementTree as ET
from typing import Any

import httpx
from loguru import logger


class PubMedClient:
    """Client for querying PubMed using NCBI Entrez E-utilities and Europe PMC API."""

    NCBI_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    NCBI_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    EUROPE_PMC_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

    def __init__(
        self,
        email: str = "team@r2ai2026.org",
        api_key: str | None = None,
        timeout_seconds: int = 15,
    ):
        self.email = email
        self.api_key = api_key
        self.timeout = httpx.Timeout(timeout_seconds)

    async def search_ncbi_pmids(
        self,
        query: str,
        retmax: int = 50,
        client: httpx.AsyncClient | None = None,
    ) -> list[str]:
        """Search NCBI for PMIDs matching query."""
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": retmax,
            "retmode": "json",
            "email": self.email,
        }
        if self.api_key:
            params["api_key"] = self.api_key

        async def _request(c: httpx.AsyncClient):
            response = await c.get(self.NCBI_ESEARCH_URL, params=params)
            response.raise_for_status()
            data = response.json()
            id_list = data.get("esearchresult", {}).get("idlist", [])
            return id_list

        try:
            if client:
                return await _request(client)
            async with httpx.AsyncClient(timeout=self.timeout) as c:
                return await _request(c)
        except Exception as e:
            logger.warning(f"NCBI ESearch failed for query '{query}': {e}")
            return []

    async def fetch_ncbi_abstracts(
        self,
        pmids: list[str],
        client: httpx.AsyncClient | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch title and abstract for a list of PMIDs via EFetch XML."""
        if not pmids:
            return []

        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "email": self.email,
        }
        if self.api_key:
            params["api_key"] = self.api_key

        async def _request(c: httpx.AsyncClient):
            response = await c.get(self.NCBI_EFETCH_URL, params=params)
            response.raise_for_status()
            return response.text

        try:
            if client:
                xml_text = await _request(client)
            else:
                async with httpx.AsyncClient(timeout=self.timeout) as c:
                    xml_text = await _request(c)
        except Exception as e:
            logger.warning(f"NCBI EFetch failed for {len(pmids)} PMIDs: {e}")
            return []

        # Parse XML
        articles = []
        try:
            root = ET.fromstring(xml_text)
            for article in root.findall(".//PubmedArticle"):
                pmid_elem = article.find(".//MedlineCitation/PMID")
                pmid = pmid_elem.text if pmid_elem is not None else ""
                if not pmid:
                    continue

                title_elem = article.find(".//ArticleTitle")
                title = "".join(title_elem.itertext()).strip() if title_elem is not None else ""

                abstract_texts = []
                for abs_elem in article.findall(".//AbstractText"):
                    label = abs_elem.get("Label")
                    text = "".join(abs_elem.itertext()).strip()
                    if label:
                        abstract_texts.append(f"{label}: {text}")
                    elif text:
                        abstract_texts.append(text)
                abstract = "\n".join(abstract_texts).strip()

                full_text = f"{title}\n\n{abstract}" if title and abstract else (title or abstract)

                articles.append({
                    "doc_id": pmid,
                    "title": title,
                    "text": full_text,
                    "lang": "en",
                    "source": "pubmed",
                })
        except Exception as e:
            logger.error(f"Error parsing NCBI XML: {e}")

        return articles

    async def search_europe_pmc(
        self,
        query: str,
        page_size: int = 50,
        client: httpx.AsyncClient | None = None,
    ) -> list[dict[str, Any]]:
        """Search Europe PMC as an alternative or complement to NCBI."""
        params = {
            "query": query,
            "format": "json",
            "pageSize": page_size,
            "resultType": "core",
        }

        async def _request(c: httpx.AsyncClient):
            response = await c.get(self.EUROPE_PMC_URL, params=params)
            response.raise_for_status()
            return response.json()

        try:
            if client:
                data = await _request(client)
            else:
                async with httpx.AsyncClient(timeout=self.timeout) as c:
                    data = await _request(c)

            results = []
            for item in data.get("resultList", {}).get("result", []):
                pmid = item.get("pmid")
                if not pmid:
                    continue
                title = item.get("title", "")
                abstract = item.get("abstractText", "")
                full_text = f"{title}\n\n{abstract}" if title and abstract else (title or abstract)
                results.append({
                    "doc_id": str(pmid),
                    "title": title,
                    "text": full_text,
                    "lang": "en",
                    "source": "europe_pmc",
                })
            return results
        except Exception as e:
            logger.warning(f"Europe PMC search failed for '{query}': {e}")
            return []
