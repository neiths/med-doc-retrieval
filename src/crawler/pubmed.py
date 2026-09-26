"""PubMed client for fetching candidate English medical documents via NCBI Entrez and Europe PMC."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import httpx
from loguru import logger


class PubMedClient:
    """Client for querying PubMed using Europe PMC, PubTator 3.0, and NCBI Entrez."""

    NCBI_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    NCBI_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    EUROPE_PMC_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    PUBTATOR_SEARCH_URL = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/search/"
    PUBTATOR_EXPORT_URL = (
        "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/biocjson"
    )

    def __init__(
        self,
        email: str = "team@r2ai2026.org",
        api_key: str | None = None,
        timeout_seconds: int = 15,
        cache_file: Path | str | None = "data/processed/pubmed_cache.jsonl",
    ):
        self.email = email
        self.api_key = api_key
        self.timeout = httpx.Timeout(timeout_seconds)
        self.cache_file = Path(cache_file) if cache_file else None
        self._memory_cache: dict[str, dict[str, Any]] = self._load_cache()

    def _load_cache(self) -> dict[str, dict[str, Any]]:
        """Loads cached articles indexed by doc_id (PMID)."""
        cache = {}
        if self.cache_file and self.cache_file.exists():
            with open(self.cache_file, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            item = json.loads(line)
                            if "doc_id" in item:
                                cache[str(item["doc_id"])] = item
                        except Exception:
                            continue
            logger.info(f"Loaded {len(cache)} cached PubMed articles from {self.cache_file}")
        return cache

    def _append_cache(self, articles: list[dict[str, Any]]):
        """Appends newly retrieved articles to disk and memory cache."""
        new_items = []
        for art in articles:
            doc_id = str(art["doc_id"])
            if doc_id not in self._memory_cache:
                self._memory_cache[doc_id] = art
                new_items.append(art)

        if new_items and self.cache_file:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "a", encoding="utf-8") as f:
                for item in new_items:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def search_europe_pmc_sync(self, query: str, page_size: int = 30) -> list[dict[str, Any]]:
        """Synchronous search on Europe PMC API."""
        params = {
            "query": query,
            "format": "json",
            "pageSize": page_size,
            "resultType": "core",
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(self.EUROPE_PMC_URL, params=params)
                response.raise_for_status()
                data = response.json()

            results = []
            for item in data.get("resultList", {}).get("result", []):
                pmid = item.get("pmid")
                if not pmid:
                    continue
                title = item.get("title", "")
                abstract = item.get("abstractText", "")
                full_text = f"{title}\n\n{abstract}" if title and abstract else (title or abstract)
                results.append(
                    {
                        "doc_id": str(pmid),
                        "title": title,
                        "text": full_text,
                        "lang": "en",
                        "source": "europe_pmc",
                    }
                )
            return results
        except Exception as e:
            logger.warning(f"Europe PMC search failed for '{query}': {e}")
            return []

    def search_ncbi_sync(self, query: str, retmax: int = 30) -> list[dict[str, Any]]:
        """Synchronous search on NCBI ESearch + EFetch."""
        params_esearch = {
            "db": "pubmed",
            "term": query,
            "retmax": retmax,
            "retmode": "json",
            "email": self.email,
        }
        if self.api_key:
            params_esearch["api_key"] = self.api_key

        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.get(self.NCBI_ESEARCH_URL, params=params_esearch)
                res.raise_for_status()
                data = res.json()
                pmids = data.get("esearchresult", {}).get("idlist", [])

                if not pmids:
                    return []

                # Check cache first
                uncached_pmids = [p for p in pmids if p not in self._memory_cache]
                articles = [self._memory_cache[p] for p in pmids if p in self._memory_cache]

                if uncached_pmids:
                    params_efetch = {
                        "db": "pubmed",
                        "id": ",".join(uncached_pmids),
                        "retmode": "xml",
                        "email": self.email,
                    }
                    if self.api_key:
                        params_efetch["api_key"] = self.api_key

                    res_fetch = client.get(self.NCBI_EFETCH_URL, params=params_efetch)
                    res_fetch.raise_for_status()

                    root = ET.fromstring(res_fetch.text)
                    newly_fetched = []
                    for article in root.findall(".//PubmedArticle"):
                        pmid_elem = article.find(".//MedlineCitation/PMID")
                        pmid = pmid_elem.text if pmid_elem is not None else ""
                        if not pmid:
                            continue

                        title_elem = article.find(".//ArticleTitle")
                        title = (
                            "".join(title_elem.itertext()).strip() if title_elem is not None else ""
                        )

                        abstract_texts = []
                        for abs_elem in article.findall(".//AbstractText"):
                            label = abs_elem.get("Label")
                            text = "".join(abs_elem.itertext()).strip()
                            if label:
                                abstract_texts.append(f"{label}: {text}")
                            elif text:
                                abstract_texts.append(text)
                        abstract = "\n".join(abstract_texts).strip()

                        full_text = (
                            f"{title}\n\n{abstract}" if title and abstract else (title or abstract)
                        )
                        newly_fetched.append(
                            {
                                "doc_id": str(pmid),
                                "title": title,
                                "text": full_text,
                                "lang": "en",
                                "source": "pubmed",
                            }
                        )

                    self._append_cache(newly_fetched)
                    articles.extend(newly_fetched)

                return articles
        except Exception as e:
            logger.warning(f"NCBI Search/Fetch failed for '{query}': {e}")
            return []

    def search_pubtator_sync(self, query: str, page_size: int = 30) -> list[dict[str, Any]]:
        """Synchronous search on PubTator 3.0 API with BiocJSON abstract export."""
        params = {"text": query, "limit": page_size}
        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.get(self.PUBTATOR_SEARCH_URL, params=params)
                res.raise_for_status()
                data = res.json()
                results_raw = data.get("results", [])
                if not results_raw:
                    return []

                pmids = []
                title_map = {}
                for item in results_raw:
                    pmid = str(item.get("pmid") or item.get("_id") or "")
                    if pmid:
                        pmids.append(pmid)
                        if "title" in item:
                            title_map[pmid] = item["title"]

                if not pmids:
                    return []

                uncached_pmids = [p for p in pmids if p not in self._memory_cache]
                articles = [self._memory_cache[p] for p in pmids if p in self._memory_cache]

                if uncached_pmids:
                    export_params = {"pmids": ",".join(uncached_pmids)}
                    exp_resp = client.get(self.PUBTATOR_EXPORT_URL, params=export_params)
                    exp_resp.raise_for_status()
                    exp_data = exp_resp.json()

                    newly_fetched = []
                    for doc in exp_data.get("PubTator3", []):
                        pmid = str(doc.get("pmid") or doc.get("id") or "")
                        if not pmid:
                            continue

                        passages = {
                            p.get("infons", {}).get("type", ""): p.get("text", "")
                            for p in doc.get("passages", [])
                        }
                        title = passages.get("title", "") or title_map.get(pmid, "")
                        abstract = passages.get("abstract", "")
                        full_text = (
                            f"{title}\n\n{abstract}" if title and abstract else (title or abstract)
                        )

                        newly_fetched.append(
                            {
                                "doc_id": pmid,
                                "title": title,
                                "text": full_text,
                                "lang": "en",
                                "source": "pubtator",
                            }
                        )

                    fetched_pmids = {a["doc_id"] for a in newly_fetched}
                    for pmid in uncached_pmids:
                        if pmid not in fetched_pmids and pmid in title_map:
                            newly_fetched.append(
                                {
                                    "doc_id": pmid,
                                    "title": title_map[pmid],
                                    "text": title_map[pmid],
                                    "lang": "en",
                                    "source": "pubtator",
                                }
                            )

                    self._append_cache(newly_fetched)
                    articles.extend(newly_fetched)

                return articles
        except Exception as e:
            logger.warning(f"PubTator search failed for '{query}': {e}")
            return []

    def search_candidate_articles(
        self,
        query: str,
        top_k: int = 30,
        source: str = "europe_pmc",
    ) -> list[dict[str, Any]]:
        """Fetches candidates using Europe PMC, PubTator 3.0, or NCBI, with deduplication and caching.

        Args:
            query: Medical keywords or search query.
            top_k: Max articles to return.
            source: 'europe_pmc', 'pubtator', 'ncbi', or 'hybrid'.
        """
        if not query or not query.strip():
            return []

        results = []
        if source in ["europe_pmc", "hybrid"]:
            results.extend(self.search_europe_pmc_sync(query, page_size=top_k))
            self._append_cache(results)

        if (len(results) < top_k) and source in ["pubtator", "hybrid"]:
            pubtator_results = self.search_pubtator_sync(query, page_size=top_k)
            results.extend(pubtator_results)

        if (len(results) < top_k) and source in ["ncbi", "hybrid"]:
            ncbi_results = self.search_ncbi_sync(query, retmax=top_k)
            results.extend(ncbi_results)

        # Fallback cascade if primary selected source returned empty
        if not results:
            if source == "europe_pmc":
                results = self.search_pubtator_sync(query, page_size=top_k)
                if not results:
                    results = self.search_ncbi_sync(query, retmax=top_k)
            elif source == "pubtator":
                results = self.search_europe_pmc_sync(query, page_size=top_k)
                if not results:
                    results = self.search_ncbi_sync(query, retmax=top_k)

        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for art in results:
            did = str(art["doc_id"])
            if did not in seen:
                seen.add(did)
                deduped.append(art)

        return deduped[:top_k]

    # Asynchronous methods for batch / crawler usage
    async def search_ncbi_pmids(
        self,
        query: str,
        retmax: int = 50,
        client: httpx.AsyncClient | None = None,
    ) -> list[str]:
        """Search NCBI for PMIDs matching query asynchronously."""
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
            return data.get("esearchresult", {}).get("idlist", [])

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
        """Fetch title and abstract for a list of PMIDs via EFetch XML asynchronously."""
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
                articles.append(
                    {
                        "doc_id": str(pmid),
                        "title": title,
                        "text": full_text,
                        "lang": "en",
                        "source": "pubmed",
                    }
                )
        except Exception as e:
            logger.error(f"Error parsing NCBI XML: {e}")

        return articles

    async def search_europe_pmc(
        self,
        query: str,
        page_size: int = 50,
        client: httpx.AsyncClient | None = None,
    ) -> list[dict[str, Any]]:
        """Search Europe PMC asynchronously."""
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
                results.append(
                    {
                        "doc_id": str(pmid),
                        "title": title,
                        "text": full_text,
                        "lang": "en",
                        "source": "europe_pmc",
                    }
                )
            return results
        except Exception as e:
            logger.warning(f"Europe PMC search failed for '{query}': {e}")
            return []

    async def search_pubtator(
        self,
        query: str,
        page_size: int = 30,
        client: httpx.AsyncClient | None = None,
    ) -> list[dict[str, Any]]:
        """Search PubTator 3.0 asynchronously with BiocJSON abstract export."""
        params = {"text": query, "limit": page_size}

        async def _do_search(c: httpx.AsyncClient):
            resp = await c.get(self.PUBTATOR_SEARCH_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
            results_raw = data.get("results", [])
            if not results_raw:
                return []

            pmids = []
            title_map = {}
            for item in results_raw:
                pmid = str(item.get("pmid") or item.get("_id") or "")
                if pmid:
                    pmids.append(pmid)
                    if "title" in item:
                        title_map[pmid] = item["title"]

            if not pmids:
                return []

            uncached_pmids = [p for p in pmids if p not in self._memory_cache]
            articles = [self._memory_cache[p] for p in pmids if p in self._memory_cache]

            if uncached_pmids:
                export_params = {"pmids": ",".join(uncached_pmids)}
                exp_resp = await c.get(self.PUBTATOR_EXPORT_URL, params=export_params)
                exp_resp.raise_for_status()
                exp_data = exp_resp.json()

                newly_fetched = []
                for doc in exp_data.get("PubTator3", []):
                    pmid = str(doc.get("pmid") or doc.get("id") or "")
                    if not pmid:
                        continue
                    passages = {
                        p.get("infons", {}).get("type", ""): p.get("text", "")
                        for p in doc.get("passages", [])
                    }
                    title = passages.get("title", "") or title_map.get(pmid, "")
                    abstract = passages.get("abstract", "")
                    full_text = (
                        f"{title}\n\n{abstract}" if title and abstract else (title or abstract)
                    )
                    newly_fetched.append(
                        {
                            "doc_id": pmid,
                            "title": title,
                            "text": full_text,
                            "lang": "en",
                            "source": "pubtator",
                        }
                    )

                fetched_pmids = {a["doc_id"] for a in newly_fetched}
                for pmid in uncached_pmids:
                    if pmid not in fetched_pmids and pmid in title_map:
                        newly_fetched.append(
                            {
                                "doc_id": pmid,
                                "title": title_map[pmid],
                                "text": title_map[pmid],
                                "lang": "en",
                                "source": "pubtator",
                            }
                        )

                self._append_cache(newly_fetched)
                articles.extend(newly_fetched)

            return articles

        try:
            if client:
                return await _do_search(client)
            async with httpx.AsyncClient(timeout=self.timeout) as c:
                return await _do_search(c)
        except Exception as e:
            logger.warning(f"PubTator async search failed for '{query}': {e}")
            return []
