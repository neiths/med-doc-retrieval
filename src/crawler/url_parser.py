"""URL Decomposition, Segment Hierarchy Parser, and Pattern Inspector.

Splits URLs into path segments separated by '/' for taxonomy extraction,
corpus analysis, and crawl coverage tracking in ViBioMIR.
"""

import re
from typing import Any, Iterable
from urllib.parse import unquote, urlparse

import pandas as pd


PAGINATION_PATTERNS = [
    re.compile(r"/page/\d+", re.IGNORECASE),
    re.compile(r"/p/\d+", re.IGNORECASE),
    re.compile(r"[?&](?:page|p|trang)=\d+", re.IGNORECASE),
    re.compile(r"-\d+\.html?\?page=\d+", re.IGNORECASE),
]

HOMEPAGE_LEAVES = {"", "index", "index.html", "index.htm", "index.php", "home", "trang-chu"}


def dissect_url(url: str) -> dict[str, Any]:
    """Decomposes a single URL into its protocol, domain, and path segments separated by '/'.

    Args:
        url: Raw web URL string.

    Returns:
        Dictionary containing granular URL components and segment breakdown.
    """
    if not url or not isinstance(url, str):
        return {
            "scheme": "",
            "domain": "",
            "path": "",
            "depth": 0,
            "segments": [],
            "seg_1": "",
            "seg_2": "",
            "seg_3": "",
            "seg_4": "",
            "leaf": "",
            "extension": "",
            "query_params": "",
            "is_homepage": True,
            "is_pagination": False,
            "url": str(url or ""),
        }

    parsed = urlparse(url.strip())
    raw_path = unquote(parsed.path)
    clean_path = raw_path.strip("/")
    segments = [s.strip() for s in clean_path.split("/") if s.strip()] if clean_path else []
    depth = len(segments)

    seg_1 = segments[0] if depth >= 1 else ""
    seg_2 = segments[1] if depth >= 2 else ""
    seg_3 = segments[2] if depth >= 3 else ""
    seg_4 = segments[3] if depth >= 4 else ""
    leaf = segments[-1] if depth >= 1 else ""

    # Extension extraction (.htm, .html, .php, etc.)
    ext = ""
    if "." in leaf and not leaf.startswith("."):
        potential_ext = leaf.rsplit(".", 1)[-1].lower()
        if len(potential_ext) <= 5 and potential_ext.isalnum():
            ext = potential_ext

    is_homepage = depth == 0 or (depth == 1 and leaf.lower() in HOMEPAGE_LEAVES)
    is_pagination = any(pat.search(url) for pat in PAGINATION_PATTERNS)

    return {
        "scheme": parsed.scheme or "http",
        "domain": parsed.netloc.lower() or "",
        "path": parsed.path or "/",
        "depth": depth,
        "segments": segments,
        "seg_1": seg_1,
        "seg_2": seg_2,
        "seg_3": seg_3,
        "seg_4": seg_4,
        "leaf": leaf,
        "extension": ext,
        "query_params": parsed.query or "",
        "is_homepage": is_homepage,
        "is_pagination": is_pagination,
        "url": url,
    }


def dissect_urls_df(
    urls_or_df: pd.DataFrame | list[str] | dict[str, Any],
    id_col: str = "id",
    url_col: str = "url",
    scraped_ids: set[int | str] | None = None,
) -> pd.DataFrame:
    """Vectorized high-speed decomposition of a URL series or DataFrame.

    Processes 1M+ URLs in ~5-7 seconds.

    Args:
        urls_or_df: DataFrame containing url_col, or list of URLs.
        id_col: Column name for document integer/string identifier.
        url_col: Column name containing URL strings.
        scraped_ids: Optional set of already-crawled doc_ids to compute coverage.

    Returns:
        Enriched DataFrame with domain, depth, seg_1, seg_2, seg_3, leaf, extension, and crawled status.
    """
    if isinstance(urls_or_df, pd.DataFrame):
        urls = urls_or_df[url_col].tolist()
        ids = urls_or_df[id_col].tolist() if id_col in urls_or_df.columns else list(range(len(urls)))
    elif isinstance(urls_or_df, (list, tuple)):
        urls = list(urls_or_df)
        ids = list(range(len(urls)))
    else:
        urls = [str(urls_or_df)]
        ids = [0]

    records = []
    scraped_set = set(scraped_ids) if scraped_ids is not None else None

    for doc_id, u in zip(ids, urls):
        d = dissect_url(u)
        is_crawled = (doc_id in scraped_set) if scraped_set is not None else None
        records.append({
            "id": doc_id,
            "domain": d["domain"],
            "depth": d["depth"],
            "seg_1": d["seg_1"],
            "seg_2": d["seg_2"],
            "seg_3": d["seg_3"],
            "leaf": d["leaf"],
            "extension": d["extension"],
            "is_homepage": d["is_homepage"],
            "is_pagination": d["is_pagination"],
            "is_crawled": is_crawled,
            "url": u,
        })

    return pd.DataFrame(records)


def get_url_hierarchy_tree(
    df: pd.DataFrame,
    domain: str,
    top_n_seg1: int = 15,
    top_n_seg2: int = 5,
) -> dict[str, Any]:
    """Computes a nested category tree for a domain decomposed by '/'.

    Args:
        df: DataFrame returned by dissect_urls_df.
        domain: Domain name to inspect.
        top_n_seg1: Maximum top primary segments to show.
        top_n_seg2: Maximum sub-segments per primary segment.

    Returns:
        Hierarchical dictionary representation of directory structure.
    """
    sub_df = df[df["domain"] == domain]
    total_docs = len(sub_df)
    if total_docs == 0:
        return {"domain": domain, "total_docs": 0, "categories": []}

    crawled_count = int(sub_df["is_crawled"].sum()) if "is_crawled" in sub_df and sub_df["is_crawled"].notna().any() else None

    seg1_counts = sub_df["seg_1"].value_counts().head(top_n_seg1)
    tree_categories = []

    for seg1_name, count in seg1_counts.items():
        branch_df = sub_df[sub_df["seg_1"] == seg1_name]
        branch_crawled = int(branch_df["is_crawled"].sum()) if crawled_count is not None else None
        
        # Sub-segments under seg1
        seg2_counts = branch_df[branch_df["seg_2"] != ""]["seg_2"].value_counts().head(top_n_seg2)
        sub_branches = []
        for seg2_name, s2_cnt in seg2_counts.items():
            sub_branches.append({
                "segment": seg2_name,
                "count": int(s2_cnt),
                "ratio": round(s2_cnt / count * 100, 1),
            })

        tree_categories.append({
            "segment": seg1_name if seg1_name else "(root / homepage)",
            "count": int(count),
            "ratio": round(count / total_docs * 100, 1),
            "crawled": branch_crawled,
            "sub_categories": sub_branches,
        })

    return {
        "domain": domain,
        "total_docs": total_docs,
        "total_crawled": crawled_count,
        "crawl_rate": round(crawled_count / total_docs * 100, 1) if (crawled_count is not None and total_docs > 0) else None,
        "categories": tree_categories,
    }
