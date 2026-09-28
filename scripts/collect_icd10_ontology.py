"""Script to harvest official Ministry of Health (Bộ Y tế) ICD-10 ontology

and match with WHO ICD-10 English clinical descriptions.

Outputs:
1. data/lexicon/icd10_ontology.json (Complete hierarchical ontology with metadata)
2. data/lexicon/icd10_vi_en.json (Bilingual mapping dictionary for retrieval)
3. Merges into data/lexicon/medical_terms.json (Enriched lexicon for QueryTranslator)
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("icd10_collector")

BASE_API = "https://ccs.whiteneuron.com/api/ICD10"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
}


def load_who_icd10(csv_path: Path) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    """Loads WHO ICD-10 codes.csv.

    Columns:
    0: Category (e.g. 'A00', 'I20', 'I210')
    1: Subcode (e.g. '0', '1', '9')
    2: Clean full code (e.g. 'A000', 'I200', 'I219')
    3: Short description
    4: Long description
    5: Category name

    Returns:
        (exact_code_map, category_map)
    """
    logger.info(f"Loading WHO ICD-10 dictionary from {csv_path}...")
    exact_map: dict[str, dict[str, str]] = {}
    cat_map: dict[str, str] = {}
    prefix_map: dict[str, str] = {}

    with open(csv_path, encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 6:
                continue
            cat = row[0].strip()
            code = row[2].strip()
            short_desc = row[3].strip()
            long_desc = row[4].strip()
            cat_name = row[5].strip()

            if cat and cat not in cat_map:
                cat_map[cat] = cat_name

            if code:
                exact_map[code] = {
                    "long": long_desc,
                    "short": short_desc,
                    "category": cat_name,
                }
                # Keep category prefix (e.g., I21 -> first or unspecified description)
                prefix3 = code[:3]
                if prefix3 not in prefix_map or code.endswith("9") or code.endswith("0"):
                    prefix_map[prefix3] = cat_name or long_desc

    # Merge prefix fallbacks into cat_map
    for k, v in prefix_map.items():
        if k not in cat_map or not cat_map[k]:
            cat_map[k] = v

    logger.info(
        f"WHO ICD-10 loaded: {len(exact_map):,} exact codes, {len(cat_map):,} categories."
    )
    return exact_map, cat_map


async def fetch_json_with_retry(
    client: httpx.AsyncClient, url: str, max_retries: int = 4
) -> dict[str, Any] | None:
    """Fetches JSON with exponential backoff."""
    for attempt in range(1, max_retries + 1):
        try:
            resp = await client.get(url)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (429, 502, 503, 504):
                await asyncio.sleep(0.5 * (2**attempt))
                continue
            logger.warning(f"HTTP {resp.status_code} for {url}")
            return None
        except (httpx.RequestError, httpx.TimeoutException) as e:
            if attempt == max_retries:
                logger.error(f"Failed to fetch {url} after {max_retries} attempts: {e}")
                return None
            await asyncio.sleep(0.5 * (2**attempt))
    return None


async def crawl_moh_icd10(
    cache_path: Path, concurrency: int = 15, force: bool = False
) -> list[dict[str, Any]]:
    """Crawls full Ministry of Health ICD-10 hierarchy.

    Hierarchy: Chapters -> Sections -> Types -> Diseases (Leaves)
    """
    if cache_path.exists() and not force:
        logger.info(f"Loading raw MOH ICD-10 data from cache: {cache_path}")
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)

    logger.info("Starting crawl of Ministry of Health ICD-10 tree from ccs.whiteneuron.com...")
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    timeout = httpx.Timeout(20.0, connect=10.0)

    all_records: list[dict[str, Any]] = []

    async with httpx.AsyncClient(limits=limits, headers=HEADERS, timeout=timeout) as client:
        # Step 1: Root chapters
        logger.info("Fetching root chapters...")
        root_data = await fetch_json_with_retry(client, f"{BASE_API}/root")
        if not root_data or "data" not in root_data:
            logger.error("Failed to fetch ICD-10 root chapters.")
            return []

        chapters = root_data["data"]
        logger.info(f"Found {len(chapters)} ICD-10 chapters.")

        # Step 2: Fetch sections for each chapter
        logger.info("Fetching sections for all chapters...")
        section_tasks = [
            fetch_json_with_retry(client, f"{BASE_API}/childs/chapter?id={ch['id']}")
            for ch in chapters
        ]
        section_results = await asyncio.gather(*section_tasks)

        all_sections: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for ch, s_res in zip(chapters, section_results):
            if s_res and "data" in s_res:
                for sec in s_res["data"]:
                    all_sections.append((ch, sec))

        logger.info(f"Found {len(all_sections)} sections across {len(chapters)} chapters.")

        # Step 3: Fetch types for each section
        logger.info("Fetching types for all sections...")
        type_sem = asyncio.Semaphore(concurrency)

        async def fetch_types_for_sec(
            ch_data: dict[str, Any], sec_data: dict[str, Any]
        ) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
            async with type_sem:
                res = await fetch_json_with_retry(
                    client, f"{BASE_API}/childs/section?id={sec_data['id']}"
                )
                if not res or "data" not in res:
                    return []
                return [(ch_data, sec_data, t) for t in res["data"]]

        type_tasks = [fetch_types_for_sec(ch, sec) for ch, sec in all_sections]
        type_nested = await asyncio.gather(*type_tasks)
        all_types = [item for sublist in type_nested for item in sublist]

        logger.info(f"Found {len(all_types)} types across all sections.")

        # Step 4: Fetch diseases (leaves) for each type
        logger.info("Fetching disease leaves for all types...")
        leaf_sem = asyncio.Semaphore(concurrency)

        async def fetch_diseases_for_type(
            ch_data: dict[str, Any], sec_data: dict[str, Any], type_data: dict[str, Any]
        ) -> list[dict[str, Any]]:
            t_id = type_data["id"]
            t_raw = type_data.get("data", {})
            t_name = t_raw.get("name", "")
            t_code = t_raw.get("code", t_id)
            ch_raw = ch_data.get("data", {})
            ch_name = ch_raw.get("name", "")
            sec_raw = sec_data.get("data", {})
            sec_name = sec_raw.get("name", "")

            # Store the type itself as an entity
            type_record = {
                "level": "type",
                "id": t_id,
                "code": t_code,
                "clean_code": t_id.replace(".", "").strip(),
                "name_vn": t_name,
                "chapter_id": ch_data["id"],
                "chapter_name": ch_name,
                "section_id": sec_data["id"],
                "section_name": sec_name,
                "parent_type_id": None,
                "parent_type_name": None,
            }

            async with leaf_sem:
                res = await fetch_json_with_retry(client, f"{BASE_API}/childs/type?id={t_id}")
                if not res or "data" not in res or not res["data"]:
                    return [type_record]

                leaf_records = [type_record]
                for leaf in res["data"]:
                    l_raw = leaf.get("data", {})
                    l_id = leaf.get("id", "")
                    l_code = l_raw.get("code", l_id)
                    l_name = l_raw.get("name", "")
                    clean_code = l_id.replace(".", "").strip()

                    leaf_records.append(
                        {
                            "level": "disease",
                            "id": l_id,
                            "code": l_code,
                            "clean_code": clean_code,
                            "name_vn": l_name,
                            "chapter_id": ch_data["id"],
                            "chapter_name": ch_name,
                            "section_id": sec_data["id"],
                            "section_name": sec_name,
                            "parent_type_id": t_id,
                            "parent_type_name": t_name,
                        }
                    )
                return leaf_records

        disease_tasks = [fetch_diseases_for_type(ch, sec, t) for ch, sec, t in all_types]
        disease_nested = await asyncio.gather(*disease_tasks)
        for sublist in disease_nested:
            all_records.extend(sublist)

    logger.info(f"Crawl completed! Total entities harvested: {len(all_records):,}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved raw crawl cache to {cache_path}")
    return all_records


def clean_vietnamese_phrase(text: str) -> str:
    """Cleans punctuation, noise words, and brackets from Vietnamese disease names."""
    # Remove text inside parentheses (often contains Latin names or codes)
    cleaned = re.sub(r"\(.*?\)", "", text)
    # Remove leading codes like A00.0 or I21
    cleaned = re.sub(r"^[A-Z][0-9]{2}(\.[0-9]+)?\s*[-:]*\s*", "", cleaned)
    # Remove special chars
    cleaned = re.sub(r"[,\.;:!?/\\\"'“”]", " ", cleaned)
    # Collapse whitespace
    cleaned = " ".join(cleaned.split()).lower()
    return cleaned


def clean_english_phrase(text: str) -> str:
    """Cleans punctuation and common qualifiers from English descriptions."""
    # Remove 'unspecified', 'other forms of', 'due to', etc.
    cleaned = re.sub(r"\b(unspecified|not elsewhere classified|without mentions? of)\b", "", text, flags=re.I)
    cleaned = re.sub(r"[,\.;:!?/\\\"'“”()]", " ", cleaned)
    cleaned = " ".join(cleaned.split())
    return cleaned


def match_and_build_lexicon(
    moh_records: list[dict[str, Any]],
    who_exact: dict[str, dict[str, str]],
    who_cats: dict[str, str],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Joins MOH Vietnamese records with WHO English definitions.

    Returns:
        (full_ontology_dict, bilingual_mapping_dict)
    """
    logger.info("Matching Vietnamese MOH records with WHO English ontology...")
    ontology_entries: list[dict[str, Any]] = []
    bilingual_map: dict[str, str] = {}

    matched_exact = 0
    matched_cat = 0
    unmatched = 0

    for rec in moh_records:
        clean_code = rec["clean_code"]
        raw_code = rec["code"]
        name_vn = rec["name_vn"].strip()
        if not name_vn:
            continue

        en_long = ""
        en_category = ""

        # Strategy 1: Exact code match in WHO
        if clean_code in who_exact:
            en_long = who_exact[clean_code]["long"]
            en_category = who_exact[clean_code]["category"]
            matched_exact += 1
        # Strategy 2: 3-char Category match
        elif clean_code[:3] in who_cats:
            en_category = who_cats[clean_code[:3]]
            en_long = en_category
            matched_cat += 1
        elif clean_code in who_cats:
            en_category = who_cats[clean_code]
            en_long = en_category
            matched_cat += 1
        else:
            unmatched += 1

        rec_entry = {
            "icd10_code": raw_code,
            "code_id": clean_code,
            "level": rec["level"],
            "name_vn": name_vn,
            "name_en": en_long or en_category or "",
            "category_en": en_category or "",
            "chapter": rec["chapter_name"],
            "parent_type": rec.get("parent_type_name"),
        }
        ontology_entries.append(rec_entry)

        # Build clean bilingual query translation pairs
        if en_long or en_category:
            clean_vn = clean_vietnamese_phrase(name_vn)
            clean_en = clean_english_phrase(en_long or en_category).lower()

            # Filter out generic or too-short noise
            if len(clean_vn) >= 3 and len(clean_en) >= 3:
                # Do not overwrite if shorter/less informative exists
                existing = bilingual_map.get(clean_vn, "")
                if existing:
                    # Append unique terms
                    merged_terms = existing.split()
                    for word in clean_en.split():
                        if word not in merged_terms:
                            merged_terms.append(word)
                    bilingual_map[clean_vn] = " ".join(merged_terms)
                else:
                    bilingual_map[clean_vn] = clean_en

    logger.info(
        f"Ontology construction summary: "
        f"{matched_exact:,} exact WHO matches, {matched_cat:,} category matches, "
        f"{unmatched:,} unmatched. Total bilingual dictionary entries: {len(bilingual_map):,}"
    )

    full_ontology = {
        "metadata": {
            "source_vi": "Cục Quản lý Khám, chữa bệnh - Bộ Y tế Việt Nam (icd.kcb.vn)",
            "source_en": "World Health Organization (WHO) ICD-10 Classification",
            "total_entities": len(ontology_entries),
            "matched_exact": matched_exact,
            "matched_category": matched_cat,
        },
        "entries": ontology_entries,
    }

    return full_ontology, bilingual_map


def enrich_medical_terms_lexicon(
    existing_lexicon_path: Path, new_bilingual_map: dict[str, str], output_path: Path
) -> int:
    """Enriches data/lexicon/medical_terms.json with official ICD-10 bilingual entities.

    Preserves all existing curated terms and expands coverage.
    """
    logger.info(f"Enriching existing lexicon at {existing_lexicon_path}...")
    existing_terms: dict[str, str] = {}
    if existing_lexicon_path.exists():
        with open(existing_lexicon_path, encoding="utf-8") as f:
            existing_terms = json.load(f)

    logger.info(f"Existing curated terms: {len(existing_terms)}")

    added_count = 0
    # First priority: keep existing manual curated mappings intact
    combined = dict(existing_terms)

    for vn_term, en_term in new_bilingual_map.items():
        if vn_term not in combined:
            combined[vn_term] = en_term
            added_count += 1
        else:
            # Merge English synonyms without duplicate words
            cur = combined[vn_term].split()
            for w in en_term.split():
                if w.lower() not in [x.lower() for x in cur]:
                    cur.append(w)
            combined[vn_term] = " ".join(cur)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    logger.info(
        f"Enriched medical terms saved to {output_path} (Total: {len(combined):,} terms, +{added_count:,} new terms)."
    )
    return len(combined)


async def async_main(args: argparse.Namespace) -> None:
    who_csv = Path(args.who_csv)
    out_dir = Path(args.output_dir)
    cache_path = out_dir / "icd10_moh_raw.json"
    ontology_path = out_dir / "icd10_ontology.json"
    bilingual_map_path = out_dir / "icd10_vi_en.json"
    medical_terms_path = out_dir / "medical_terms.json"

    # Step 1: Load WHO English definitions
    who_exact, who_cats = load_who_icd10(who_csv)

    # Step 2: Crawl Vietnamese Ministry of Health tree
    moh_records = await crawl_moh_icd10(
        cache_path=cache_path, concurrency=args.concurrency, force=args.force_recrawl
    )

    if not moh_records:
        logger.error("No records collected from MOH ICD-10 API. Exiting.")
        sys.exit(1)

    # Step 3: Match and build ontology + bilingual dictionary
    full_ontology, bilingual_map = match_and_build_lexicon(moh_records, who_exact, who_cats)

    # Step 4: Write output files
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(ontology_path, "w", encoding="utf-8") as f:
        json.dump(full_ontology, f, ensure_ascii=False, indent=2)
    logger.info(f"Full ontology saved to {ontology_path}")

    with open(bilingual_map_path, "w", encoding="utf-8") as f:
        json.dump(bilingual_map, f, ensure_ascii=False, indent=2)
    logger.info(f"Bilingual query dictionary saved to {bilingual_map_path}")

    # Step 5: Enrich medical_terms.json
    enrich_medical_terms_lexicon(medical_terms_path, bilingual_map, medical_terms_path)

    logger.info("ICD-10 bilingual harvesting and integration successfully finished!")


def main():
    parser = argparse.ArgumentParser(
        description="Harvest official ICD-10 bilingual medical ontology."
    )
    parser.add_argument(
        "--who-csv",
        type=str,
        default="data/lexicon/who_icd10_codes.csv",
        help="Path to WHO ICD-10 CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/lexicon",
        help="Output directory for lexicon files.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=15,
        help="Max concurrent async HTTP requests.",
    )
    parser.add_argument(
        "--force-recrawl",
        action="store_true",
        help="Force re-crawl from MOH portal ignoring raw cache.",
    )
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
