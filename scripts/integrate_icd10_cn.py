"""Integration script for ICD-10-CN Chinese medical ontology.

Combines ICD-10-CN (disease.csv, disease_catalog.csv) with Vietnam MOH ICD-10
to build a trilingual medical ontology (VI-EN-ZH) and a direct VI -> ZH query dictionary.

Outputs:
1. data/lexicon/icd10_vi_zh.json (Direct VI -> ZH medical lexicon for retrieval)
2. data/lexicon/icd10_ontology.json (Updated with name_cn, category_cn, chapter_cn)
"""

import csv
import json
import logging
from pathlib import Path
from typing import Any

from scripts.collect_icd10_ontology import clean_vietnamese_phrase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("icd10_cn_integrator")

# Common Vietnamese clinical phrases mapped directly to Chinese medical terminology
VI_ZH_CLINICAL_LEXICON = {
    "sỏi thận": "肾结石",
    "sỏi niệu quản": "输尿管结石",
    "sỏi bàng quang": "膀胱结石",
    "tắc nghẽn đường tiết niệu": "尿路梗阻",
    "đau quặn thận": "肾绞痛",
    "suy thận": "肾衰竭",
    "suy thận cấp": "急性肾衰竭",
    "suy thận mạn": "慢性肾衰竭",
    "viêm gan": "肝炎",
    "viêm gan b": "乙型肝炎",
    "viêm gan c": "丙型肝炎",
    "tiểu đường": "糖尿病",
    "đái tháo đường": "糖尿病",
    "tăng huyết áp": "高血压",
    "cao huyết áp": "高血压",
    "hạ huyết áp": "低血压",
    "ung thư": "癌症 肿瘤",
    "ung thư phổi": "肺癌",
    "ung thư gan": "肝癌",
    "ung thư dạ dày": "胃癌",
    "nhiễm trùng đường tiết niệu": "尿路感染",
    "tán sỏi ngoài cơ thể": "体外冲击波碎石术",
    "nội soi tán sỏi": "输尿管镜碎石术",
    "viêm phổi": "肺炎",
    "viêm ruột thừa": "阑尾炎",
    "nhồi máu cơ tim": "心肌梗死",
    "đột quỵ": "中风 脑卒中",
    "tai biến mạch máu não": "脑血管意外",
    "xơ gan": "肝硬化",
    "loét dạ dày": "胃溃疡",
    "viêm phế quản": "支气管炎",
    "hen suyễn": "哮喘",
    "suy tim": "心力衰竭",
}


def load_icd10_cn_data(
    disease_csv: Path | str,
    catalog_csv: Path | str,
) -> tuple[dict[str, str], list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    """Loads disease.csv and disease_catalog.csv from ICD-10-CN repository."""
    d_path = Path(disease_csv)
    c_path = Path(catalog_csv)

    if not d_path.exists():
        raise FileNotFoundError(f"Missing {d_path}. Please download it first.")
    if not c_path.exists():
        raise FileNotFoundError(f"Missing {c_path}. Please download it first.")

    # 1. Disease map: code -> disease name
    disease_map: dict[str, str] = {}
    with open(d_path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        _ = next(reader, None)  # header
        for r in reader:
            if len(r) >= 2 and r[0].strip() and r[1].strip():
                disease_map[r[0].strip().upper()] = r[1].strip()

    # 2. Catalog ranges: (lower_bound, upper_bound, catalog_name)
    level1_ranges: list[tuple[str, str, str]] = []
    level2_ranges: list[tuple[str, str, str]] = []

    with open(c_path, encoding="utf-8") as f:
        reader_dict = csv.DictReader(f, delimiter="\t")
        for r in reader_dict:
            low = r.get("code_lower_bound", "").strip().upper()
            high = r.get("code_upper_bound", "").strip().upper()
            lvl = r.get("level", "").strip()
            name = r.get("catalog", "").strip()
            if low and high and name:
                if lvl == "1":
                    level1_ranges.append((low, high, name))
                elif lvl == "2":
                    level2_ranges.append((low, high, name))

    logger.info(
        f"Loaded ICD-10-CN: {len(disease_map)} 3-char codes, "
        f"{len(level1_ranges)} chapters, {len(level2_ranges)} subchapters."
    )
    return disease_map, level1_ranges, level2_ranges


def find_catalog_name(code: str, ranges: list[tuple[str, str, str]]) -> str:
    """Finds the catalog/chapter name matching an ICD-10 code range."""
    code3 = code[:3].upper()
    for low, high, name in ranges:
        if low <= code3 <= high:
            return name
    return ""


def enrich_ontology_and_build_lexicon(
    ontology_path: Path | str = "data/lexicon/icd10_ontology.json",
    disease_csv: Path | str = "data/lexicon/icd10_cn/disease.csv",
    catalog_csv: Path | str = "data/lexicon/icd10_cn/disease_catalog.csv",
    output_vi_zh_path: Path | str = "data/lexicon/icd10_vi_zh.json",
) -> tuple[dict[str, Any], dict[str, str]]:
    """Enriches icd10_ontology.json with Chinese terms and produces icd10_vi_zh.json."""
    disease_map, level1_ranges, level2_ranges = load_icd10_cn_data(disease_csv, catalog_csv)

    ont_file = Path(ontology_path)
    with open(ont_file, encoding="utf-8") as f:
        ontology_data = json.load(f)

    entries = ontology_data.get("entries", [])
    logger.info(f"Enriching {len(entries)} entries in {ont_file} with Chinese terms...")

    matched_names = 0
    matched_cats = 0
    vi_zh_map: dict[str, str] = dict(VI_ZH_CLINICAL_LEXICON)

    for item in entries:
        raw_code = item.get("icd10_code", "")
        clean_code = raw_code.replace("†", "").replace("*", "").strip().upper()
        prefix3 = clean_code[:3]

        name_cn = disease_map.get(clean_code) or disease_map.get(prefix3) or ""
        cat2_cn = find_catalog_name(clean_code, level2_ranges)
        cat1_cn = find_catalog_name(clean_code, level1_ranges)

        if name_cn:
            matched_names += 1
        if cat2_cn:
            matched_cats += 1

        item["name_cn"] = name_cn
        item["category_cn"] = cat2_cn
        item["chapter_cn"] = cat1_cn

        # Extract Vietnamese clean phrase
        raw_vn = item.get("name_vn", "")
        clean_vn = clean_vietnamese_phrase(raw_vn)

        target_zh = name_cn or cat2_cn
        if clean_vn and target_zh and len(clean_vn) >= 3:
            if clean_vn not in vi_zh_map:
                vi_zh_map[clean_vn] = target_zh

    # Update metadata
    if "metadata" not in ontology_data:
        ontology_data["metadata"] = {}
    ontology_data["metadata"]["source_cn"] = "ICD-10-CN (chaseliu/ICD-10-CN)"
    ontology_data["metadata"]["matched_name_cn"] = matched_names
    ontology_data["metadata"]["matched_category_cn"] = matched_cats
    ontology_data["metadata"]["total_vi_zh_pairs"] = len(vi_zh_map)

    logger.info(
        f"Enrichment complete: {matched_names}/{len(entries)} ({(matched_names/len(entries))*100:.1f}%) "
        f"matched Chinese names. Total VI-ZH mappings: {len(vi_zh_map)}"
    )

    # Save updated ontology
    with open(ont_file, "w", encoding="utf-8") as f:
        json.dump(ontology_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved trilingual ontology to {ont_file}")

    # Save VI -> ZH lexicon
    out_vi_zh = Path(output_vi_zh_path)
    out_vi_zh.parent.mkdir(parents=True, exist_ok=True)
    with open(out_vi_zh, "w", encoding="utf-8") as f:
        json.dump(vi_zh_map, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved VI -> ZH lexicon to {out_vi_zh}")

    return ontology_data, vi_zh_map


if __name__ == "__main__":
    enrich_ontology_and_build_lexicon()
