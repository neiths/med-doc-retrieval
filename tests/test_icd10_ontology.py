import json
from pathlib import Path

from scripts.collect_icd10_ontology import (
    clean_english_phrase,
    clean_vietnamese_phrase,
    enrich_medical_terms_lexicon,
    match_and_build_lexicon,
)


def test_clean_phrases():
    # Vietnamese phrase cleaning
    assert clean_vietnamese_phrase("A00.0 Bệnh tả do Vibrio cholerae 01") == "bệnh tả do vibrio cholerae 01"
    assert clean_vietnamese_phrase("Sỏi thận (Calculus of kidney)") == "sỏi thận"
    assert clean_vietnamese_phrase("I21.9 Nhồi máu cơ tim cấp, không đặc hiệu") == "nhồi máu cơ tim cấp không đặc hiệu"

    # English phrase cleaning
    assert clean_english_phrase("Acute myocardial infarction, unspecified") == "Acute myocardial infarction"
    assert clean_english_phrase("Pneumonia, unspecified organism") == "Pneumonia organism"


def test_match_and_build_lexicon(tmp_path):
    moh_records = [
        {
            "level": "disease",
            "id": "N200",
            "code": "N20.0",
            "clean_code": "N200",
            "name_vn": "Sỏi thận",
            "chapter_id": "N00-N99",
            "chapter_name": "Bệnh hệ tiết niệu",
            "section_id": "N20-N23",
            "section_name": "Sỏi thận và sỏi niệu quản",
            "parent_type_id": "N20",
            "parent_type_name": "Sỏi thận và sỏi niệu quản",
        },
        {
            "level": "type",
            "id": "I21",
            "code": "I21",
            "clean_code": "I21",
            "name_vn": "Nhồi máu cơ tim cấp",
            "chapter_id": "I00-I99",
            "chapter_name": "Bệnh hệ tuần hoàn",
            "section_id": "I20-I25",
            "section_name": "Bệnh tim thiếu máu cục bộ",
            "parent_type_id": None,
            "parent_type_name": None,
        },
    ]

    who_exact = {
        "N200": {
            "long": "Calculus of kidney",
            "short": "Calculus of kidney",
            "category": "Calculus of kidney and ureter",
        }
    }
    who_cats = {
        "I21": "Acute myocardial infarction"
    }

    ontology, bilingual_map = match_and_build_lexicon(moh_records, who_exact, who_cats)

    assert ontology["metadata"]["total_entities"] == 2
    assert ontology["metadata"]["matched_exact"] == 1
    assert ontology["metadata"]["matched_category"] == 1
    assert "sỏi thận" in bilingual_map
    assert "calculus" in bilingual_map["sỏi thận"]
    assert "nhồi máu cơ tim cấp" in bilingual_map
    assert "acute myocardial infarction" in bilingual_map["nhồi máu cơ tim cấp"]


def test_enrich_medical_terms_lexicon(tmp_path):
    existing_file = tmp_path / "medical_terms.json"
    existing_file.write_text(
        json.dumps({"sonde jj": "double J stent", "sỏi thận": "kidney stones"}),
        encoding="utf-8",
    )

    new_map = {
        "sỏi thận": "calculus of kidney nephrolithiasis",
        "nhồi máu cơ tim": "myocardial infarction",
    }

    out_file = tmp_path / "enriched.json"
    total = enrich_medical_terms_lexicon(existing_file, new_map, out_file)

    with open(out_file, encoding="utf-8") as f:
        data = json.load(f)

    assert total == 3
    assert data["sonde jj"] == "double J stent"
    assert "kidney stones" in data["sỏi thận"]
    assert "calculus" in data["sỏi thận"]
    assert data["nhồi máu cơ tim"] == "myocardial infarction"


def test_harvested_lexicon_integrity():
    lexicon_file = Path("data/lexicon/medical_terms.json")
    assert lexicon_file.exists(), "medical_terms.json must exist"

    with open(lexicon_file, encoding="utf-8") as f:
        data = json.load(f)

    # Must contain thousands of terms from ICD-10
    assert len(data) >= 5000, f"Expected >= 5000 terms, got {len(data)}"
    assert "sỏi thận" in data
    assert "nhồi máu cơ tim" in data
