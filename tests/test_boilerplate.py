"""Unit tests for frequency-based site boilerplate detection."""

from src.ingestion.boilerplate import (
    BoilerplateCounter,
    doc_line_hashes,
    load_table,
    save_table,
    strip_boilerplate,
)

CHROME = "Đặt lịch khám qua hotline 1900 8888"


def _doc(body: str) -> str:
    return f"{body}\n{CHROME}"


def test_line_hashes_include_footer_paragraphs_but_not_article_text():
    """A fixed byline block is one long line; article prose is even longer and unique anyway."""
    footer = "Nội dung được kiểm duyệt chặt chẽ bởi đội ngũ chuyên gia y tế theo quy trình. " * 3
    article = (
        "Bệnh nhân nam 45 tuổi nhập viện vì đau ngực trái kéo dài ba giờ, kèm khó thở và vã mồ hôi, "
        "huyết áp 90/60 mmHg, mạch 110 lần/phút, điện tâm đồ cho thấy ST chênh lên ở chuyển đạo trước "
        "ngực, men tim tăng gấp ba lần giới hạn bình thường, siêu âm tim giảm động vùng trước bên. "
        "Bệnh nhân được chẩn đoán nhồi máu cơ tim cấp và chuyển phòng can thiệp mạch vành để đặt stent "
        "trong vòng sáu mươi phút kể từ lúc nhập viện, sau đó theo dõi tại khoa hồi sức tích cực."
    )
    assert 120 < len(footer) < 400
    assert len(article) > 400
    hashes = doc_line_hashes(f"Đặt lịch\n{footer}\n{article}\n\n")
    assert len(hashes) == 2
    assert doc_line_hashes(article) == set()


def test_counter_selects_lines_shared_by_most_documents():
    counter = BoilerplateCounter(min_share=0.5, min_docs=2)
    for i in range(4):
        counter.add("benhvien.vn", _doc(f"Đoạn nội dung riêng số {i}"))
    table = counter.selected()
    assert table["benhvien.vn"] == {next(iter(doc_line_hashes(CHROME)))}


def test_counter_ignores_lines_seen_in_a_single_document():
    counter = BoilerplateCounter(min_share=0.5, min_docs=2)
    counter.add("benhvien.vn", _doc("Nội dung A"))
    counter.add("benhvien.vn", _doc("Nội dung B"))
    stripped = strip_boilerplate(_doc("Nội dung A"), counter.selected()["benhvien.vn"])
    assert "Nội dung A" in stripped
    assert CHROME not in stripped


def test_thresholds_scale_with_domain_size():
    """A line in 2 of 3 docs is chrome; in 2 of 400 it is not."""
    small = BoilerplateCounter(min_share=0.5, min_docs=2)
    for _ in range(3):
        small.add("small.vn", _doc("Nội dung"))
    assert "small.vn" in small.selected()

    large = BoilerplateCounter(min_share=0.2, min_docs=20)
    for i in range(400):
        text = _doc("Nội dung") if i < 2 else f"Nội dung khác {i}"
        large.add("large.vn", text)
    assert "large.vn" not in large.selected()


def test_strip_keeps_body_and_blank_lines():
    hashes = {next(iter(doc_line_hashes(CHROME)))}
    text = f"Đoạn một.\n\n{CHROME}\nĐoạn hai."
    stripped = strip_boilerplate(text, hashes)
    assert "Đoạn một." in stripped
    assert "Đoạn hai." in stripped
    assert CHROME not in stripped
    assert "\n\n" in stripped


def test_strip_is_a_noop_without_a_table():
    text = f"Nội dung\n{CHROME}"
    assert strip_boilerplate(text, None) == text
    assert strip_boilerplate(text, set()) == text


def test_table_roundtrip(tmp_path):
    table = {"a.vn": {1, 2, 3}, "b.vn": {2**63 + 5}}
    path = tmp_path / "boilerplate.parquet"
    save_table(table, path)
    assert load_table(path) == table
