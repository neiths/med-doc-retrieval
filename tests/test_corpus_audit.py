"""Unit tests for corpus audit signals and verdict ordering."""

from src.ingestion.corpus_audit import (
    decide_verdict,
    head_fingerprint,
    is_binary,
    is_template_title,
    language_mismatch,
    looks_like_payload,
    medical_hits,
    resolve_duplicates,
    short_line_char_ratio,
    template_title_kind,
)

VOCAB = {"vi": {"viêm", "phổi", "điều", "trị"}, "en": {"pneumonia", "treatment"}}


def _signals(**overrides):
    base = {
        "doc_id": "1",
        "status": "success",
        "is_binary": False,
        "payload_like": False,
        "char_count": 5000,
        "template_title": "",
        "lang_mismatch": False,
        "med_hits": 10,
        "med_density": 3.0,
        "short_line_ratio": 0.1,
    }
    base.update(overrides)
    return base


def _verdict(signals, dup=None):
    return decide_verdict(
        signals,
        dup,
        min_chars=400,
        min_chars_hard=200,
        min_med_density=0.0,
        max_short_line_ratio=0.6,
    )


def test_binary_detection_catches_zip_and_pdf_payloads():
    assert is_binary("PK\x03\x04\x14\x00\x06\x00[Content_Types].xml")
    assert is_binary("%PDF-1.7\n1 0 obj")
    assert not is_binary("Bệnh nhân nam 45 tuổi nhập viện vì đau ngực.")


def test_short_line_ratio_separates_listing_from_article():
    listing = "\n".join(["Trang chủ", "Giới thiệu", "Liên hệ", "Tin tức", "Sức khỏe"] * 20)
    article = "\n".join(
        ["Bệnh nhân nam 45 tuổi nhập viện vì đau ngực trái kéo dài ba giờ, kèm khó thở."] * 5
    )
    assert short_line_char_ratio(listing) > 0.9
    assert short_line_char_ratio(article) < 0.1


def test_template_titles():
    assert is_template_title("Trang chủ")
    assert is_template_title("疾病详情-疾病-人卫临床助手-人民卫生出版社")
    assert not is_template_title("Ung thư đại tràng giai đoạn 2")
    assert template_title_kind("trang chủ") == "exact"
    assert template_title_kind("疾病详情-疾病-人卫临床助手-人民卫生出版社") == "pattern"
    assert template_title_kind("Ung thư đại tràng giai đoạn 2") == ""


def test_pattern_title_only_drops_contentless_pages():
    # A site-wide label over a real article must not delete the article.
    assert _verdict(_signals(template_title="pattern")) == ("review", ["template_title"])
    assert _verdict(_signals(template_title="pattern", med_hits=0))[0] == "drop"
    assert _verdict(_signals(template_title="exact", med_hits=10))[0] == "drop"


def test_short_documents_are_reviewed_not_dropped_until_they_are_fragments():
    assert _verdict(_signals(char_count=320)) == ("review", ["too_short"])
    assert _verdict(_signals(char_count=150)) == ("drop", ["fragment"])


def test_medical_hits_per_language():
    assert medical_hits("Điều trị viêm phổi ở trẻ em", "vi", VOCAB) == 4
    assert medical_hits("社区获得性肺炎的治疗", "zh", VOCAB) >= 2
    assert medical_hits("pneumonia treatment guidelines", "en", VOCAB) == 2
    assert medical_hits("Đặt tên bé trai hay và ý nghĩa", "vi", VOCAB) == 0


def test_classical_chinese_counts_as_medical():
    classical = "【圣惠】治小儿肺脏伤冷，鼻流清涕。前胡散方：前胡、陈皮、白茯苓、桂心、白术、人参。"
    assert medical_hits(classical, "zh", VOCAB) > 0
    modern_only = "本公司提供电脑维修服务，价格优惠。"
    assert medical_hits(modern_only, "zh", VOCAB) == 0


def test_duplicates_keep_longest_then_smallest_id():
    doc_ids = ["b", "a", "c", "d"]
    sizes = [900, 900, 500, 800]
    prints = ["h1", "h1", "h1", "h2"]
    canonical = resolve_duplicates(doc_ids, sizes, prints)
    assert canonical["b"] == "a"
    assert canonical["c"] == "a"
    assert "d" not in canonical


def test_head_fingerprint_distinguishes_documents():
    a = head_fingerprint("Viêm phổi là tình trạng nhiễm trùng nhu mô phổi.")
    b = head_fingerprint("Đặt tên bé trai: 1001+ cách đặt tên hay cho bé trai.")
    assert a != b
    assert a == head_fingerprint("  Viêm   phổi là tình trạng nhiễm trùng nhu mô phổi.  ")


def test_drop_reasons_precede_review_reasons():
    verdict, reasons = _verdict(_signals(is_binary=True, template_title="exact", med_density=0.0))
    assert verdict == "drop"
    assert reasons == ["binary_blob", "template_title"]


def test_duplicate_is_dropped_even_when_otherwise_clean():
    verdict, reasons = _verdict(_signals(), dup="99")
    assert verdict == "drop"
    assert reasons == ["duplicate_of:99"]


def test_crawl_failure_drops_document():
    verdict, reasons = _verdict(_signals(status="timeout"))
    assert verdict == "drop"
    assert reasons == ["crawl_status:timeout"]


def test_review_reasons_are_reported_together_and_keep_is_empty():
    assert _verdict(_signals(med_density=0.0, short_line_ratio=0.9)) == (
        "review",
        ["no_medical_terms", "listing_page"],
    )
    assert _verdict(_signals()) == ("keep", [])


def test_payload_detection_flags_encoded_blobs_only():
    blob = (
        "w53huq0lPTnhurkow7PDs+G7pCNtw5LhuqnDsuG6ucOgI+G7suG7tyk5PcOzxJE9OXvhur3huq09OCnDqj3hu6NiPc"
    )
    assert looks_like_payload(blob * 4)
    assert not looks_like_payload("Bệnh nhân nam 45 tuổi nhập viện vì đau ngực trái." * 5)
    assert not looks_like_payload("")


def test_language_mismatch_catches_mislabelled_documents():
    vi_text = "Bệnh nhân nam 45 tuổi nhập viện vì đau ngực trái kéo dài ba giờ, kèm khó thở."
    zh_text = "患者男性，45岁，因胸痛三小时入院，伴有呼吸困难。"
    assert language_mismatch(vi_text, "zh")
    assert language_mismatch(zh_text, "vi")
    assert not language_mismatch(vi_text, "vi")
    assert not language_mismatch(zh_text, "zh")


def test_payload_and_lang_mismatch_land_in_the_right_bucket():
    assert _verdict(_signals(payload_like=True)) == ("drop", ["encoded_payload"])
    assert _verdict(_signals(lang_mismatch=True)) == ("review", ["lang_mismatch"])
