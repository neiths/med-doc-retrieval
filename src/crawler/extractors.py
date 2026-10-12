"""Host-specific text and metadata extractors preserving document structure (Mentor Day 1).

Preserves:
- Article titles (H1) and section headings (H2, H3)
- Q&A pairing for medical forums (120ask, 39.net) keeping Question + Doctor Answers together
- Clean article body containers for major Vietnamese medical & pharmacy portals
- Eliminates navigation menus, appointment banners, pharmacist bios, and advertisement sidebars.
"""

import re
from typing import Any
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import trafilatura

from src.ingestion.cleaner import clean_boilerplate, fix_mojibake, normalize_text


def extract_120ask(soup: BeautifulSoup) -> dict[str, str] | None:
    """Extracts Chinese medical Q&A from www.120ask.com."""
    # 1. Title
    h1 = soup.find("h1")
    title = h1.get_text(strip=True) if h1 else ""

    # 2. Question description
    q_box = soup.find(class_="b_askbox") or soup.find(class_="crazy_new") or soup.find(class_="ms_content")
    q_text = ""
    if q_box:
        # Strip metadata buttons
        for noisy in q_box.find_all(class_=re.compile(r"(b_askbtn|b_askti|complaint)", re.I)):
            noisy.decompose()
        q_text = q_box.get_text(separator="\n", strip=True)

    # 3. Doctor answers
    answers = []
    ans_boxes = soup.find_all(class_=re.compile(r"b_answerli|b_anscont|b_ansbox"))
    for ab in ans_boxes:
        # Doctor reply body
        txt_box = ab.find(class_=re.compile(r"b_anscont|b_cont")) or ab
        # Clone to avoid mutating
        ans_soup = BeautifulSoup(str(txt_box), "html.parser")
        for tag in ans_soup(["script", "style", "a", "form", "iframe"]):
            tag.decompose()
        # Remove complaint / doctor rating buttons
        for noisy in ans_soup.find_all(class_=re.compile(r"(complaint|b_answerti|doctor_detail|good_list)", re.I)):
            noisy.decompose()
        ans_str = ans_soup.get_text(separator="\n", strip=True)
        if len(ans_str) > 15:
            answers.append(ans_str)

    if not q_text and not answers and not title:
        return None

    parts = []
    if title:
        parts.append(f"# {title}")
    if q_text:
        parts.append(f"## 患者提问\n{q_text}")
    for idx, ans in enumerate(answers, 1):
        parts.append(f"## 医生回答 {idx}\n{ans}")

    full_text = "\n\n".join(parts)
    return {"title": title, "text": fix_mojibake(full_text)}


def extract_ask39(soup: BeautifulSoup) -> dict[str, str] | None:
    """Extracts Chinese medical Q&A from ask.39.net."""
    h1 = soup.find("h1") or soup.find(class_=re.compile(r"ask_tit|txt_title"))
    title = h1.get_text(strip=True) if h1 else ""

    q_box = soup.find(class_=re.compile(r"txt_ms|txt_cure|ask_content"))
    q_text = q_box.get_text(separator="\n", strip=True) if q_box else ""

    answers = []
    ans_boxes = soup.find_all(class_=re.compile(r"show_all_ans|doc_ans|ans_content"))
    for ab in ans_boxes:
        ans_soup = BeautifulSoup(str(ab), "html.parser")
        for tag in ans_soup(["script", "style", "a", "form"]):
            tag.decompose()
        ans_str = ans_soup.get_text(separator="\n", strip=True)
        if len(ans_str) > 15:
            answers.append(ans_str)

    if not q_text and not answers:
        return None

    parts = []
    if title:
        parts.append(f"# {title}")
    if q_text:
        parts.append(f"## 患者提问\n{q_text}")
    for idx, ans in enumerate(answers, 1):
        parts.append(f"## 医生回答 {idx}\n{ans}")

    full_text = "\n\n".join(parts)
    return {"title": title, "text": fix_mojibake(full_text)}


def extract_longchau(soup: BeautifulSoup) -> dict[str, str] | None:
    """Extracts medical article from nhathuoclongchau.com.vn preserving H2/H3 sections."""
    h1 = soup.find("h1")
    title = h1.get_text(strip=True) if h1 else ""

    # Locate article body container
    container = soup.find("article") or soup.find(class_=re.compile(r"content-post|article-body|detail-content"))
    if not container:
        return None

    # Remove pharmacist intro cards, rating cards, related products
    for noisy in container.find_all(class_=re.compile(r"(author|pharmacist|rating|related|product-card|banner|breadcrumb)", re.I)):
        noisy.decompose()

    # Reconstruct with markdown headings
    parts = []
    if title:
        parts.append(f"# {title}")

    current_section = []
    for elem in container.children:
        if not getattr(elem, "name", None):
            continue
        tag_name = elem.name.lower()
        if tag_name in ["h2", "h3", "h4"]:
            if current_section:
                parts.append("\n".join(current_section))
                current_section = []
            h_prefix = "##" if tag_name == "h2" else "###"
            h_text = elem.get_text(strip=True)
            if h_text:
                parts.append(f"{h_prefix} {h_text}")
        elif tag_name in ["p", "ul", "ol", "div"]:
            text_val = elem.get_text(separator="\n", strip=True)
            if text_val:
                current_section.append(text_val)

    if current_section:
        parts.append("\n".join(current_section))

    full_text = "\n\n".join(parts)
    if len(full_text) < 100:
        return None
    return {"title": title, "text": full_text}


def extract_suckhoedoisong(soup: BeautifulSoup) -> dict[str, str] | None:
    """Extracts articles from suckhoedoisong.vn preserving sections."""
    h1 = soup.find("h1")
    title = h1.get_text(strip=True) if h1 else ""

    container = soup.find(class_=re.compile(r"detail-content|fck|article-content"))
    if not container:
        return None

    # Remove noisy widgets
    for noisy in container.find_all(class_=re.compile(r"(relate|hotline|box-share|adv|banner|author)", re.I)):
        noisy.decompose()

    parts = []
    if title:
        parts.append(f"# {title}")

    for elem in container.find_all(["h2", "h3", "p"]):
        tag = elem.name.lower()
        t = elem.get_text(strip=True)
        if not t:
            continue
        if tag == "h2":
            parts.append(f"## {t}")
        elif tag == "h3":
            parts.append(f"### {t}")
        else:
            parts.append(t)

    full_text = "\n\n".join(parts)
    if len(full_text) < 80:
        return None
    return {"title": title, "text": full_text}


def extract_vinmec(soup: BeautifulSoup) -> dict[str, str] | None:
    """Extracts articles from vinmec.com with language check (Mentor Day 1 example)."""
    # Check if page is language redirect notification
    page_text = soup.get_text()
    if "This content is not available in English" in page_text or "Yêu cầu tiếp tục để đọc tiếng Việt" in page_text:
        return None

    h1 = soup.find("h1")
    title = h1.get_text(strip=True) if h1 else ""

    container = soup.find(class_=re.compile(r"article-body|content-detail|detail-content")) or soup.find("article")
    if not container:
        return None

    # Remove appointment / hotline cards
    for noisy in container.find_all(class_=re.compile(r"(booking|doctor|contact|banner|social)", re.I)):
        noisy.decompose()

    body = container.get_text(separator="\n", strip=True)
    if len(body) < 100:
        return None

    return {"title": title, "text": f"# {title}\n\n{body}" if title else body}


def extract_article_structured(html: str, url: str) -> dict[str, str]:
    """Smart structured extractor dispatching to specialized host extractors or Trafilatura."""
    if not html:
        return {"title": "", "text": ""}

    domain = ""
    try:
        domain = urlparse(url).netloc.lower()
    except Exception:
        pass

    soup = BeautifulSoup(html, "html.parser")

    # Clean HTML script, style, comments
    for tag in soup(["script", "style", "noscript", "svg", "iframe", "form"]):
        tag.decompose()

    res = None
    if "120ask.com" in domain:
        res = extract_120ask(soup)
    elif "39.net" in domain:
        res = extract_ask39(soup)
    elif "longchau" in domain:
        res = extract_longchau(soup)
    elif "suckhoedoisong.vn" in domain:
        res = extract_suckhoedoisong(soup)
    elif "vinmec.com" in domain:
        res = extract_vinmec(soup)

    if res and len(res["text"].strip()) > 50:
        return {
            "title": res["title"].strip(),
            "text": clean_boilerplate(res["text"].strip()),
        }

    # Fallback to Trafilatura with markdown formatting to preserve headings
    try:
        extracted = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            include_formatting=True,
            no_fallback=False,
            output_format="txt",  # Trafilatura markdown-like txt with ## headings
        )
        title = ""
        meta = trafilatura.extract_metadata(html)
        if meta and meta.title:
            title = meta.title.strip()
        elif soup.find("h1"):
            title = soup.find("h1").get_text(strip=True)
        elif soup.title and soup.title.string:
            title = soup.title.string.strip()

        if extracted and len(extracted.strip()) > 50:
            return {
                "title": title,
                "text": clean_boilerplate(extracted.strip()),
            }
    except Exception:
        pass

    # Final fallback: Clean body text from soup
    h1 = soup.find("h1")
    title = h1.get_text(strip=True) if h1 else (soup.title.string.strip() if soup.title and soup.title.string else "")
    body_text = soup.get_text(separator="\n", strip=True)
    return {
        "title": title,
        "text": clean_boilerplate(body_text),
    }
