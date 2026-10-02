"""Document chunking module tailored for multilingual medical retrieval."""

from typing import Any

from pydantic import BaseModel, Field

from src.ingestion.cleaner import detect_language, normalize_text


def build_contextual_text(
    chunk_text: str,
    title: str | None = None,
    section: str | None = None,
    lang: str = "en",
) -> str:
    """Builds an enriched contextual representation for embedding and reranking.

    Prepends document title and section information using language-specific prefixes,
    while leaving chunk_text untouched for submission.
    """
    clean_title = (title or "").strip()
    clean_section = (section or "").strip()

    if not clean_title and not clean_section:
        return chunk_text

    header_parts = []
    if lang == "vi":
        if clean_title:
            header_parts.append(f"Tiêu đề: {clean_title}")
        if clean_section:
            header_parts.append(f"Mục: {clean_section}")
        header_parts.append(f"Nội dung: {chunk_text}")
    elif lang == "zh":
        if clean_title:
            header_parts.append(f"标题: {clean_title}")
        if clean_section:
            header_parts.append(f"章节: {clean_section}")
        header_parts.append(f"内容: {chunk_text}")
    else:  # en or other
        if clean_title:
            header_parts.append(f"Title: {clean_title}")
        if clean_section:
            header_parts.append(f"Section: {clean_section}")
        header_parts.append(f"Content: {chunk_text}")

    return "\n".join(header_parts)


class DocumentChunk(BaseModel):
    chunk_id: str
    doc_id: str
    chunk_text: str
    char_start: int
    char_end: int
    lang: str = "en"
    contextual_text: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# Common medical and Vietnamese/English abbreviations to avoid false sentence splits
COMMON_ABBREVIATIONS = {
    "bs.", "ts.", "pgs.", "gs.", "ths.", "ds.", "bv.", "tp.",
    "dr.", "mr.", "mrs.", "ms.", "prof.", "vs.", "etc.", "i.e.", "e.g."
}


def split_sentences_with_spans(text: str, lang: str = "vi") -> list[tuple[int, int, str]]:
    """Splits text into complete sentences while tracking exact character offsets [start, end]."""
    if not text:
        return []

    import re

    if lang == "zh":
        pattern = re.compile(r"([。！？\n]+)")
    else:
        pattern = re.compile(r"([.!?]+(?:\s+|\n+|$)|(?:\n{1,}))")

    spans = []
    start = 0
    for match in pattern.finditer(text):
        m_start = match.start()
        m_end = match.end()
        # Avoid splitting on common abbreviations
        token_before = (
            text[max(0, m_start - 6) : m_start + 1].lower().split()[-1]
            if text[max(0, m_start - 6) : m_start + 1]
            else ""
        )
        if token_before in COMMON_ABBREVIATIONS and not match.group(0).startswith("\n"):
            continue

        segment = text[start:m_end]
        if segment.strip():
            spans.append((start, m_end, segment.strip()))
        start = m_end

    if start < len(text):
        segment = text[start:]
        if segment.strip():
            spans.append((start, len(text), segment.strip()))

    return spans


class DocumentChunker:
    """Splits documents into overlapping chunks at complete sentence boundaries while preserving exact substring spans."""

    def __init__(
        self,
        max_chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_size: int = 50,
        split_by_sentences: bool = True,
        enable_contextual: bool = True,
    ):
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.split_by_sentences = split_by_sentences
        self.enable_contextual = enable_contextual

    def chunk_document(
        self,
        doc_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
        lang: str | None = None,
    ) -> list[DocumentChunk]:
        """Chunks a document text into DocumentChunk instances with exact slice tracking.

        Guarantees that chunks never cut across words or mid-sentence.
        """
        cleaned_text = normalize_text(text)
        if not cleaned_text:
            return []

        doc_lang = lang or detect_language(cleaned_text)
        meta = metadata or {}

        # If text is already shorter than max_chunk_size, return it as a single chunk
        if len(cleaned_text) <= self.max_chunk_size:
            ctx_text = (
                build_contextual_text(
                    cleaned_text,
                    title=meta.get("title"),
                    section=meta.get("section"),
                    lang=doc_lang,
                )
                if self.enable_contextual
                else None
            )
            return [
                DocumentChunk(
                    chunk_id=f"{doc_id}__c0",
                    doc_id=doc_id,
                    chunk_text=cleaned_text,
                    char_start=0,
                    char_end=len(cleaned_text),
                    lang=doc_lang,
                    contextual_text=ctx_text,
                    metadata=meta,
                )
            ]

        # Use sentence-aware splitting to avoid cutting across sentences
        sentence_spans = split_sentences_with_spans(cleaned_text, lang=doc_lang)
        if not sentence_spans:
            return []

        chunks: list[DocumentChunk] = []
        i = 0
        n_sents = len(sentence_spans)
        chunk_idx = 0

        while i < n_sents:
            start_char = sentence_spans[i][0]
            j = i
            # Accumulate full sentences until max_chunk_size is reached
            while j < n_sents:
                cand_end_char = sentence_spans[j][1]
                cand_len = cand_end_char - start_char
                if cand_len > self.max_chunk_size and j > i:
                    break
                j += 1

            # Chunk spans from start of sentence i to end of sentence j-1
            chunk_start = sentence_spans[i][0]
            chunk_end = sentence_spans[j - 1][1]
            chunk_slice = cleaned_text[chunk_start:chunk_end].strip()

            if len(chunk_slice) >= self.min_chunk_size:
                ctx_text = (
                    build_contextual_text(
                        chunk_slice,
                        title=meta.get("title"),
                        section=meta.get("section"),
                        lang=doc_lang,
                    )
                    if self.enable_contextual
                    else None
                )
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{doc_id}__c{chunk_idx}",
                        doc_id=doc_id,
                        chunk_text=chunk_slice,
                        char_start=chunk_start,
                        char_end=chunk_end,
                        lang=doc_lang,
                        contextual_text=ctx_text,
                        metadata=meta,
                    )
                )
                chunk_idx += 1

            if j >= n_sents:
                break

            # Calculate overlap by stepping back complete sentences
            next_i = j
            overlap_accum = 0
            for k in range(j - 1, i, -1):
                sent_len = sentence_spans[k][1] - sentence_spans[k][0]
                if overlap_accum + sent_len <= self.chunk_overlap:
                    overlap_accum += sent_len
                    next_i = k
                else:
                    break

            # Ensure forward progress
            if next_i <= i:
                next_i = i + 1
            i = next_i

        return chunks
