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


class DocumentChunker:
    """Splits documents into overlapping chunks while preserving exact substring spans."""

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

    def _find_split_point(self, text: str, target_end: int) -> int:
        """Finds a natural boundary (paragraph, sentence, or word) near target_end."""
        if target_end >= len(text):
            return len(text)

        search_window = text[max(0, target_end - 100) : min(len(text), target_end + 50)]
        offset = max(0, target_end - 100)

        # 1. Paragraph boundary (\n\n)
        pos = search_window.rfind("\n\n")
        if pos != -1 and offset + pos > 0:
            return offset + pos + 2

        # 2. Sentence boundary (. , ? , ! , 。, ！, ？)
        delimiters = [". ", "? ", "! ", "。\n", "。", "！", "？", "\n"]
        best_pos = -1
        for d in delimiters:
            pos = search_window.rfind(d)
            if pos > best_pos:
                best_pos = pos + len(d)
        if best_pos != -1 and offset + best_pos > 0:
            return offset + best_pos

        # 3. Space boundary
        pos = search_window.rfind(" ")
        if pos != -1 and offset + pos > 0:
            return offset + pos + 1

        return target_end

    def chunk_document(
        self,
        doc_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
        lang: str | None = None,
    ) -> list[DocumentChunk]:
        """Chunks a document text into DocumentChunk instances with exact slice tracking."""
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

        chunks: list[DocumentChunk] = []
        start = 0
        chunk_idx = 0
        total_len = len(cleaned_text)

        while start < total_len:
            raw_end = min(start + self.max_chunk_size, total_len)
            if raw_end < total_len and self.split_by_sentences:
                end = self._find_split_point(cleaned_text, raw_end)
            else:
                end = raw_end

            # Ensure progress
            if end <= start:
                end = min(start + self.max_chunk_size, total_len)

            chunk_slice = cleaned_text[start:end].strip()

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
                        char_start=start,
                        char_end=end,
                        lang=doc_lang,
                        contextual_text=ctx_text,
                        metadata=meta,
                    )
                )
                chunk_idx += 1

            if end >= total_len:
                break

            # Advance with overlap
            next_start = end - self.chunk_overlap
            if next_start <= start:
                next_start = end
            start = next_start

        return chunks
