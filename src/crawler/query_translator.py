"""Query translation and biomedical keyword extraction from Vietnamese to English."""

import re

import torch
from loguru import logger

# Common medical stopwords to strip from translated queries for PubMed search
PUBMED_STOPWORDS = {
    "what",
    "should",
    "be",
    "done",
    "how",
    "to",
    "treat",
    "treatment",
    "of",
    "for",
    "the",
    "a",
    "an",
    "is",
    "are",
    "can",
    "with",
    "in",
    "on",
    "and",
    "or",
    "about",
    "patient",
    "patients",
    "causes",
    "caused",
    "by",
    "there",
    "any",
    "do",
    "does",
}

# Domain dictionary mapping common Vietnamese clinical phrases directly to MeSH/English
VI_EN_MEDICAL_LEXICON = {
    "sỏi thận": "kidney stones nephrolithiasis",
    "sỏi niệu quản": "ureteral calculi",
    "tắc nghẽn đường tiết niệu": "urinary tract obstruction",
    "đau quặn thận": "renal colic",
    "suy thận": "renal failure",
    "viêm gan": "hepatitis",
    "tiểu đường": "diabetes mellitus",
    "tăng huyết áp": "hypertension",
    "ung thư": "cancer neoplasm",
    "nhiễm trùng đường tiết niệu": "urinary tract infection",
    "tán sỏi ngoài cơ thể": "extracorporeal shock wave lithotripsy",
    "nội soi tán sỏi": "ureteroscopy lithotripsy",
}


class QueryTranslator:
    """Translates Vietnamese queries to English and extracts biomedical keywords for PubMed."""

    def __init__(
        self,
        model_name: str = "Helsinki-NLP/opus-mt-vi-en",
        device: str = "auto",
    ):
        self.model_name = model_name
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._tokenizer = None
        self._model = None

    def _load_model(self):
        """Lazy loader for MarianMT model."""
        if self._model is None or self._tokenizer is None:
            from transformers import MarianMTModel, MarianTokenizer

            logger.info(f"Loading translation model weights from {self.model_name}...")
            self._tokenizer = MarianTokenizer.from_pretrained(self.model_name)
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self._model = MarianMTModel.from_pretrained(self.model_name, torch_dtype=dtype)
            self._model.to(self.device)
            self._model.eval()

    def translate_to_english(self, vi_text: str) -> str:
        """Translates a Vietnamese query string to English."""
        if not vi_text or not vi_text.strip():
            return ""

        try:
            self._load_model()
            inputs = self._tokenizer([vi_text], return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                generated_tokens = self._model.generate(**inputs, max_length=128)
            translated = self._tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            return translated.strip()
        except Exception as e:
            logger.warning(
                f"Translation failed for '{vi_text}': {e}. Falling back to lexicon extraction."
            )
            return ""

    def extract_pubmed_keywords(self, vi_query: str) -> str:
        """Extracts concise English keywords suitable for PubMed ESearch / Europe PMC API.

        Combines:
        1. Direct domain lexicon match.
        2. MarianMT translation with stopword filtering.
        """
        lexicon_terms = []
        vi_lower = vi_query.lower()
        for vi_phrase, en_phrase in VI_EN_MEDICAL_LEXICON.items():
            if vi_phrase in vi_lower:
                lexicon_terms.append(en_phrase)

        translated = self.translate_to_english(vi_query)
        keywords = []

        if translated:
            words = re.findall(r"[a-zA-Z]+", translated.lower())
            filtered = [w for w in words if w not in PUBMED_STOPWORDS and len(w) > 2]
            keywords.extend(filtered)

        # Merge lexicon terms and translated keywords (preserving order, removing duplicates)
        seen = set()
        merged = []
        for term in lexicon_terms:
            for sub in term.split():
                if sub.lower() not in seen and len(sub) > 2:
                    seen.add(sub.lower())
                    merged.append(sub)

        for w in keywords:
            if w not in seen:
                seen.add(w)
                merged.append(w)

        if not merged:
            return translated or vi_query

        return " ".join(merged[:8])
