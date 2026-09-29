import json
import re
from pathlib import Path

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


def load_lexicon(path: Path | str) -> dict[str, str]:
    """Loads bilingual medical lexicon from a JSON file."""
    path = Path(path)
    if not path.exists():
        logger.warning(f"Lexicon file {path} not found. Using defaults.")
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return {k.strip().lower(): v.strip() for k, v in data.items() if k.strip()}
    except Exception as e:
        logger.error(f"Failed to load lexicon from {path}: {e}")
        return {}


def load_acronyms(path: Path | str) -> dict[str, dict[str, str]]:
    """Loads clinical acronyms dictionary from a JSON file."""
    path = Path(path)
    if not path.exists():
        logger.warning(f"Acronyms file {path} not found. Using defaults.")
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        result = {}
        for k, v in data.items():
            k_clean = k.strip()
            if not k_clean:
                continue
            if isinstance(v, dict):
                result[k_clean] = {
                    lang: val.strip() for lang, val in v.items() if isinstance(val, str)
                }
            elif isinstance(v, str):
                result[k_clean] = {"en": v.strip()}
        return result
    except Exception as e:
        logger.error(f"Failed to load acronyms from {path}: {e}")
        return {}


def load_stopwords(path: Path | str) -> set[str]:
    """Loads stopword list from a text file (one word/phrase per line)."""
    path = Path(path)
    if not path.exists():
        logger.warning(f"Stopwords file {path} not found. Using defaults.")
        return set()
    stopwords = set()
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    stopwords.add(line.lower())
        return stopwords
    except Exception as e:
        logger.error(f"Failed to load stopwords from {path}: {e}")
        return set()


class QueryTranslator:
    """Translates Vietnamese queries to English and extracts biomedical keywords for PubMed."""

    def __init__(
        self,
        model_name: str = "Helsinki-NLP/opus-mt-vi-en",
        prompt_prefix: str = "",
        device: str = "auto",
        lexicon_path: Path | str | None = "data/lexicon/medical_terms.json",
        zh_lexicon_path: Path | str | None = "data/lexicon/icd10_vi_zh.json",
        acronyms_path: Path | str | None = "data/lexicon/medical_acronyms.json",
        stopwords_path: Path | str | None = "configs/pubmed_stopwords.txt",
    ):
        self.model_name = model_name
        self.prompt_prefix = prompt_prefix
        # Auto-detect T5 / ndhieu models if prompt_prefix is empty
        if not self.prompt_prefix and any(k in model_name.lower() for k in ["t5", "ndhieu"]):
            self.prompt_prefix = "vi: "

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._tokenizer = None
        self._model = None

        # Initialize base lexicon, acronyms and stopwords
        self.lexicon: dict[str, str] = dict(VI_EN_MEDICAL_LEXICON)
        self.zh_lexicon: dict[str, str] = {}
        self.acronyms: dict[str, dict[str, str]] = {}
        self.stopwords: set[str] = set(PUBMED_STOPWORDS)

        if lexicon_path:
            loaded_lexicon = load_lexicon(lexicon_path)
            self.lexicon.update(loaded_lexicon)
            logger.debug(
                f"Loaded {len(loaded_lexicon)} terms from {lexicon_path}. Total: {len(self.lexicon)}"
            )

        if zh_lexicon_path:
            loaded_zh = load_lexicon(zh_lexicon_path)
            self.zh_lexicon.update(loaded_zh)
            logger.debug(
                f"Loaded {len(loaded_zh)} Chinese medical terms from {zh_lexicon_path}."
            )

        if acronyms_path:
            loaded_acronyms = load_acronyms(acronyms_path)
            self.acronyms.update(loaded_acronyms)
            logger.debug(
                f"Loaded {len(loaded_acronyms)} medical acronyms from {acronyms_path}."
            )

        if stopwords_path:
            loaded_stopwords = load_stopwords(stopwords_path)
            self.stopwords.update(loaded_stopwords)
            logger.debug(
                f"Loaded {len(loaded_stopwords)} stopwords from {stopwords_path}. Total: {len(self.stopwords)}"
            )


    def _load_model(self):
        """Lazy loader for sequence-to-sequence translation models (MarianMT, T5, ViT5)."""
        if self._model is None or self._tokenizer is None:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            logger.info(f"Loading translation model weights from {self.model_name}...")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, torch_dtype=dtype)
            self._model.to(self.device)
            self._model.eval()

    def translate_to_english(self, vi_text: str) -> str:
        """Translates a Vietnamese query string to English."""
        if not vi_text or not vi_text.strip():
            return ""

        try:
            self._load_model()
            input_text = f"{self.prompt_prefix}{vi_text}" if self.prompt_prefix else vi_text
            inputs = self._tokenizer(
                [input_text], return_tensors="pt", padding=True, truncation=True
            )
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

    def expand_acronyms(self, query: str) -> dict[str, str]:
        """Expands clinical acronyms (e.g. HFrEF, STEMI, COPD, CURB-65, rt-PA) into VI, EN, and ZH medical terms."""
        if not query or not self.acronyms:
            return {"vi": "", "en": "", "zh": ""}

        found_vi = []
        found_en = []
        found_zh = []
        for acr, exps in self.acronyms.items():
            if isinstance(exps, dict):
                pattern = rf"(?i)(?:\b|_|\(){re.escape(acr)}(?:\b|_|\))"
                if re.search(pattern, query):
                    if exps.get("vi"):
                        found_vi.append(exps["vi"])
                    if exps.get("en"):
                        found_en.append(exps["en"])
                    if exps.get("zh"):
                        found_zh.append(exps["zh"])

        return {
            "vi": " ".join(found_vi),
            "en": " ".join(found_en),
            "zh": " ".join(found_zh),
        }

    def extract_pubmed_keywords(self, vi_query: str) -> str:
        """Extracts concise English keywords suitable for PubMed ESearch / Europe PMC API.

        Combines:
        1. Clinical acronym expansions (HFrEF -> heart failure with reduced ejection fraction).
        2. Direct domain lexicon match (sorted by length descending).
        3. MarianMT translation with stopword filtering.
        """
        lexicon_terms = []
        vi_lower = vi_query.lower()

        # 1. Acronym expansion
        acr_exp = self.expand_acronyms(vi_query)
        if acr_exp["en"]:
            lexicon_terms.append(acr_exp["en"])

        # 2. Domain lexicon match (longest phrases first)
        sorted_phrases = sorted(self.lexicon.keys(), key=len, reverse=True)
        for vi_phrase in sorted_phrases:
            if vi_phrase in vi_lower:
                lexicon_terms.append(self.lexicon[vi_phrase])

        translated = self.translate_to_english(vi_query)
        keywords = []

        if translated:
            words = re.findall(r"[a-zA-Z]+", translated.lower())
            filtered = [w for w in words if w not in self.stopwords and len(w) > 2]
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

    def extract_chinese_keywords(self, vi_query: str, max_terms: int = 5) -> str:
        """Extracts Chinese medical keywords from a Vietnamese query using ICD-10-CN and acronym lexicon.

        Matches clinical entities against ICD-10-CN trilingual lexicon (longest-phrase first).
        """
        if not vi_query:
            return ""

        matched = []
        seen = set()

        # 1. Acronym expansion for Chinese
        acr_exp = self.expand_acronyms(vi_query)
        if acr_exp["zh"]:
            for term in acr_exp["zh"].split():
                if term not in seen:
                    seen.add(term)
                    matched.append(term)

        # 2. Lexicon matching
        if self.zh_lexicon:
            vi_lower = vi_query.lower()
            sorted_phrases = sorted(self.zh_lexicon.keys(), key=len, reverse=True)
            for phrase in sorted_phrases:
                if phrase in vi_lower:
                    zh_term = self.zh_lexicon[phrase]
                    if zh_term not in seen:
                        seen.add(zh_term)
                        matched.append(zh_term)
                    if len(matched) >= max_terms:
                        break

        return " ".join(matched[:max_terms])

