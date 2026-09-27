"""Official competition evaluation metrics (Precision, Recall, F2 macro).

According to Road to AI 2026 guidelines:
- Evaluated at both Document level and Chunk level.
- F2 score weights Recall higher than Precision:
    F2 = (5 * Precision * Recall) / (4 * Precision + Recall)
- Macro-average computed across all test queries.
"""

from typing import Any

import numpy as np


def compute_prf(
    retrieved: set[Any], relevant: set[Any], beta: float = 2.0
) -> tuple[float, float, float]:
    """Computes Precision, Recall, and F_beta score between retrieved and ground-truth sets.

    Args:
        retrieved: Set of retrieved identifiers/chunks.
        relevant: Set of ground-truth relevant identifiers/chunks.
        beta: Weight of recall relative to precision (default 2.0).

    Returns:
        (precision, recall, f_score)
    """
    if not relevant and not retrieved:
        return 1.0, 1.0, 1.0
    if not relevant or not retrieved:
        return 0.0, 0.0, 0.0

    intersection = retrieved.intersection(relevant)
    precision = len(intersection) / len(retrieved)
    recall = len(intersection) / len(relevant)

    beta_sq = beta**2
    denominator = (beta_sq * precision) + recall
    if denominator == 0:
        f_score = 0.0
    else:
        f_score = (1 + beta_sq) * precision * recall / denominator

    return precision, recall, f_score


def compute_chunk_prf(
    pred_chunks: list[tuple[str, str]],
    gt_chunks: list[tuple[str, str]],
    match_mode: str = "overlap",
    min_iou: float = 0.4,
    beta: float = 2.0,
) -> tuple[float, float, float]:
    """Computes Precision, Recall, and F_beta for chunks supporting exact, containment, or token IoU overlap."""
    if not gt_chunks or not pred_chunks:
        return 0.0, 0.0, 0.0

    if match_mode == "exact":
        norm_pred = {(d, t.strip()) for d, t in pred_chunks}
        norm_gt = {(d, t.strip()) for d, t in gt_chunks}
        return compute_prf(norm_pred, norm_gt, beta=beta)

    pred_matched = set()
    gt_matched = set()

    for p_idx, (p_did, p_text) in enumerate(pred_chunks):
        p_text_clean = p_text.strip()
        p_tokens = set(p_text_clean.split())
        for g_idx, (g_did, g_text) in enumerate(gt_chunks):
            if str(p_did) != str(g_did):
                continue
            g_text_clean = g_text.strip()
            # Exact match or containment (gold in pred or pred in gold)
            if (
                p_text_clean == g_text_clean
                or g_text_clean in p_text_clean
                or p_text_clean in g_text_clean
            ):
                pred_matched.add(p_idx)
                gt_matched.add(g_idx)
                continue

            if match_mode == "overlap":
                g_tokens = set(g_text_clean.split())
                iou = len(p_tokens & g_tokens) / max(len(p_tokens | g_tokens), 1)
                if iou >= min_iou:
                    pred_matched.add(p_idx)
                    gt_matched.add(g_idx)

    precision = len(pred_matched) / len(pred_chunks) if pred_chunks else 0.0
    recall = len(gt_matched) / len(gt_chunks) if gt_chunks else 0.0

    beta_sq = beta**2
    denominator = (beta_sq * precision) + recall
    f_score = (1 + beta_sq) * precision * recall / denominator if denominator > 0 else 0.0
    return precision, recall, f_score


def evaluate_query(
    pred_docs: list[str],
    gt_docs: list[str],
    pred_chunks: list[tuple[str, str]],  # (doc_id, chunk_text)
    gt_chunks: list[tuple[str, str]],  # (doc_id, chunk_text)
    chunk_match_mode: str = "overlap",
) -> dict[str, float]:
    """Evaluates a single query at document and chunk levels."""
    # Document level
    doc_p, doc_r, doc_f2 = compute_prf(set(pred_docs), set(gt_docs), beta=2.0)

    # Chunk level
    chunk_p, chunk_r, chunk_f2 = compute_chunk_prf(
        pred_chunks, gt_chunks, match_mode=chunk_match_mode, beta=2.0
    )

    return {
        "doc_precision": doc_p,
        "doc_recall": doc_r,
        "doc_f2": doc_f2,
        "chunk_precision": chunk_p,
        "chunk_recall": chunk_r,
        "chunk_f2": chunk_f2,
    }


def evaluate_predictions(
    predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    chunk_match_mode: str = "overlap",
) -> dict[str, float]:
    """Computes Macro-Averaged Precision, Recall, and F2 scores across all queries.

    Args:
        predictions: List of dicts matching competition submission format:
            [{ "id": int, "relevant_docs": [...], "relevant_chunks": [{"doc_id": ..., "chunk_text": ...}] }]
        ground_truth: List of dicts in the same schema with gold labels.
        chunk_match_mode: Match criteria for chunks ('overlap', 'containment', or 'exact').

    Returns:
        Dict containing macro averages and overall combined score.
    """
    gt_map = {item["id"]: item for item in ground_truth}

    doc_f2s = []
    chunk_f2s = []
    doc_precisions = []
    doc_recalls = []
    chunk_precisions = []
    chunk_recalls = []

    for pred in predictions:
        qid = pred["id"]
        if qid not in gt_map:
            continue
        gt = gt_map[qid]

        pred_docs = [str(d) for d in pred.get("relevant_docs", [])]
        gt_docs = [str(d) for d in gt.get("relevant_docs", [])]

        pred_chunks = [
            (str(c.get("doc_id", "")), str(c.get("chunk_text", "")))
            for c in pred.get("relevant_chunks", [])
        ]
        gt_chunks = [
            (str(c.get("doc_id", "")), str(c.get("chunk_text", "")))
            for c in gt.get("relevant_chunks", [])
        ]

        scores = evaluate_query(
            pred_docs, gt_docs, pred_chunks, gt_chunks, chunk_match_mode=chunk_match_mode
        )
        doc_f2s.append(scores["doc_f2"])
        chunk_f2s.append(scores["chunk_f2"])
        doc_precisions.append(scores["doc_precision"])
        doc_recalls.append(scores["doc_recall"])
        chunk_precisions.append(scores["chunk_precision"])
        chunk_recalls.append(scores["chunk_recall"])

    macro_doc_f2 = float(np.mean(doc_f2s)) if doc_f2s else 0.0
    macro_chunk_f2 = float(np.mean(chunk_f2s)) if chunk_f2s else 0.0

    return {
        "macro_doc_f2": macro_doc_f2,
        "macro_chunk_f2": macro_chunk_f2,
        "combined_f2": (macro_doc_f2 + macro_chunk_f2) / 2.0,
        "macro_doc_precision": float(np.mean(doc_precisions)) if doc_precisions else 0.0,
        "macro_doc_recall": float(np.mean(doc_recalls)) if doc_recalls else 0.0,
        "macro_chunk_precision": float(np.mean(chunk_precisions)) if chunk_precisions else 0.0,
        "macro_chunk_recall": float(np.mean(chunk_recalls)) if chunk_recalls else 0.0,
        "num_queries_evaluated": len(doc_f2s),
    }
