"""Runs end-to-end evaluation benchmark on the mock validation dataset."""

import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src.config import load_config
from src.evaluation.metrics import evaluate_predictions, evaluate_query
from src.pipeline import MedicalRetrievalPipeline

console = Console()


def run_mock_evaluation(
    articles_file: str = "data/mock/articles_all.jsonl",
    queries_file: str = "data/mock/queries_val.jsonl",
    ground_truth_file: str = "data/mock/ground_truth.json",
    collection_name: str = "mock_val_chunks",
    chunk_match_mode: str = "overlap",
    reindex: bool = True,
):
    config = load_config("configs/config.yaml")
    config.retrieval.collection_name = collection_name
    config.paths.indices_dir = "data/mock/indices"
    # Pure local retrieval benchmark on the mock corpus
    config.pubmed.enabled = False

    console.print("[bold cyan]=== Initializing Medical Retrieval Pipeline ===[/bold cyan]")
    pipeline = MedicalRetrievalPipeline(config=config)

    # 1. Indexing
    if config.retrieval.engine == "qdrant":
        if reindex or pipeline.qdrant_index is None or pipeline.qdrant_index.count() == 0:
            console.print(f"[bold yellow]Indexing mock documents from {articles_file}...[/bold yellow]")
            pipeline.build_indices(articles_file=articles_file, output_indices_dir=config.paths.indices_dir)
        console.print(
            f"[green]Total chunks in Qdrant collection '{collection_name}': {pipeline.qdrant_index.count()}[/green]"
        )
    else:
        if reindex or pipeline.dense_index is None:
            console.print(f"[bold yellow]Indexing mock documents from {articles_file}...[/bold yellow]")
            pipeline.build_indices(articles_file=articles_file, output_indices_dir=config.paths.indices_dir)
            pipeline.load_indices(config.paths.indices_dir)
        total_chunks = pipeline.dense_index.index.ntotal if pipeline.dense_index else 0
        console.print(f"[green]Total chunks in FAISS index: {total_chunks}[/green]")

    # 2. Load Queries and Ground Truth
    with open(queries_file, encoding="utf-8") as f:
        queries = [json.loads(line) for line in f if line.strip()]

    with open(ground_truth_file, encoding="utf-8") as f:
        ground_truth = json.load(f)

    gt_map = {item["id"]: item for item in ground_truth}

    # 3. Run Inference on Queries
    console.print(
        f"\n[bold cyan]Running retrieval inference on {len(queries)} mock queries...[/bold cyan]"
    )
    predictions = []
    per_query_results = []

    for q in queries:
        qid = q["id"]
        q_text = q["query"]
        res = pipeline.search_query(q_text)

        pred_entry = {
            "id": qid,
            "relevant_docs": res["relevant_docs"],
            "relevant_chunks": res["relevant_chunks"],
        }
        predictions.append(pred_entry)

        # Evaluate individual query
        if qid in gt_map:
            gt = gt_map[qid]
            pred_docs = [str(d) for d in res["relevant_docs"]]
            gt_docs = [str(d) for d in gt["relevant_docs"]]

            pred_chunks = [(str(c["doc_id"]), str(c["chunk_text"])) for c in res["relevant_chunks"]]
            gt_chunks = [(str(c["doc_id"]), str(c["chunk_text"])) for c in gt["relevant_chunks"]]

            q_metrics = evaluate_query(
                pred_docs, gt_docs, pred_chunks, gt_chunks, chunk_match_mode=chunk_match_mode
            )
            per_query_results.append(
                {
                    "id": qid,
                    "query": q_text,
                    "pred_docs": pred_docs,
                    "gt_docs": gt_docs,
                    "metrics": q_metrics,
                }
            )

    # 4. Overall Macro F2 Evaluation
    overall_metrics = evaluate_predictions(
        predictions, ground_truth, chunk_match_mode=chunk_match_mode
    )

    # 5. Display Per-Query Table
    q_table = Table(title="Per-Query Evaluation Breakdown (Macro F2)")
    q_table.add_column("QID", justify="center", style="cyan")
    q_table.add_column("Query (Excerpt)", style="white")
    q_table.add_column("Retrieved Docs", style="magenta")
    q_table.add_column("GT Docs", style="green")
    q_table.add_column("Doc F2", justify="right", style="yellow")
    q_table.add_column("Chunk F2", justify="right", style="bold yellow")

    for item in per_query_results:
        m = item["metrics"]
        short_q = item["query"][:45] + "..." if len(item["query"]) > 45 else item["query"]
        p_docs_str = ", ".join(item["pred_docs"][:3]) + (
            f" (+{len(item['pred_docs']) - 3})" if len(item["pred_docs"]) > 3 else ""
        )
        gt_docs_str = ", ".join(item["gt_docs"])
        q_table.add_row(
            str(item["id"]),
            short_q,
            p_docs_str,
            gt_docs_str,
            f"{m['doc_f2']:.3f}",
            f"{m['chunk_f2']:.3f}",
        )

    console.print("\n")
    console.print(q_table)

    # 6. Display Summary Table
    summary_table = Table(title="Overall Benchmark Summary (Competition Macro F2)")
    summary_table.add_column("Metric Level", style="cyan", no_wrap=True)
    summary_table.add_column("Precision", justify="right", style="magenta")
    summary_table.add_column("Recall", justify="right", style="green")
    summary_table.add_column("Macro F2 (β=2)", justify="right", style="bold yellow")

    summary_table.add_row(
        "Document Level",
        f"{overall_metrics['macro_doc_precision']:.4f}",
        f"{overall_metrics['macro_doc_recall']:.4f}",
        f"{overall_metrics['macro_doc_f2']:.4f}",
    )
    summary_table.add_row(
        "Chunk Level",
        f"{overall_metrics['macro_chunk_precision']:.4f}",
        f"{overall_metrics['macro_chunk_recall']:.4f}",
        f"{overall_metrics['macro_chunk_f2']:.4f}",
    )
    summary_table.add_row(
        "Combined Score",
        "-",
        "-",
        f"[bold underline green]{overall_metrics['combined_f2']:.4f}[/bold underline green]",
    )

    console.print("\n")
    console.print(summary_table)

    # Save mock prediction output
    output_pred_file = Path("outputs/mock_val_predictions.json")
    output_pred_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_pred_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    console.print(f"\n[dim]Saved mock predictions to {output_pred_file}[/dim]")

    return overall_metrics


if __name__ == "__main__":
    reindex_flag = "--no-reindex" not in sys.argv
    run_mock_evaluation(reindex=reindex_flag)
