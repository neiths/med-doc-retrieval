"""CLI Entrypoint for Medical Document Retrieval (Road to AI 2026)."""

import asyncio
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from src.config import load_config
from src.crawler.pubmed import PubMedClient
from src.crawler.url_scraper import ArticleScraper
from src.evaluation.metrics import evaluate_predictions
from src.pipeline import MedicalRetrievalPipeline

app = typer.Typer(
    name="med-retrieval",
    help="CLI for Road to AI 2026 - Multilingual Medical Document Retrieval System",
    add_completion=False,
)
console = Console()


@app.command()
def crawl_urls(
    input_file: Path = typer.Option(
        Path("data/raw/urls.jsonl"), "--input", "-i", help="JSONL with {id, url}"
    ),
    output_file: Path = typer.Option(
        Path("data/processed/crawled_articles.jsonl"), "--output", "-o", help="Output JSONL"
    ),
    concurrency: int = typer.Option(10, "--concurrency", "-c", help="Concurrent request limit"),
):
    """Crawls articles from Vietnamese and Chinese URLs provided by competition organizers."""
    config = load_config()
    scraper = ArticleScraper(
        user_agent=config.crawler.user_agent,
        timeout_seconds=config.crawler.timeout_seconds,
        max_retries=config.crawler.max_retries,
        concurrency=concurrency,
    )
    console.print(f"[bold green]Starting web scraper on {input_file}...[/bold green]")
    asyncio.run(scraper.scrape_urls_jsonl(input_file=input_file, output_file=output_file))
    console.print(f"[bold green]Crawled articles saved to {output_file}[/bold green]")


@app.command()
def fetch_pubmed(
    query: str = typer.Option(..., "--query", "-q", help="English medical query keyword"),
    max_results: int = typer.Option(50, "--max-results", "-m", help="Number of abstracts to fetch"),
    output_file: Path = typer.Option(
        Path("data/processed/pubmed_articles.jsonl"), "--output", "-o", help="Output JSONL"
    ),
):
    """Searches and fetches English biomedical documents from PubMed using NCBI E-utilities."""
    config = load_config()
    client = PubMedClient(
        email=config.crawler.ncbi_email,
        api_key=config.crawler.ncbi_api_key,
        timeout_seconds=config.crawler.timeout_seconds,
    )

    async def _run():
        console.print(f"[cyan]Searching PubMed for:[/cyan] {query}")
        pmids = await client.search_ncbi_pmids(query, retmax=max_results)
        console.print(f"Found {len(pmids)} PMIDs. Fetching abstracts...")
        articles = await client.fetch_ncbi_abstracts(pmids)

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "a", encoding="utf-8") as f:
            f.writelines(json.dumps(art, ensure_ascii=False) + "\n" for art in articles)
        console.print(f"[bold green]Saved {len(articles)} PubMed articles to {output_file}[/bold green]")

    asyncio.run(_run())


@app.command()
def build_index(
    articles_file: Path = typer.Option(
        Path("data/processed/all_articles.jsonl"), "--input", "-i", help="Merged articles JSONL"
    ),
    indices_dir: Path = typer.Option(
        Path("data/indices"), "--output-dir", "-o", help="Directory to save indices"
    ),
    config_path: Path = typer.Option(Path("configs/config.yaml"), "--config", help="Path to config.yaml"),
):
    """Chunks documents, computes dense BGE-M3 embeddings, and creates FAISS + BM25 indices."""
    config = load_config(config_path)
    pipeline = MedicalRetrievalPipeline(config=config)
    pipeline.build_indices(articles_file=articles_file, output_indices_dir=indices_dir)
    console.print(f"[bold green]All indices successfully built and saved to {indices_dir}[/bold green]")


@app.command()
def search(
    query: str = typer.Argument(..., help="Vietnamese medical query string"),
    config_path: Path = typer.Option(Path("configs/config.yaml"), "--config", help="Path to config.yaml"),
):
    """Tests end-to-end hybrid retrieval & reranker on a single query."""
    config = load_config(config_path)
    pipeline = MedicalRetrievalPipeline(config=config)
    result = pipeline.search_query(query)

    console.print(f"\n[bold yellow]Query:[/bold yellow] {query}")
    console.print(f"\n[bold green]Top Relevant Documents:[/bold green] {result['relevant_docs']}")
    console.print("\n[bold green]Top Relevant Chunks:[/bold green]")
    for i, c in enumerate(result["relevant_chunks"], 1):
        console.print(f" [cyan]#{i} (Doc: {c['doc_id']})[/cyan]: {c['chunk_text'][:200]}...")


@app.command()
def evaluate(
    predictions_file: Path = typer.Option(..., "--predictions", "-p", help="Predicted JSON submission file"),
    ground_truth_file: Path = typer.Option(..., "--ground-truth", "-g", help="Ground truth JSON file"),
):
    """Calculates official competition metrics (Precision, Recall, Macro F2 at Doc & Chunk levels)."""
    with open(predictions_file, encoding="utf-8") as f:
        preds = json.load(f)
    with open(ground_truth_file, encoding="utf-8") as f:
        gt = json.load(f)

    metrics = evaluate_predictions(preds, gt)

    table = Table(title="Competition Evaluation Results (Macro F2)")
    table.add_column("Level", style="cyan", no_wrap=True)
    table.add_column("Precision", style="magenta")
    table.add_column("Recall", style="green")
    table.add_column("F2 Score (β=2)", style="bold yellow")

    table.add_row(
        "Document Level",
        f"{metrics['macro_doc_precision']:.4f}",
        f"{metrics['macro_doc_recall']:.4f}",
        f"{metrics['macro_doc_f2']:.4f}",
    )
    table.add_row(
        "Chunk Level",
        f"{metrics['macro_chunk_precision']:.4f}",
        f"{metrics['macro_chunk_recall']:.4f}",
        f"{metrics['macro_chunk_f2']:.4f}",
    )
    table.add_row(
        "Combined / Overall",
        "-",
        "-",
        f"[bold underline]{metrics['combined_f2']:.4f}[/bold underline]",
    )

    console.print(table)
    console.print(f"Total queries evaluated: [bold]{metrics['num_queries_evaluated']}[/bold]")


@app.command()
def generate_submission(
    queries_file: Path = typer.Option(
        Path("data/raw/queries.jsonl"), "--queries", "-q", help="Test queries JSONL"
    ),
    submission_name: str = typer.Option("submission.json", "--name", help="Name of output JSON file"),
    config_path: Path = typer.Option(Path("configs/config.yaml"), "--config", help="Config file"),
):
    """Runs batch inference on queries and packages official submission ZIP for leaderboard upload."""
    config = load_config(config_path)
    pipeline = MedicalRetrievalPipeline(config=config)
    zip_path = pipeline.generate_submission(queries_file=queries_file, submission_filename=submission_name)
    console.print(
        f"[bold green]Ready to submit![/bold green] Upload [yellow]{zip_path}[/yellow] to leaderboard."
    )


if __name__ == "__main__":
    app()
