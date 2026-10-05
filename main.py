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
def download_dataset(
    repo_id: str = typer.Option(
        "AIGuruTinix/ViBioMIR", "--repo-id", "-r", help="Hugging Face dataset repository"
    ),
    output_dir: Path = typer.Option(
        Path("data/raw/vibio_mir"), "--output-dir", "-o", help="Directory to save raw files"
    ),
    queries_jsonl: Path = typer.Option(
        Path("data/raw/queries.jsonl"), "--queries", "-q", help="Export queries JSONL"
    ),
    sample_urls: Path = typer.Option(
        Path("data/raw/sample_urls.jsonl"), "--sample-urls", "-s", help="Export sample URLs JSONL"
    ),
):
    """Downloads ViBioMIR competition dataset from Hugging Face and extracts queries & sample URLs."""
    from scripts.download_vibio_mir import (
        download_dataset as hf_download,
    )
    from scripts.download_vibio_mir import (
        inspect_and_export_corpus,
        inspect_and_export_queries,
    )

    console.print(f"[bold green]Downloading {repo_id} to {output_dir}...[/bold green]")
    d_dir = hf_download(repo_id=repo_id, output_dir=output_dir)
    inspect_and_export_queries(d_dir / "query.parquet", export_jsonl_path=queries_jsonl)
    inspect_and_export_corpus(d_dir / "links_corpus.parquet", export_sample_path=sample_urls)
    console.print(f"[bold green]ViBioMIR dataset ready at {output_dir}![/bold green]")


@app.command()
def crawl_urls(
    input_file: Path = typer.Option(
        Path("data/raw/urls.jsonl"), "--input", "-i", help="JSONL or Parquet with {id, url}"
    ),
    output_file: Path = typer.Option(
        Path("data/processed/crawled_articles.jsonl"), "--output", "-o", help="Output JSONL"
    ),
    concurrency: int = typer.Option(10, "--concurrency", "-c", help="Concurrent request limit"),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="Resume from existing output if present"),
):
    """Crawls articles from Vietnamese and Chinese URLs provided by competition organizers."""
    config = load_config()
    scraper = ArticleScraper(
        user_agent=config.crawler.user_agent,
        timeout_seconds=config.crawler.timeout_seconds,
        max_retries=config.crawler.max_retries,
        concurrency=concurrency,
    )
    console.print(f"[bold green]Starting web scraper on {input_file} (resume={resume})...[/bold green]")
    asyncio.run(scraper.scrape_urls_jsonl(input_file=input_file, output_file=output_file, resume=resume))
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
        Path("data/raw/queries.jsonl"), "--queries", "-q", help="Test queries JSONL or Parquet"
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


@app.command()
def clean_parquet(
    input_paths: list[str] = typer.Option(
        ["data/raw"], "--input", "-i", help="JSONL file(s) or directory containing raw crawl data"
    ),
    output_dir: Path = typer.Option(
        Path("data/processed/parquet_corpus"), "--output-dir", "-o", help="Output directory for Parquet shards"
    ),
    chunk_size: int = typer.Option(100_000, "--chunk-size", help="Max rows per Parquet file shard"),
    min_chars: int = typer.Option(50, "--min-chars", help="Minimum article length to keep"),
    sync: bool = typer.Option(False, "--sync", help="Sync to Hugging Face Bucket"),
    bucket: str = typer.Option("hf://buckets/nieths/ViBioMIR/corpus", "--bucket", help="HF Bucket URI"),
):
    """Normalizes text, strips boilerplate regex, deduplicates IDs, and converts JSONL to Parquet."""
    from scripts.convert_jsonl_to_parquet import convert_and_validate, sync_to_hf_bucket
    import glob

    resolved_files: list[Path] = []
    for item in input_paths:
        p = Path(item)
        if p.is_dir():
            resolved_files.extend(sorted(p.glob("*.jsonl")))
        elif "*" in str(item):
            for match in glob.glob(str(item)):
                resolved_files.append(Path(match))
        elif p.exists():
            resolved_files.append(p)
        else:
            console.print(f"[yellow]Input path not found: {item}[/yellow]")

    if not resolved_files:
        console.print("[bold red]No valid JSONL files found. Exiting.[/bold red]")
        raise typer.Exit(1)

    console.print(f"[bold green]Starting conversion of {len(resolved_files)} JSONL files to Parquet...[/bold green]")
    created = convert_and_validate(
        input_paths=resolved_files,
        output_dir=output_dir,
        chunk_size=chunk_size,
        min_chars=min_chars,
    )

    if sync and created:
        sync_to_hf_bucket(output_dir, bucket)


@app.command()
def ui():
    """Launches the Streamlit Pattern Cleaner & Live Regex Testing UI."""
    import subprocess
    console.print("[bold green]Starting Pattern Cleaner Streamlit UI...[/bold green]")
    subprocess.run(["streamlit", "run", "scripts/pattern_cleaner_ui.py"])


if __name__ == "__main__":
    app()
