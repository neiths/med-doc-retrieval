"""Benchmark and VRAM stress test for BGE-M3 and BGE-Reranker on RTX 3050 (6GB)."""

import time

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.embedding.bge_m3 import BGEM3Embedder
from src.reranker.bge_reranker import BGEReranker

console = Console()


def get_vram_info() -> dict[str, float]:
    """Returns allocated, reserved, and total VRAM in MB."""
    if not torch.cuda.is_available():
        return {"allocated": 0.0, "reserved": 0.0, "total": 0.0, "free": 0.0}

    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
    free = total - reserved
    return {"allocated": allocated, "reserved": reserved, "total": total, "free": free}


def run_benchmark():
    console.print(Panel.fit("[bold cyan]GPU Memory & Inference Benchmark (FP16)[/bold cyan]"))

    if not torch.cuda.is_available():
        console.print("[bold red]CUDA is not available! Running on CPU.[/bold red]")
        return

    device_name = torch.cuda.get_device_name(0)
    initial_vram = get_vram_info()
    console.print(f"Device: [bold green]{device_name}[/bold green]")
    console.print(
        f"Total VRAM: [bold]{initial_vram['total']:.1f} MB[/bold] (~{initial_vram['total']/1024:.2f} GB)"
    )
    console.print(f"Initial Allocated: {initial_vram['allocated']:.1f} MB | Reserved: {initial_vram['reserved']:.1f} MB\n")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # -------------------------------------------------------------
    # 1. Test BGE-M3 Embedder (FP16)
    # -------------------------------------------------------------
    console.print("[bold yellow]===> 1. Testing BGE-M3 Embedder (FP16)...[/bold yellow]")
    t0 = time.perf_counter()
    embedder = BGEM3Embedder(
        model_name="BAAI/bge-m3",
        device="cuda",
        batch_size=16,
        use_fp16=True,
    )
    _ = embedder.model  # Trigger model load
    t_load_emb = time.perf_counter() - t0

    mem_after_emb_load = get_vram_info()
    console.print(f"  • Load time: {t_load_emb:.2f}s")
    console.print(f"  • VRAM after loading BGE-M3: [cyan]{mem_after_emb_load['allocated']:.1f} MB[/cyan] (Reserved: {mem_after_emb_load['reserved']:.1f} MB)")

    # Sample multilingual texts (VI, EN, ZH)
    test_texts = [
        "Cần làm gì đối với tình trạng tắc nghẽn đường tiết niệu do sỏi thận?",
        "Phương pháp phẫu thuật nội soi tán sỏi niệu quản ngược dòng bằng laser.",
        "Acute ureteral colic caused by obstructive urolithiasis requires immediate decompression.",
        "Management of kidney stones and prevention of recurrence using dietary modifications.",
        "输尿管结石引起的急性尿路梗阻治疗原则与临床干预措施。",
        "体外冲击波碎石术（ESWL）在肾结石患者中的有效性与并发症分析。",
    ] * 8  # 48 items to test batching

    t0 = time.perf_counter()
    embeddings = embedder.encode(test_texts, show_progress_bar=False)
    t_infer_emb = time.perf_counter() - t0
    peak_emb_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)

    console.print(f"  • Encoded {len(test_texts)} sentences in {t_infer_emb:.3f}s ({len(test_texts)/t_infer_emb:.1f} sent/s)")
    console.print(f"  • Embedding shape: {embeddings.shape}")
    console.print(f"  • Peak VRAM during BGE-M3 inference: [bold magenta]{peak_emb_vram:.1f} MB[/bold magenta]\n")

    # -------------------------------------------------------------
    # 2. Test BGE-Reranker (FP16) resident concurrently
    # -------------------------------------------------------------
    console.print("[bold yellow]===> 2. Testing BGE-Reranker-Large (FP16) Concurrently...[/bold yellow]")
    t0 = time.perf_counter()
    reranker = BGEReranker(
        model_name="BAAI/bge-reranker-large",
        device="cuda",
        batch_size=16,
        use_fp16=True,
    )
    reranker._load_model()
    t_load_rerank = time.perf_counter() - t0

    mem_after_both = get_vram_info()
    console.print(f"  • Load time: {t_load_rerank:.2f}s")
    console.print(f"  • VRAM with [bold]BOTH models[/bold] loaded: [cyan]{mem_after_both['allocated']:.1f} MB[/cyan] (Reserved: {mem_after_both['reserved']:.1f} MB)")

    # Sample reranking pairs
    sample_pairs = [
        ("Cần làm gì đối với tình trạng tắc nghẽn đường tiết niệu do sỏi thận?", text)
        for text in test_texts[:32]
    ]

    t0 = time.perf_counter()
    scores = reranker.compute_scores(sample_pairs)
    t_infer_rerank = time.perf_counter() - t0
    total_peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)

    console.print(f"  • Reranked {len(sample_pairs)} pairs in {t_infer_rerank:.3f}s ({len(sample_pairs)/t_infer_rerank:.1f} pairs/s)")
    console.print(f"  • Top score: {max(scores):.4f} | Min score: {min(scores):.4f}")
    console.print(f"  • Overall Peak VRAM (Both Models + Inference): [bold magenta]{total_peak_vram:.1f} MB[/bold magenta] (~{total_peak_vram/1024:.2f} GB)\n")

    # -------------------------------------------------------------
    # Summary Table
    # -------------------------------------------------------------
    table = Table(title="VRAM Usage & Benchmark Summary (RTX 3050 6GB)")
    table.add_column("Component / Stage", style="cyan")
    table.add_column("Allocated VRAM", style="magenta")
    table.add_column("Total Reserved", style="yellow")
    table.add_column("Free Headroom", style="green")

    free_remaining = initial_vram["total"] - total_peak_vram
    table.add_row(
        "Initial State",
        f"{initial_vram['allocated']:.1f} MB",
        f"{initial_vram['reserved']:.1f} MB",
        f"{initial_vram['total'] - initial_vram['reserved']:.1f} MB",
    )
    table.add_row(
        "BGE-M3 (FP16)",
        f"{mem_after_emb_load['allocated']:.1f} MB",
        f"{mem_after_emb_load['reserved']:.1f} MB",
        f"{initial_vram['total'] - mem_after_emb_load['reserved']:.1f} MB",
    )
    table.add_row(
        "Both Models (Embed + Rerank)",
        f"{mem_after_both['allocated']:.1f} MB",
        f"{mem_after_both['reserved']:.1f} MB",
        f"{initial_vram['total'] - mem_after_both['reserved']:.1f} MB",
    )
    table.add_row(
        "Peak During Full Inference",
        f"[bold]{total_peak_vram:.1f} MB[/bold]",
        f"{get_vram_info()['reserved']:.1f} MB",
        f"[bold]{free_remaining:.1f} MB (~{free_remaining/1024:.2f} GB)[/bold]",
    )

    console.print(table)

    if free_remaining > 1000:
        console.print(
            f"[bold green]✔ SUCCESS: Both models run comfortably in FP16 with {free_remaining/1024:.2f} GB VRAM headroom to spare![/bold green]"
        )
    elif free_remaining > 300:
        console.print(
            f"[bold yellow]⚠ CAUTION: Fits in VRAM ({free_remaining:.1f} MB headroom). Keep batch size <= 16.[/bold yellow]"
        )
    else:
        console.print(
            "[bold red]✖ WARNING: High VRAM usage. Recommend reducing batch size to 8 or unloading embedder before reranking.[/bold red]"
        )


if __name__ == "__main__":
    run_benchmark()
