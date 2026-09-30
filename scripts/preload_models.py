"""Pre-downloads and warms up Hugging Face model weights for offline inference.

Ensures zero network latency, avoids HF Hub rate limits, and verifies VRAM headroom.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import torch
from loguru import logger
from rich.console import Console
from rich.table import Table
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

console = Console()

MODELS_TO_PRELOAD = [
    {
        "name": "BAAI/bge-m3",
        "type": "Embedding (Dense + Sparse)",
        "loader": "bgem3",
    },
    {
        "name": "BAAI/bge-reranker-large",
        "type": "Cross-Encoder Reranker",
        "loader": "reranker",
    },
    {
        "name": "Helsinki-NLP/opus-mt-vi-en",
        "type": "Query Translator (MarianMT)",
        "loader": "seq2seq",
    },
    {
        "name": "ndhieu1101/medical-bidirectional-machine-translation-checkpoints-511042",
        "type": "Medical Domain MT (T5)",
        "loader": "seq2seq",
    },
]


def print_gpu_info() -> None:
    """Displays CUDA GPU device details and VRAM stats."""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        allocated_vram_gb = torch.cuda.memory_allocated(0) / (1024**3)
        reserved_vram_gb = torch.cuda.memory_reserved(0) / (1024**3)
        console.print(
            f"[bold green]GPU detected:[/bold green] [cyan]{device_name}[/cyan] "
            f"| Total VRAM: [bold]{total_vram_gb:.2f} GB[/bold] "
            f"| Allocated: {allocated_vram_gb:.2f} GB | Reserved: {reserved_vram_gb:.2f} GB"
        )
    else:
        console.print("[bold yellow]No CUDA GPU detected. Running in CPU mode.[/bold yellow]")


def preload_model(model_info: dict[str, str], device: str, warmup: bool = True) -> dict[str, Any]:
    """Downloads tokenizer & weights, then runs a warm-up inference pass."""
    model_name = model_info["name"]
    loader_type = model_info["loader"]
    console.print(f"\n[cyan]Checking model:[/cyan] [bold]{model_name}[/bold] ({model_info['type']})...")

    t0 = time.time()
    warmup_latency_ms = 0.0
    status = "SUCCESS"

    try:
        if loader_type == "bgem3":
            from FlagEmbedding import BGEM3FlagModel

            use_fp16 = device == "cuda"
            model = BGEM3FlagModel(model_name, use_fp16=use_fp16, devices=device)
            load_time = time.time() - t0
            logger.info(f"Loaded {model_name} in {load_time:.2f}s")

            if warmup:
                tw0 = time.time()
                sample_text = ["Bệnh nhân có triệu chứng sỏi thận đau quặn thắt lưng."]
                _ = model.encode(sample_text, batch_size=1, max_length=512)
                warmup_latency_ms = (time.time() - tw0) * 1000

        elif loader_type == "reranker":
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            dtype = torch.float16 if device == "cuda" else torch.float32
            model = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=dtype)
            model.to(device)
            model.eval()
            load_time = time.time() - t0
            logger.info(f"Loaded {model_name} in {load_time:.2f}s")

            if warmup:
                tw0 = time.time()
                inputs = tokenizer(
                    [("Sỏi thận điều trị thế nào?", "Tán sỏi ngoài cơ thể là phương pháp phổ biến.")],
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                ).to(device)
                with torch.no_grad():
                    _ = model(**inputs).logits
                warmup_latency_ms = (time.time() - tw0) * 1000

        elif loader_type == "seq2seq":
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            dtype = torch.float16 if device == "cuda" else torch.float32
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dtype)
            model.to(device)
            model.eval()
            load_time = time.time() - t0
            logger.info(f"Loaded {model_name} in {load_time:.2f}s")

            if warmup:
                tw0 = time.time()
                prefix = "vi: " if "ndhieu" in model_name.lower() else ""
                inputs = tokenizer([f"{prefix}sỏi thận"], return_tensors="pt").to(device)
                with torch.no_grad():
                    _ = model.generate(**inputs, max_length=32)
                warmup_latency_ms = (time.time() - tw0) * 1000

    except Exception as e:
        status = f"FAILED: {e}"
        logger.error(f"Error loading {model_name}: {e}")

    # Check VRAM if CUDA
    current_vram_gb = 0.0
    if torch.cuda.is_available():
        current_vram_gb = torch.cuda.memory_allocated(0) / (1024**3)

    return {
        "name": model_name,
        "type": model_info["type"],
        "status": status,
        "load_time_s": time.time() - t0,
        "warmup_ms": warmup_latency_ms,
        "vram_gb": current_vram_gb,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Pre-download and warm up Hugging Face model weights for offline inference."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to load models on (default: auto)",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip warm-up inference pass after loading",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Specific model names to preload (default: all core models)",
    )
    args = parser.parse_args()

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu"

    console.print("\n[bold magenta]══════════════════════════════════════════════════════[/bold magenta]")
    console.print("[bold magenta]🚀 Road to AI 2026 - Model Pre-cache & Warm-up Utility[/bold magenta]")
    console.print("[bold magenta]══════════════════════════════════════════════════════[/bold magenta]\n")
    print_gpu_info()

    models_to_run = MODELS_TO_PRELOAD
    if args.models:
        models_to_run = [m for m in MODELS_TO_PRELOAD if m["name"] in args.models]

    results = []
    for m in models_to_run:
        res = preload_model(m, device=device, warmup=not args.no_warmup)
        results.append(res)

    table = Table(title="\nModel Preload & Verification Summary")
    table.add_column("Model Name", style="cyan", no_wrap=True)
    table.add_column("Type / Role", style="magenta")
    table.add_column("Status", style="bold")
    table.add_column("Load Time", justify="right")
    table.add_column("Warm-up Latency", justify="right")
    table.add_column("Allocated VRAM", justify="right", style="green")

    all_ok = True
    for r in results:
        status_styled = "[green]READY[/green]" if r["status"] == "SUCCESS" else f"[red]{r['status']}[/red]"
        if r["status"] != "SUCCESS":
            all_ok = False
        table.add_row(
            r["name"],
            r["type"],
            status_styled,
            f"{r['load_time_s']:.2f}s",
            f"{r['warmup_ms']:.1f}ms" if r["warmup_ms"] > 0 else "-",
            f"{r['vram_gb']:.2f} GB" if r["vram_gb"] > 0 else "-",
        )

    console.print(table)
    if all_ok:
        console.print(
            "\n[bold green]✅ All models are successfully cached locally and ready for 100% offline inference![/bold green]\n"
        )
    else:
        console.print(
            "\n[bold red]⚠️ Some models failed to load. Please inspect logs above.[/bold red]\n"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
