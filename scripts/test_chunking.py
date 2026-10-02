"""Script to test and visualize document chunking for arbitrary text or files."""

import argparse
from pathlib import Path
from src.ingestion.chunker import DocumentChunker


def main():
    parser = argparse.ArgumentParser(description="Split text into overlapping chunks using DocumentChunker.")
    parser.add_argument(
        "--text",
        "-t",
        type=str,
        default=None,
        help="Text string to chunk.",
    )
    parser.add_argument(
        "--file",
        "-f",
        type=Path,
        default=None,
        help="Path to a text file to chunk.",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=400,
        help="Maximum chunk size in characters (default: 400).",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=80,
        help="Overlap size in characters between consecutive chunks (default: 80).",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="vi",
        choices=["vi", "zh", "en"],
        help="Language of the text ('vi', 'zh', or 'en').",
    )

    args = parser.parse_args()

    if args.file and args.file.exists():
        raw_text = args.file.read_text(encoding="utf-8")
    elif args.text:
        raw_text = args.text
    else:
        # Default sample medical text
        raw_text = """Viêm loét dạ dày tá tràng là một bệnh lý đường tiêu hóa phổ biến. Bệnh xảy ra khi lớp niêm mạc bảo vệ dạ dày bị tổn thương do axit dịch vị. Các nguyên nhân chính bao gồm nhiễm vi khuẩn Helicobacter pylori (HP) và việc lạm dụng thuốc kháng viêm không steroid (NSAIDs).

Triệu chứng điển hình là đau vùng thượng vị, ợ hơi, ợ chua và cảm giác buồn nôn sau khi ăn. Nếu không được điều trị kịp thời, bệnh có thể dẫn đến các biến chứng nguy hiểm như xuất huyết tiêu hóa, thủng dạ dày hoặc hẹp môn vị. Người bệnh cần đi khám chuyên khoa tiêu hóa để được nội soi và điều trị đúng phác đồ."""

    chunker = DocumentChunker(
        max_chunk_size=args.max_size,
        chunk_overlap=args.overlap,
        split_by_sentences=True,
    )

    chunks = chunker.chunk_document(doc_id="sample_doc", text=raw_text, lang=args.lang)

    print("=" * 60)
    print(f"Original Text Length: {len(raw_text):,} characters")
    print(f"Total Chunks Created: {len(chunks)}")
    print(f"Settings: max_chunk_size={args.max_size}, chunk_overlap={args.overlap}")
    print("=" * 60)

    for i, chunk in enumerate(chunks):
        print(f"\n[Chunk {i+1}/{len(chunks)}] ID: {chunk.chunk_id} | Span: [{chunk.char_start}:{chunk.char_end}] | Length: {len(chunk.chunk_text)} chars")
        print("-" * 50)
        print(chunk.chunk_text)


if __name__ == "__main__":
    main()
