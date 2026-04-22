"""Training-pack preparation for custom wakeword sample batches."""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

from .wakeword_samples import WakewordSampleRecord


@dataclass(frozen=True, slots=True)
class WakewordTrainingPack:
    keyword: str
    slug: str
    sample_count: int
    pack_dir: str
    archive_path: str


def _normalize_keyword(value: str) -> str:
    return value.strip().lower()


def _slugify(value: str) -> str:
    pieces = []
    for char in _normalize_keyword(value):
        pieces.append(char if char.isalnum() else "-")
    slug = "".join(pieces).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "keyword"


def load_sample_manifest(batch_dir: str | Path) -> list[WakewordSampleRecord]:
    manifest_path = Path(batch_dir) / "manifest.jsonl"
    records: list[WakewordSampleRecord] = []
    for raw in manifest_path.read_text().splitlines():
        if not raw.strip():
            continue
        records.append(WakewordSampleRecord(**json.loads(raw)))
    return records


def export_training_packs(
    batch_dir: str | Path,
    output_dir: str | Path,
    *,
    keywords: list[str] | None = None,
) -> list[WakewordTrainingPack]:
    batch_dir = Path(batch_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = None
    if keywords:
        selected = {_normalize_keyword(keyword) for keyword in keywords if keyword.strip()}

    grouped: dict[str, list[WakewordSampleRecord]] = {}
    for record in load_sample_manifest(batch_dir):
        normalized = _normalize_keyword(record.text)
        if selected is not None and normalized not in selected:
            continue
        grouped.setdefault(normalized, []).append(record)

    packs: list[WakewordTrainingPack] = []
    index_rows: list[dict[str, object]] = []
    for keyword in sorted(grouped):
        records = sorted(grouped[keyword], key=lambda row: row.relative_path)
        slug = _slugify(keyword)
        pack_root = output_dir / slug
        if pack_root.exists():
            shutil.rmtree(pack_root)
        samples_dir = pack_root / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)

        copied_names: list[str] = []
        for record in records:
            source = batch_dir / record.relative_path
            if not source.exists():
                raise FileNotFoundError(source)
            destination = samples_dir / Path(record.relative_path).name
            shutil.copy2(source, destination)
            copied_names.append(destination.name)

        archive_path = pack_root / f"{slug}-samples.zip"
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in copied_names:
                archive.write(samples_dir / name, arcname=name)

        manifest_payload = {
            "keyword": keyword,
            "slug": slug,
            "sample_count": len(records),
            "records": [asdict(record) for record in records],
        }
        (pack_root / "manifest.json").write_text(
            json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n"
        )

        pack = WakewordTrainingPack(
            keyword=keyword,
            slug=slug,
            sample_count=len(records),
            pack_dir=str(pack_root),
            archive_path=str(archive_path),
        )
        packs.append(pack)
        index_rows.append(asdict(pack))

    (output_dir / "manifest.json").write_text(
        json.dumps({"packs": index_rows}, indent=2, sort_keys=True) + "\n"
    )
    return packs


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Group a wakeword sample batch into per-keyword training packs."
    )
    parser.add_argument("--batch-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--keyword", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    packs = export_training_packs(
        args.batch_dir,
        args.output_dir,
        keywords=args.keyword or None,
    )
    print(f"Wrote {len(packs)} wakeword training packs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
