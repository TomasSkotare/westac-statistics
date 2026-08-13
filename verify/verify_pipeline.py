#!/usr/bin/env python3
"""Verification harness for the pyriksdagen pipeline.

Uses real corpus files (from riksdagen-corpus) to confirm that refactoring
has NOT changed pipeline output, and to catch performance regressions.

Modes (mutually exclusive):
  --golden-write    Compute the pipeline output fingerprint and write verify/golden.json
  --golden-check    Recompute the fingerprint and compare to verify/golden.json
                    (exits non-zero on any difference)
  --bench           Time the hot paths and print results
  --bench-write     Time the hot paths and write verify/benchmark.json
  --bench-check     Time the hot paths and compare against verify/benchmark.json
                    (exits non-zero if any step regresses beyond the tolerance)

Examples:
    python verify/verify_pipeline.py --golden-write   # establish baseline (current code)
    python verify/verify_pipeline.py --golden-check   # verify nothing changed
    python verify/verify_pipeline.py --bench-write    # establish perf baseline
    python verify/verify_pipeline.py --bench-check    # check for perf regressions
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from westac_statistics.config import CorpusConfig
from westac_statistics.corpus_loader import CorpusLoader
from westac_statistics.metadata_loader import MetadataLoader

VERIFY_DIR = Path(__file__).resolve().parent
GOLDEN_FILE = VERIFY_DIR / "golden.json"
BENCH_FILE = VERIFY_DIR / "benchmark.json"

# A small, deterministic slice of the real corpus used for all checks.
# Years that exist in the corpus and have enough files to be representative.
SAMPLE_YEARS = ("1867", "1900", "1920", "1975")
FILES_PER_YEAR = 4
BENCH_FILES_PER_YEAR = 12

# Relative regression threshold for the benchmark gate.
REGRESSION_TOLERANCE = 0.20  # allow up to 20% slowdown before failing


def _json_safe(value):
    """Convert a single cell to a JSON-serializable, deterministic value."""
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if np.isnan(value):
            return "NaN"
        if np.isinf(value):
            return "Inf" if value > 0 else "-Inf"
    return value


def canonical_fingerprint(df: pd.DataFrame) -> str:
    """Return a sha256 fingerprint of the dataframe content (order-insensitive)."""
    columns = sorted(str(c) for c in df.columns)
    rows = [_json_safe(r) for r in df[columns].to_dict("records")]
    payload = json.dumps(rows, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _select_files(loader: CorpusLoader, per_year: int) -> None:
    """Restrict loader.xml_files to a deterministic subset spread across years."""
    by_year = {}
    for p in loader.xml_files:
        parts = p.parts
        if len(parts) >= 2:
            by_year.setdefault(parts[-2], []).append(p)
    chosen = []
    for year in SAMPLE_YEARS:
        year_files = sorted(by_year.get(year, []), key=lambda p: p.name)
        chosen.extend(year_files[:per_year])
    loader.xml_files = chosen


def run_pipeline(per_year: int):
    """Run the active pipeline on the sample subset. Returns enriched df and loader."""
    with tempfile.TemporaryDirectory(prefix="westac_verify_") as tmp:
        config = CorpusConfig(cache_dir=tmp)
        loader = CorpusLoader(config=config)
        _select_files(loader, per_year)
        loader.initialize(threads=2, force_update=True)

        ml = MetadataLoader(config)
        ml.initialize()
        enriched = ml.enrich_speech_dataframe(loader.speech_dataframe)
        return enriched, loader, ml


def golden_write() -> dict:
    enriched, loader, _ = run_pipeline(FILES_PER_YEAR)
    fingerprint = canonical_fingerprint(enriched)
    summary = {
        "fingerprint": fingerprint,
        "n_speeches": int(len(enriched)),
        "n_files": int(len(loader.xml_files)),
        "n_columns": int(len(enriched.columns)),
        "columns": sorted(str(c) for c in enriched.columns),
    }
    GOLDEN_FILE.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def golden_check() -> bool:
    current = run_pipeline(FILES_PER_YEAR)[0]
    fingerprint = canonical_fingerprint(current)
    if not GOLDEN_FILE.exists():
        print(f"ERROR: no golden file at {GOLDEN_FILE}. Run --golden-write first.")
        return False
    golden = json.loads(GOLDEN_FILE.read_text())
    ok = fingerprint == golden["fingerprint"]
    print(f"  golden fingerprint: {golden['fingerprint']}")
    print(f"  current fingerprint: {fingerprint}")
    print(f"  n_speeches: golden={golden['n_speeches']} current={len(current)}")
    if ok:
        print("  RESULT: MATCH - pipeline output unchanged.")
    else:
        print("  RESULT: MISMATCH - pipeline output changed!")
    return ok


def _timed(label, fn, repeats=3):
    """Return median wall time in seconds for fn() across repeats."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    times.sort()
    return times[len(times) // 2]


def benchmark() -> dict:
    """Time the hot paths on a larger sample. Returns dict of median seconds."""
    # Parsing: build loader + discover + parse.
    with tempfile.TemporaryDirectory(prefix="westac_verify_") as tmp:
        config = CorpusConfig(cache_dir=tmp)
        loader = CorpusLoader(config=config)
        _select_files(loader, BENCH_FILES_PER_YEAR)
        parse_seconds = _timed("parse", lambda: loader.initialize(threads=2, force_update=True))
        parsed = loader.speech_dataframe

        ml = MetadataLoader(config)
        ml.initialize()
        enrich_seconds = _timed("enrich", lambda: ml.enrich_speech_dataframe(parsed.copy()))

    return {
        "n_files": int(len(loader.xml_files)),
        "n_speeches": int(len(parsed)),
        "parse_seconds": round(parse_seconds, 4),
        "enrich_seconds": round(enrich_seconds, 4),
    }


def bench_write() -> dict:
    result = benchmark()
    BENCH_FILE.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def bench_check() -> bool:
    current = benchmark()
    if not BENCH_FILE.exists():
        print(f"ERROR: no benchmark file at {BENCH_FILE}. Run --bench-write first.")
        return False
    baseline = json.loads(BENCH_FILE.read_text())
    ok = True
    print(f"  {'step':<16} {'baseline':>10} {'current':>10} {'delta%':>8} {'verdict':>8}")
    for key in ("parse_seconds", "enrich_seconds"):
        base = baseline.get(key)
        cur = current.get(key)
        if base is None or cur is None:
            continue
        delta = (cur - base) / base
        verdict = "OK" if delta <= REGRESSION_TOLERANCE else "REGRESSION"
        if verdict != "OK":
            ok = False
        print(
            f"  {key:<16} {base:>10.3f} {cur:>10.3f} {delta * 100:>7.1f}% {verdict:>8}"
        )
    if ok:
        print("  RESULT: no performance regression detected.")
    else:
        print(f"  RESULT: regression(s) detected (tolerance {REGRESSION_TOLERANCE:.0%}).")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--golden-write", action="store_true")
    group.add_argument("--golden-check", action="store_true")
    group.add_argument("--bench", action="store_true")
    group.add_argument("--bench-write", action="store_true")
    group.add_argument("--bench-check", action="store_true")
    args = parser.parse_args()

    if args.golden_write:
        print("=== Writing golden pipeline output baseline ===")
        summary = golden_write()
        print(json.dumps(summary, indent=2))
        return 0
    if args.golden_check:
        print("=== Checking pipeline output against golden baseline ===")
        return 0 if golden_check() else 1
    if args.bench:
        print("=== Benchmarking hot paths ===")
        print(json.dumps(benchmark(), indent=2))
        return 0
    if args.bench_write:
        print("=== Writing benchmark baseline ===")
        result = bench_write()
        print(json.dumps(result, indent=2))
        return 0
    if args.bench_check:
        print("=== Checking for performance regressions ===")
        return 0 if bench_check() else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
