#!/usr/bin/env python3
"""
Integration test for the pyriksdagen-based corpus loader pipeline.

Validates that:
1. CorpusLoader extracts speeches with correct schema
2. MetadataLoader enriches with names, parties, and dates
3. Output is compatible with FastCorpusTokenizer.prepare_dataframe()

Usage:
    source .venv/bin/activate && python test_pyriksdagen_pipeline.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from westac_statistics.config import CorpusConfig
from westac_statistics.corpus_fast_tokenizer import FastCorpusTokenizer
from westac_statistics.corpus_loader import CorpusLoader, _extract_speeches_from_file
from westac_statistics.metadata_loader import MetadataLoader


EXPECTED_PARSER_COLUMNS = [
    "who",
    "who_intro_id",
    "who_intro",
    "date",
    "protocol",
    "n_tokens",
    "u_id",
    "u_ids_json",
    "text_json",
    "file_name",
]

ENRICHED_EXTRA_COLUMNS = [
    "party_affiliation",
    "party_abbrev",
    "party_affiliation_uncertain",
    "party_affiliation_message",
    "name",
    "alternative_names",
    "year",
    "decade",
    "text_merged",
]

TOKENIZER_REQUIRED = ["text_merged", "year", "decade", "u_id"]


def test_single_file_extraction():
    xml_file = "riksdagen-corpus/records/data/1975/prot-1975--001.xml"
    df, errors = _extract_speeches_from_file(xml_file)

    assert not df.empty, "Expected non-empty DataFrame from XML file"
    assert list(df.columns) == EXPECTED_PARSER_COLUMNS, (
        f"Column mismatch: {list(df.columns)}"
    )

    assert df["who"].iloc[0].startswith("i-"), (
        f"Expected speaker ID, got {df['who'].iloc[0]}"
    )
    assert df["date"].iloc[0] == "1975-01-10", (
        f"Expected date string, got {df['date'].iloc[0]}"
    )
    assert isinstance(json.loads(df["text_json"].iloc[0]), list), (
        "text_json should deserialize to a list"
    )
    assert isinstance(json.loads(df["u_ids_json"].iloc[0]), list), (
        "u_ids_json should deserialize to a list"
    )
    print(f"  PASSED: Single file extraction ({len(df)} speeches)")


def test_corpus_loader_caching():
    config = CorpusConfig()
    loader = CorpusLoader(config=config)
    loader.xml_files = loader.xml_files[:5]

    loader.initialize(threads=2, force_update=True)
    assert not loader.speech_dataframe.empty, "Expected non-empty DataFrame"
    assert loader.speech_dataframe["date"].dtype.name.startswith("datetime"), (
        f"Expected datetime column, got {loader.speech_dataframe['date'].dtype}"
    )

    loader2 = CorpusLoader(config=config)
    loader2.xml_files = loader2.xml_files[:5]
    loader2.initialize(force_update=False)

    assert len(loader.speech_dataframe) == len(loader2.speech_dataframe), (
        "Cache read should match fresh parse"
    )
    print(f"  PASSED: CorpusLoader caching ({len(loader.speech_dataframe)} speeches)")


def test_metadata_enrichment():
    config = CorpusConfig()
    ml = MetadataLoader(config)
    ml.initialize()

    assert len(ml.name_lookup) > 0, "Name lookup should not be empty"
    assert len(ml.aff_by_person) > 0, "Affiliation lookup should not be empty"

    name = ml.resolve_speaker_name("i-BSEtbZG4ePv3rfebXQWRkd")
    assert name == "Torsten Nilsson", f"Expected primary name, got {name}"

    party, abbrev, uncertain, msg = ml.resolve_party_for_speech(
        "i-BSEtbZG4ePv3rfebXQWRkd", pd.Timestamp("1975-01-10")
    )
    assert party == "Socialdemokraterna", f"Expected full party name, got {party}"
    assert abbrev == "S", f"Expected abbreviation, got {abbrev}"
    assert uncertain is False, f"Expected certain, got uncertain={uncertain}"
    print("  PASSED: Metadata enrichment")


def test_full_pipeline():
    config = CorpusConfig()
    loader = CorpusLoader(config=config)
    loader.xml_files = loader.xml_files[:5]
    loader.initialize(threads=2, force_update=True)

    ml = MetadataLoader(config)
    ml.initialize()
    enriched = ml.enrich_speech_dataframe(loader.speech_dataframe)

    for col in ENRICHED_EXTRA_COLUMNS:
        assert col in enriched.columns, f"Missing column: {col}"

    assert enriched["who"].iloc[0] != enriched["name"].iloc[0] or enriched["who"].iloc[
        0
    ] == "unknown", "who should remain speaker ID, not be overwritten with name"

    known_parties = enriched[enriched["party_affiliation"] != "unknown"]
    if len(known_parties) > 0:
        assert len(known_parties["party_affiliation"].iloc[0]) > 2, (
            "party_affiliation should be full name, not abbreviation"
        )

    for col in TOKENIZER_REQUIRED:
        assert col in enriched.columns, f"Missing tokenizer column: {col}"

    prepared = FastCorpusTokenizer.prepare_dataframe(enriched.copy())
    assert len(prepared) == len(enriched), "prepare_dataframe should not drop rows"
    assert "text_merged" in prepared.columns
    assert "year" in prepared.columns
    assert "decade" in prepared.columns
    assert "u_id" in prepared.columns
    print(f"  PASSED: Full pipeline ({len(prepared)} speeches, tokenizer-ready)")


def main():
    print("=== pyriksdagen pipeline integration tests ===\n")
    tests = [
        ("Single file extraction", test_single_file_extraction),
        ("CorpusLoader caching", test_corpus_loader_caching),
        ("Metadata enrichment", test_metadata_enrichment),
        ("Full pipeline", test_full_pipeline),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {name}: {e}")
            failed += 1

    print(f"\n{'=' * 40}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
    print("All tests passed!")


if __name__ == "__main__":
    main()
