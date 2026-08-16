import hashlib
import json
import os
import re
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from lxml import etree
from nltk.tokenize import word_tokenize

from .config import CorpusConfig, load_config


TEI_NS = "{http://www.tei-c.org/ns/1.0}"
XML_NS = "{http://www.w3.org/XML/1998/namespace}"
DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def _extract_date(root):
    doc_date = root.find(f".//{TEI_NS}docDate")
    if doc_date is not None:
        when = doc_date.get("when", "")
        m = DATE_RE.search(when)
        if m:
            return m.group(1)
        text = doc_date.text or ""
        m = DATE_RE.search(text)
        if m:
            return m.group(1)
    return None


def _extract_protocol(root):
    preface = root.find(f".//{TEI_NS}div[@type='preface']")
    if preface is not None:
        head = preface.find(f"{TEI_NS}head")
        if head is not None and head.text:
            return head.text.strip()
    return None


def _extract_speeches_from_file(xml_file):
    xml_file = str(xml_file)
    errors = []
    try:
        parser = etree.XMLParser(remove_blank_text=True, recover=True)
        root = etree.parse(xml_file, parser).getroot()
    except Exception as e:
        return pd.DataFrame(), [{"file": xml_file, "error": str(e)}]

    if root is None:
        return pd.DataFrame(), [
            {"file": xml_file, "error": "unparseable document (no root element)"}
        ]

    if len(parser.error_log) > 0:
        first = parser.error_log[0]
        errors.append(
            {
                "file": xml_file,
                "error": (
                    f"malformed XML recovered ({len(parser.error_log)} parser "
                    f"messages, first: {first.message.strip()})"
                ),
            }
        )

    date_str = _extract_date(root)
    protocol = _extract_protocol(root)

    speaker_notes = root.findall(f".//{TEI_NS}note[@type='speaker']")
    if not speaker_notes:
        errors.append(
            {
                "file": xml_file,
                "protocol": protocol,
                "error": (
                    "no speaker notes in file; all <u> elements merged into "
                    "one speech (attribution unreliable)"
                ),
            }
        )

    speech_blocks = []
    current_block = []
    speaker_note = None

    all_elements = sorted(
        root.findall(f".//{TEI_NS}u") + speaker_notes,
        key=lambda e: e.sourceline,
    )

    for elem in all_elements:
        tag = elem.tag.replace(TEI_NS, "")
        if tag == "note":
            if current_block:
                speech_blocks.append((speaker_note, current_block))
            current_block = []
            speaker_note = elem
        elif tag == "u":
            current_block.append(elem)

    if current_block:
        speech_blocks.append((speaker_note, current_block))

    speeches = []

    for speaker_note_elem, u_elems in speech_blocks:
        who_ids = set()
        for u in u_elems:
            w = u.get("who")
            if w:
                who_ids.add(w)

        who = "unknown_no_xml_id"
        if len(who_ids) > 1:
            errors.append({
                "file": xml_file,
                "protocol": protocol,
                "who_ids": sorted(who_ids),
            })
            known = [w for w in who_ids if not w.startswith("unknown")]
            who = known[0] if known else sorted(who_ids)[0]
        elif len(who_ids) == 1:
            who = list(who_ids)[0]

        who_intro = ""
        who_intro_id = ""
        if speaker_note_elem is not None:
            who_intro_id = speaker_note_elem.get(f"{XML_NS}id", "")
            who_intro = (speaker_note_elem.text or "").strip()

        text_parts = []
        for u in u_elems:
            segs = u.findall(f"{TEI_NS}seg")
            if segs:
                for seg in segs:
                    t = seg.text or ""
                    if t.strip():
                        text_parts.append(" ".join(t.strip().split()))
            else:
                t = u.text or ""
                if t.strip():
                    text_parts.append(" ".join(t.strip().split()))

        n_tokens = int(np.sum([len(word_tokenize(t)) for t in text_parts]))
        u_ids = [u.get(f"{XML_NS}id", "") for u in u_elems]

        speeches.append({
            "who": who,
            "who_intro_id": who_intro_id,
            "who_intro": who_intro,
            "date": date_str,
            "protocol": protocol,
            "n_tokens": n_tokens,
            "u_id": u_ids[0] if u_ids else "",
            "u_ids_json": json.dumps(u_ids, ensure_ascii=False),
            "text_json": json.dumps(text_parts, ensure_ascii=False),
            "file_name": xml_file,
        })

    return pd.DataFrame(speeches), errors


class CorpusLoader:
    speech_dataframe: pd.DataFrame
    empty_speeches: pd.DataFrame

    def __init__(
        self,
        corpus_directory: str | None = None,
        database_file: str | None = None,
        config: CorpusConfig | None = None,
    ):
        self.config = config or load_config()
        self.corpus_directory = corpus_directory or self.config.records_path
        self.database_file = database_file or os.path.join(
            self.config.cache_dir,
            f"corpus_{self.config.corpus_version.replace('.', '_')}.feather",
        )
        self.error_file = self.database_file.replace(".feather", "_errors.json")
        self._discover_files()

    def _discover_files(self):
        try:
            from pyriksdagen.utils import corpus_iterator

            paths = list(
                corpus_iterator(
                    "prot",
                    corpus_root=self.corpus_directory,
                )
            )
            self.xml_files = sorted(
                [Path(p) for p in paths if os.path.exists(p)],
                key=lambda p: p.name,
            )
        except Exception:
            self.xml_files = sorted(
                Path(self.corpus_directory).glob("**/*.xml"),
                key=lambda p: p.name,
            )

    # Number of bytes sampled from the start and end of each file when
    # computing the corpus hash. Catches content changes that preserve
    # name/size/mtime (cp -a, tar, hardlinks) at a fraction of the cost of a
    # full content digest.
    _HASH_SAMPLE_BYTES = 64 * 1024

    def _compute_file_hash(self) -> str:
        h = hashlib.sha256()
        for p in self.xml_files:
            st = p.stat()
            h.update(p.name.encode())
            h.update(str(st.st_mtime).encode())
            h.update(str(st.st_size).encode())
            if st.st_size > 0:
                with open(p, "rb") as f:
                    head = f.read(self._HASH_SAMPLE_BYTES)
                    f.seek(max(0, st.st_size - self._HASH_SAMPLE_BYTES))
                    tail = f.read(self._HASH_SAMPLE_BYTES)
                h.update(head)
                h.update(tail)
        return h.hexdigest()[:16]

    def _cache_valid(self) -> bool:
        if not os.path.exists(self.database_file):
            return False
        meta_file = self.database_file + ".meta"
        if not os.path.exists(meta_file):
            return False
        try:
            with open(meta_file) as f:
                meta = json.load(f)
            return (
                meta.get("file_hash") == self._compute_file_hash()
                and meta.get("parser_version") == self.config.parser_version
            )
        except Exception:
            return False

    def _save_cache_meta(self):
        meta_file = self.database_file + ".meta"
        meta = {
            "file_hash": self._compute_file_hash(),
            "parser_version": self.config.parser_version,
            "corpus_version": self.config.corpus_version,
            "row_count": len(self.speech_dataframe),
            "xml_file_count": len(self.xml_files),
        }
        os.makedirs(os.path.dirname(self.database_file) or ".", exist_ok=True)
        with open(meta_file, "w") as f:
            json.dump(meta, f)

    def perform_threaded_parsing(self, threads: int | None = None):
        n = threads or self.config.threads
        with Pool(n) as pool:
            results = pool.map(_extract_speeches_from_file, self.xml_files)
        dataframes = []
        all_errors = []
        for df, errors in results:
            if not df.empty:
                dataframes.append(df)
            all_errors.extend(errors)
        if all_errors:
            os.makedirs(os.path.dirname(self.error_file) or ".", exist_ok=True)
            with open(self.error_file, "w") as f:
                json.dump(all_errors, f)
        elif os.path.exists(self.error_file):
            os.remove(self.error_file)
        return dataframes

    def read_speech_dataframe_from_disk(self) -> pd.DataFrame:
        df = pd.read_feather(self.database_file)
        for col in [c for c in df.columns if c.endswith("_json")]:
            df[col.removesuffix("_json")] = df[col].apply(json.loads)
            df = df.drop(columns=col)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df

    def initialize(self, threads: int | None = None, force_update: bool = False):
        if not force_update and self._cache_valid():
            self.speech_dataframe = self.read_speech_dataframe_from_disk()
        else:
            per_xml_dataframes = self.perform_threaded_parsing(threads=threads)
            if os.path.exists(self.database_file):
                os.remove(self.database_file)
            if per_xml_dataframes:
                df = pd.concat(per_xml_dataframes, ignore_index=True)
            else:
                df = pd.DataFrame(
                    columns=[
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
                )
            os.makedirs(os.path.dirname(self.database_file) or ".", exist_ok=True)
            df.to_feather(self.database_file, compression="lz4")
            self.speech_dataframe = self.read_speech_dataframe_from_disk()
            self._save_cache_meta()

        self.empty_speeches = self.speech_dataframe[
            self.speech_dataframe.n_tokens == 0
        ]
        self.speech_dataframe = self.speech_dataframe.drop(
            self.empty_speeches.index
        )
