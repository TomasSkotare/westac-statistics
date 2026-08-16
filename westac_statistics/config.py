from dataclasses import dataclass, field
import os


@dataclass(frozen=True)
class CorpusConfig:
    records_path: str = field(default="riksdagen-corpus/records/data")
    persons_path: str = field(default="riksdagen-corpus/persons/data")
    cache_dir: str = field(default=".cache")
    threads: int = field(default=26)
    corpus_version: str = field(default="v1.6.0")
    persons_version: str = field(default="v1.2.2")
    parser_version: str = field(default="2.1")


def load_config(env_prefix: str = "WESTAC") -> CorpusConfig:
    kwargs = {}
    if v := os.environ.get(f"{env_prefix}_RECORDS_PATH"):
        kwargs["records_path"] = v
    if v := os.environ.get(f"{env_prefix}_PERSONS_PATH"):
        kwargs["persons_path"] = v
    if v := os.environ.get(f"{env_prefix}_CACHE_DIR"):
        kwargs["cache_dir"] = v
    if v := os.environ.get(f"{env_prefix}_THREADS"):
        kwargs["threads"] = int(v)
    return CorpusConfig(**kwargs)
