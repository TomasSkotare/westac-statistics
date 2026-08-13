"""Legacy, deprecated modules.

These modules predate the pyriksdagen-based pipeline introduced in v2.0 and are
superseded by ``corpus_loader`` and ``metadata_loader``:

- ``corpus_parser``      -> replaced by :class:`westac_statistics.CorpusLoader`
- ``metadata_parser``    -> replaced by :class:`westac_statistics.MetadataLoader`
- ``git_repository``     -> not used by the active pipeline
- ``dataframe_optimizer``-> not used by the active pipeline

They are kept only so that archived notebooks under ``docker/py_notebooks`` keep
importing them from their original paths. They are not used by the active
pipeline and should not be extended. The top-level ``westac_statistics`` package
re-exports them so ``from westac_statistics import corpus_parser`` (and the
equivalent ``westac_statistics.<module>`` paths) continue to work.
"""

from . import corpus_parser
from . import dataframe_optimizer
from . import git_repository
from . import metadata_parser
from .corpus_parser import CorpusParser
from .dataframe_optimizer import DataFrameOptimizer
from .git_repository import GitRepo
from .metadata_parser import MetadataParser

__all__ = [
    "corpus_parser",
    "dataframe_optimizer",
    "git_repository",
    "metadata_parser",
    "CorpusParser",
    "DataFrameOptimizer",
    "GitRepo",
    "MetadataParser",
]
