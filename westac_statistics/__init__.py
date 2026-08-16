__version__ = "2.0.0"

from .config import CorpusConfig, load_config
from .corpus_loader import CorpusLoader
from .metadata_loader import MetadataLoader
from .tf_idf_calculator import TF_IDF_Calculator, TfIdfCalculator

# Legacy modules, superseded by the pyriksdagen pipeline. Moved to
# westac_statistics._deprecated but still importable from their original paths
# so archived notebooks (docker/py_notebooks) keep working.
from . import _deprecated
from ._deprecated import (
    CorpusParser,
    DataFrameOptimizer,
    GitRepo,
    MetadataParser,
)

import sys as _sys

# Keep the pre-v2.1 module paths working, e.g.
#   from westac_statistics import corpus_parser
#   from westac_statistics.git_repository import GitRepo
# The implementation now lives in westac_statistics._deprecated.*.
for _legacy in ("corpus_parser", "dataframe_optimizer", "git_repository", "metadata_parser"):
    _module = getattr(_deprecated, _legacy)
    _module.__name__ = f"{__name__}.{_legacy}"
    _sys.modules[f"{__name__}.{_legacy}"] = _module
    setattr(_sys.modules[__name__], _legacy, _module)
del _legacy, _module, _sys

__all__ = [
    "CorpusConfig",
    "CorpusLoader",
    "CorpusParser",
    "DataFrameOptimizer",
    "GitRepo",
    "MetadataLoader",
    "MetadataParser",
    "TF_IDF_Calculator",
    "TfIdfCalculator",
    "corpus_parser",
    "dataframe_optimizer",
    "git_repository",
    "load_config",
    "metadata_parser",
]
