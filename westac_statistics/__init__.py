__version__ = "2.0.0"

from .config import CorpusConfig, load_config
from .corpus_loader import CorpusLoader
from .corpus_parser import CorpusParser
from .dataframe_optimizer import DataFrameOptimizer
from .git_repository import GitRepo
from .metadata_loader import MetadataLoader
from .metadata_parser import MetadataParser
from .tf_idf_calculator import TF_IDF_Calculator

__all__ = [
    "CorpusConfig",
    "CorpusLoader",
    "CorpusParser",
    "DataFrameOptimizer",
    "GitRepo",
    "MetadataLoader",
    "MetadataParser",
    "TF_IDF_Calculator",
    "load_config",
]
