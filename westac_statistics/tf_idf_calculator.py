import inspect
from multiprocessing import Pool
import sys

import pandas as pd

import numpy as np
from scipy.sparse import spmatrix

from .corpus_fast_tokenizer import FastCorpusTokenizer

from tqdm.auto import tqdm
from numba import jit

class TF_IDF_Calculator:
    TOKENIZER: FastCorpusTokenizer
    SPARSE_COUNT_TRANSPOSED: spmatrix
    SPEECH_INDEX: pd.DataFrame

    def __init__(self, tokenizer: FastCorpusTokenizer) -> None:
        self.TOKENIZER = tokenizer
        self.EXISTS_IN_GROUP = np.asarray(
            np.sum(self.TOKENIZER.SPARSE_COUNT > 0, axis=0)
        ).flatten()
        self.SPARSE_COUNT_TRANSPOSED = self.TOKENIZER.SPARSE_COUNT.T
        self.SPEECH_INDEX = self.TOKENIZER.SPEECH_INDEX

    def get_usage_for_ngram(self, ngram_idx):
        sparse_t = self.SPARSE_COUNT_TRANSPOSED
        indexes = sparse_t.rows[ngram_idx]
        counts = sparse_t.data[ngram_idx]
        # print(' '.join([my_cft.INDEX_TO_WORD[x] for x in my_cft.ALLOWED_NGRAMS[ngram_idx,:]]))
        temp_df = self.SPEECH_INDEX.iloc[indexes].copy()
        temp_df["count"] = counts
        return temp_df

    def calculate_ngram_groups(self, groupby_key, ngram_allowed_indexes=None):
        group_count = np.sum(
            [1 for x in self.SPEECH_INDEX.groupby(groupby_key)]
        )  # How many groups there are
        # Calculate modified IDF per ngram
        IDF_per_ngram = np.log((group_count + 1) / (self.EXISTS_IN_GROUP + 1))
        if ngram_allowed_indexes is not None:
            start, end = ngram_allowed_indexes
            mask = np.ones(IDF_per_ngram.shape, dtype=bool)
            mask[start:end] = False
            IDF_per_ngram[mask] = 0

        group_data = {}
        for name, group in tqdm(
            self.SPEECH_INDEX.groupby(groupby_key), total=group_count
        ):
            df = self.TOKENIZER.VECTORIZED_TEX_DF.loc[group.index]
            group_total_ngrams = np.floor(
                df.vectorized_text.apply(len).sum() / self.TOKENIZER.NGRAM_LENGTH
            ).astype(np.uint32)
            group_ngram_count = np.asarray(
                np.sum(self.TOKENIZER.SPARSE_COUNT[df.index.values, :], axis=0)
            ).flatten()
            group_tf = group_ngram_count / group_total_ngrams
            group_tfidf = group_tf * IDF_per_ngram
            group_data[name] = {
                "ngram_count": group_ngram_count,
                "total_ngrams": group_total_ngrams,
                "uids": df.u_id.values,
                "tf": group_tf,
                "tfidf": group_tfidf,
            }
            # break ## TODO: REMOVE THIS BREAK
        return group_data

    def group_data_to_dataframe(self, group_data, no_to_return: int = 1000, threads=24):
        """Converts the output of calculate_ngram_groups to a dataframe"""
        df_dict = {}

        name_keys = list(group_data.keys())

        global worker_fun

        def worker_fun(chunk):
            worker_keys = [name_keys[x] for x in chunk]
            worker_results = []
            for name in worker_keys:
                group = group_data[name]
                group_results = []
                tfidf = group["tfidf"]
                ngram_count = group["ngram_count"]

                sorted_order = np.argsort(tfidf)[::-1][:no_to_return]  # Top n ngrams

                for i in sorted_order:
                    ngram = self.TOKENIZER.ALLOWED_NGRAMS[i, :]
                    ngram_str = " ".join(
                        [self.TOKENIZER.INDEX_TO_WORD[x] for x in ngram]
                    )
                    usages = self.get_usage_for_ngram(i)
                    comb_str = ""
                    for row, val in (
                        usages.sort_values(by="year")[
                            ["party_abbrev", "year", "decade"]
                        ]
                        .groupby("decade")
                        .party_abbrev.agg(lambda x: np.unique(x))
                        .items()
                    ):
                        comb_str = comb_str + f'{row}: {", ".join(val)}' + " "
                    group_results.append(
                        {
                            "n_gram": ngram_str,
                            "TF-IDF": tfidf[i],
                            "used_by": comb_str,
                            "Instances in group": ngram_count[i],
                            "Instances in any document: ": np.sum(
                                self.SPARSE_COUNT_TRANSPOSED.data[i]
                            ),
                            "ngram_index": i,
                        }
                    )
                    # What to do with the usages?

                    df_dict[name] = group_results
            return df_dict

        chunks = np.array_split(range(len(name_keys)), threads)
        with Pool(threads) as pool:
            results = pool.map(worker_fun, chunks)
        del worker_fun
        merged_results = {}
        for d in results:
            merged_results = merged_results | d

        return merged_results

    def save_ngrams_to_excel(self, df_dict: dict, output_path: str):
        """Saves a df_dict (generated from group_data_to_dataframe) to a set of excel files in output_path"""
        output_directory = f"{output_path}/ngram_length_{self.TOKENIZER.NGRAM_LENGTH}/"
        # Ensure that output_directory exists:
        import os

        os.makedirs(output_directory, exist_ok=True)
        for name, group_data in df_dict.items():
            party, year = name
            df = pd.DataFrame(group_data)

            df.to_excel(f"{output_directory}{party}_{year}.xlsx")




if __name__ == "__main__":
    TF_IDF_Calculator(tokenizer=None)
