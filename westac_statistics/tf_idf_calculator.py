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
        # This calculates if a specific ngram exists in a specific group
        self.SPARSE_COUNT_TRANSPOSED = self.TOKENIZER.NGRAM_SPARSE_COUNT.T
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
        """ Calculate TF-IDF for each ngram in each group.

        Calculations are based on the explanation in the wikipedia article:
        https://en.wikipedia.org/wiki/Tf%E2%80%93idf

        Args:
            groupby_key (string): The pandas groupby parameter for the SPEECH_INDEX
            ngram_allowed_indexes (tuple, optional): A tuple containing a start and
                end index for the ngrams to be included in the calculation. Defaults to None,
                which means all ngrams are included.

        Returns:
            _type_: _description_
        """
        # Total number of documents in the corpus
        # This is called N in the wikipedia article
        total_documents = np.sum([1 for x in self.SPEECH_INDEX.groupby(groupby_key)])  

        group_data = {}
        for name, group in tqdm(self.SPEECH_INDEX.groupby(groupby_key), total=total_documents):
            # Grab the precalculated vectorized text
            df = self.TOKENIZER.VECTORIZED_TEX_DF.loc[group.index]
            # Calculate total possible ngrams per group 
            total_ngrams = np.floor(df.vectorized_text.apply(len).sum() / self.TOKENIZER.NGRAM_LENGTH).astype(np.uint32)
            # Found ngrams per group
            ngram_raw_count = np.asarray(
                np.sum(self.TOKENIZER.NGRAM_SPARSE_COUNT[df.index.values, :], axis=0)
            ).flatten()
            # calculate TF (term frequency) vector for each ngram 
            group_tf = ngram_raw_count / total_ngrams

            # Summarize results for the group
            group_data[name] = {
                "ngram_raw_count": ngram_raw_count,
                "total_ngrams": total_ngrams,
                "uids": df.u_id.values,
                "tf": group_tf,
            }

        _, ngram_used_in_group = self.count_group_usages(group_data)
        ngram_used_in_group_count = np.sum(ngram_used_in_group, axis=0)
        document_count = len(group_data)

        # Now that we have this information, we can calculate IDF for each group
        for name, gdata in group_data.items():
            # IDF is the inverse document frequency
            # This is the log of the number of documents divided by the number of documents containing the ngram
            # This is called n in the wikipedia article
            # We add the extra 1 to avoid division by zero
            IDF = np.log((document_count +1) / (ngram_used_in_group_count + 1))
            
            # Set the IDF to zero for ngrams that are not allowed
            if ngram_allowed_indexes is not None:
                start, end = ngram_allowed_indexes
                mask = np.ones(IDF.shape, dtype=bool)
                mask[start:end] = False
                IDF[mask] = 0
            
            group_data[name]["idf"] = IDF
            # Calculate TF-IDF now that we have both TF and IDF
            group_data[name]["tfidf"] = gdata["tf"] * IDF
            
            # Add 'count-idf' as a separate column
            group_data[name]["count-idf"] = gdata["ngram_raw_count"] * IDF       
            
            
        return group_data
    
    
    def count_group_usages(self, group_data):
        """
        Merge the results from the group to a single matrix containing the total number of ngrams used in each group.
        Also creates a boolean matrix indicating if a specific ngram was used in a specific group.
        
        This can be used when calculating the TF-IDF values, as the "documents" in the TF-IDF calculation
        refers to these values instead of the actual documents in the corpus.

        Args:
            group_data (dict): The by-group ngram data created by calculate_ngram_groups

        Returns:
            tuple: A tuple containing the total number of ngrams used in each group and 
              a boolean matrix indicating if a specific ngram was used in a specific group.
        """
        ngram_total_usage_per_group = np.zeros((len(group_data), len(self.TOKENIZER.ALLOWED_COUNTER)),dtype=np.uint32)
        for idx, (_, gdata) in enumerate(group_data.items()):
            ngram_total_usage_per_group[idx,:] = gdata['ngram_raw_count']
        ngram_used_in_group = ngram_total_usage_per_group > 0
        return ngram_total_usage_per_group, ngram_used_in_group


    def ngram_usages_by_decade(self, i):
        """Creates a string for an ngram containing the parties that used the ngram, sorted by year

        TODO: This only works if the grouping makes sense for this. Can it be generalized?
        
        Args:
            i (uint): The ngram index

        Returns:
            str: The string containing the parties that used the ngram, sorted by year
        """
        usages = self.get_usage_for_ngram(i)
                # Create a string containing the parties that used the ngram, sorted by year
                # Consider breaking this out into a separate function
        comb_str = ""
        for row, val in (usages.sort_values(by="year")[["party_abbrev", "year", "decade"]]
                    .groupby("decade")
                    .party_abbrev.agg(lambda x: np.unique(x))
                    .items()
                ):
            comb_str = comb_str + f'{row}: {", ".join(val)}' + " "
        return comb_str

    def group_data_to_dataframe(self, group_data, no_to_return: int = 1000, threads=24):
        """Converts the output of calculate_ngram_groups to a dataframe"""
        df_dict = {}

        name_keys = list(group_data.keys())
        
        # TODO: Check if we can do this only once (it's done twice currently)
        _, ngram_used_in_group = self.count_group_usages(group_data)


        global worker_fun

        def worker_fun(chunk):
            worker_keys = [name_keys[x] for x in chunk]
            for name in worker_keys:
                group = group_data[name]
                group_results = []
                ngram_raw_count = group["ngram_raw_count"]
                group_total_ngrams = group['total_ngrams']
                tfidf = group["tfidf"]
                count_idf = group["count-idf"]

                # Determine which ngrams to include in the document...
                sorted_order = np.argsort(tfidf)[::-1][:no_to_return]  # Top n ngrams

                # Start with the first ngram and go to the end
                for i in sorted_order:
                    ngram = self.TOKENIZER.ALLOWED_NGRAMS[i, :]
                    ngram_str = " ".join(
                        [self.TOKENIZER.INDEX_TO_WORD[x] for x in ngram]
                    )
                    # Dataframe containing the documents used by the ngram
                    comb_str = self.ngram_usages_by_decade(i)
                        
                    # Combine the results into a dictionary    
                    group_results.append(
                        {
                            "n_gram": ngram_str,
                            "TF-IDF": tfidf[i],
                            'count-idf': count_idf[i],
                            "used_by": comb_str,
                            "Instance in document": ngram_raw_count[i],
                            "Instances in any document: ": np.sum(
                                self.SPARSE_COUNT_TRANSPOSED.data[i]
                            ),
                            "Number of documents containing ngram": np.sum(ngram_used_in_group[:,i] > 0),
                            "Total ngrams in document": group_total_ngrams,                            
                            "ngram_index": i,
                            
                        }
                    )

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
