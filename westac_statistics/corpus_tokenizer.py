import glob
import os
import pickle
from collections import defaultdict
from multiprocessing import Pool

import nltk
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import warnings
warnings.warn("The CorpusTokenizer class is deprecated and will be removed in future versions.", DeprecationWarning)

class CorpusTokenizer:
    """The main intent of this class is to load a SPEECH_INDEX dataframe, 
    and convert the found text to tokens represented by integers.
    
    The reason for this is to improve calculation speed and reduce memory footprint.
    
    Each word will be represented by a single integer. 
    
    The SPEECH_INDEX dataframe must only contain two relevant columns:
        * u_id: The unique identifier for the speech
        * text_merged: The text, as a single string per entry in the dataframe

    """

    chunk_directory: str
    SPEECH_INDEX: pd.DataFrame
    chunk_pickle_locations: list
    converted_dataframe: pd.DataFrame = None

    def __init__(self, SPEECH_INDEX: pd.DataFrame, temporary_chunk_directory: str):
        if not os.path.isdir(temporary_chunk_directory):
            os.makedirs(temporary_chunk_directory)
        self.chunk_directory = temporary_chunk_directory
        self.SPEECH_INDEX = SPEECH_INDEX

    @staticmethod
    def increment():
        i = 0
        while True:
            yield i
            i += 1

    def initialize(self, force_reload=False, verbose=True):
        if force_reload or not self.attempt_load_chunks():
            if verbose:
                print("Force reload, or no saved results found.")

            self.start_tokenization()
        else:
            print('Loaded cached results')
        if verbose:
            print("Merging results")
        self.merge_pickled_chunks()
        if verbose:
            print("Done")

    @staticmethod
    def _tokenize_chunk(inp):
        """Tokenizes a chunk (part) of a larger dataframe.
        
        This is used in multiprocessing, and is not intended to be used separately.
        
        Results are saved as pickled files instead of returning due to memory concerns.

        Args:
            inp (_type_): The input, as a dict. It contains two variables:
                - chunk: A part of a larger dataframe
                - pickle_location: The location of where the pickled files should be saved
        """
        chunk, pickle_location = inp
        word_indexes = defaultdict(CorpusTokenizer.increment().__next__)
        print("Starting chunk, length: ", len(chunk), " ", chunk.columns)

        # Here we tokenize the strings into words. If we want to use something else 
        # than nltk.word_tokenize, this is where to change it.
        chunk["text_tokens"] = chunk.text_merged.apply(
            lambda text: np.fromiter(
                (word_indexes[x] for x in nltk.regexp_tokenize(text, pattern=R"(?u)\b\w\w+\b")), dtype=np.uint32
            )
        )
        chunk.drop(columns="text_merged", inplace=True)
        print(f"Finished calculations, dumping results in {pickle_location}")
        with open(pickle_location, "wb") as f:
            pickle.dump((dict(word_indexes), chunk), f)

        print("Completed chunk, returning...")
        # return dict(word_indexes), int_representations

    @staticmethod
    def remove_files_with_extension(directory, extension):
        # use glob to find all files with the specified extension in the directory
        files = glob.glob(os.path.join(directory, f"*.{extension}"))

        # loop over the files and remove them
        for file in files:
            os.remove(file)

    def start_tokenization(self, threads=26):
        # First we remove all old files
        self.__class__.remove_files_with_extension(self.chunk_directory, "pickle")

        speech_chunks = np.array_split(
            self.SPEECH_INDEX[["u_id", "text_merged"]], threads
        )
        self.chunk_pickle_locations = [
            f"{self.chunk_directory}/chunk_{i:02d}.pickle" for i in range(threads)
        ]

        with Pool(threads) as pool:
            print("Starting tokenization.\nSpeech chunks: ", len(speech_chunks))
            pool.map(
                self.__class__._tokenize_chunk,
                zip(speech_chunks, self.chunk_pickle_locations),
            )

            print("Finished!")

    def attempt_load_chunks(self) -> bool:
        files = glob.glob(os.path.join(self.chunk_directory, f"*.pickle"))
        if files:
            self.chunk_pickle_locations = files
            return True
        return False

    def merge_pickled_chunks(self):
        pickle_files = self.chunk_pickle_locations
        merged_dict = defaultdict(self.__class__.increment().__next__)
        loaded_chunks = []
        for file in tqdm(pickle_files):
            with open(file, "rb") as f:
                word_index, chunk = pickle.load(f)
                loaded_chunks.append(
                    (
                        {idx: merged_dict[word] for word, idx in word_index.items()},
                        word_index,
                        chunk,
                    )
                )

        self.word_to_integer = merged_dict
        self.integer_to_word = {v: k for k, v in self.word_to_integer.items()}

        converted_chunks = []
        while loaded_chunks:
            convert_dict, word_index, chunk = loaded_chunks.pop()
            convert_array = np.array(list(convert_dict.values()))

            chunk.text_tokens = chunk.text_tokens.apply(
                lambda word_sequence: convert_array[word_sequence]
            )
            converted_chunks.append(chunk)

        self.converted_dataframe = pd.concat(converted_chunks)

    def count_tokens(self):
        all_words = np.concatenate(self.converted_dataframe.text_tokens.values)
        unique, counts = np.unique(all_words, return_counts=True)

        # Sort to ensure that the order is valid
        sorted_indices = np.argsort(unique)
        self.unique_tokens = unique[sorted_indices]
        self.counts_tokens = counts[sorted_indices]

    @staticmethod
    def create_ngrams(data, ngram_length):
        # create a sliding window view of the array
        ngrams = np.lib.stride_tricks.sliding_window_view(data, window_shape=ngram_length)
        return ngrams.astype(np.uint32)

    def get_tokens(self, u_ids):
        df = self.converted_dataframe
        group = df[df.u_id.isin(u_ids)]
        tokens = np.concatenate(group.text_tokens.values).astype(np.uint32)
        return tokens

    @staticmethod
    def delete_rows_with_word(ngrams, ints_to_skip):
        rows_to_delete = np.where(np.sum(np.isin(ngrams, ints_to_skip),axis=1) > 0)[0]
        inverted_selection = np.ones(ngrams.shape[0],bool)
        inverted_selection[rows_to_delete] = 0
        return ngrams[inverted_selection,:]

    def get_ngrams_per_group(self, groupby, words_to_skip, ngram_length):
        group_dict = {}    
        ints_to_skip = [self.word_to_integer[x] for x in words_to_skip]
        
        for name, group in tqdm(self.SPEECH_INDEX.groupby(groupby)):
            ngrams = self.__class__.create_ngrams(self.get_tokens(group.u_id), ngram_length)
            # Remove words to skip
            ngrams = self.__class__.delete_rows_with_word(ngrams, ints_to_skip)
            
            group_dict[name] = ngrams

        return group_dict
    
    def tokens_to_text(self, tokens: list) -> str:
        return ' '.join([self.integer_to_word[x] for x in tokens])    
        
        