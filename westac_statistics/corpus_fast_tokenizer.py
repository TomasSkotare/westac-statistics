from multiprocessing import Pool

import nltk
import numpy as np
import pandas as pd
from numba import jit
from tqdm.auto import tqdm
from scipy.sparse import dok_matrix
from collections import Counter, defaultdict


class FastCorpusTokenizer:
    # The SPEECH_INDEX contains the raw text and metadata for the speeches
    SPEECH_INDEX: pd.DataFrame
    # The ALLOWED_NGRAMS contains the ngrams that are allowed to use.
    # Other ngrams either are too rare or contain stop words.
    ALLOWED_NGRAMS: np.array
    # This contains an index of word-to-integer
    CONVERT_DICT: dict
    # This is the opposite of the CONVERT_DICT
    INDEX_TO_WORD: list
    # The word length for each word in the CONVERT_DICT (in order)
    WORD_LENGTH: np.array
    # The documents with words replaced by integers, in this case np.uint32
    VECTORIZED_DOCUMENTS: list
    # How many times each word was used
    SUMMED_WORD_COUNTS: np.array
    # The dataframe containing uid and vectorized text
    VECTORIZED_TEX_DF: pd.DataFrame
    # An array containing the stop words, i.e. words that are disallowed
    STOP_WORD_ARRAY: np.array
    MERGED_NGRAMS: np.array
    MERGED_COUNTER: np.array
    MINIMUM_NGRAM_COUNT: int
    ALLOWED_NGRAMS: np.array
    ALLOWED_COUNTER: np.array
    ALLOWED_PER_COL: np.array
    WORD_TO_NGRAM_INDEXES: np.array
    NGRAM_SPARSE_COUNT: dok_matrix
    DOCUMENT_FREQUENCY: np.array
    INVERSE_DOCUMENT_FREQUENCY_MODIFIER: np.array
    DOCUMENT_TOTAL_COUNT: np.array

    # The number of threads to use when running in parallel
    THREADS: int

    @property
    def word_count(self):
        return len(self.CONVERT_DICT)

    # Create property that contains number of rows in allowed_ngrams:
    @property
    def num_allowed_ngrams(self):
        return self.ALLOWED_NGRAMS.shape[0]

    @staticmethod
    def prepare_dataframe(SPEECH_INDEX: pd.DataFrame):
        """Creates merged text from a SPEECH_INDEX dataframe.
        The original dataframe is assumed to contain text and year columns.

        Args:
            SPEECH_INDEX (pd.DataFrame): _description_

        Returns:
            _type_: _description_
        """
        SPEECH_INDEX["text_merged"] = SPEECH_INDEX.text.apply(
            lambda x: " ".join(iter(x)).lower()
        )
        SPEECH_INDEX["decade"] = SPEECH_INDEX.year.apply(lambda x: x - (x % 10))
        SPEECH_INDEX.drop(
            ["file_name", "alternative_names", "text", "u_ids"], axis=1, inplace=True
        )
        return SPEECH_INDEX

    def __init__(

        self,
        SPEECH_INDEX: pd.DataFrame,  # A DataFrame containing the speech index
        ngram_length=7,  # The length of the n-grams to be used, default is 7
        threads: int = 24,  # The number of threads to be used for parallel processing, default is 24
        stop_word_file: str = None,  # The file containing stop words, default is None
        minimum_ngram_count: int = 10,  # The minimum count for an n-gram to be considered, default is 10
        regex_pattern=R"(?u)\b\w\w+\b",  # The regex pattern to be used for word matching, default is any word of length 2 or more
    ):
        """
        Initializes the instance with the given parameters and performs several preprocessing steps on the speech index.

        Parameters:
        SPEECH_INDEX (pd.DataFrame): A DataFrame containing the speech index.
        ngram_length (int, optional): The length of the n-grams to be used. Defaults to 7.
        threads (int, optional): The number of threads to be used for parallel processing. Defaults to 24.
        stop_word_file (str, optional): The file containing stop words. Defaults to None.
        minimum_ngram_count (int, optional): The minimum count for an n-gram to be considered. Defaults to 10.
        regex_pattern (str, optional): The regex pattern to be used for word matching. Defaults to any word of length 2 or more.

        The function performs the following steps:
        1. Finds all words in the corpus.
        2. Vectorizes the documents in the corpus.
        3. Loads stop words from a file if provided, otherwise initializes an array of zeros.
        4. Counts the n-grams in the corpus.
        5. Filters out the n-grams that occur less than the minimum count.
        6. Calculates which words are involved in each column of the n-grams.
        7. Gets the indexes of the n-grams for each word.
        8. Counts the valid n-grams for each uid.
        9. Creates a sparse matrix from the n-gram counts.
        10. Calculates the document frequency, modified document frequency, and total count for each n-gram.
        """
        
        # Initialize instance variables
        self.SPEECH_INDEX = SPEECH_INDEX
        self.NGRAM_LENGTH = ngram_length
        self.MINIMUM_NGRAM_COUNT = minimum_ngram_count
        self.REGEX_PATTERN = regex_pattern
        self.THREADS = threads

        print("Finding all unique words in corpus...")
        # Find all words in the corpus using multiple threads
        self.CONVERT_DICT, self.INDEX_TO_WORD = self.threaded_find_words(
            threads=threads
        )
        # Calculate the length of each word and store it as a numpy array
        self.WORD_LENGTH = np.array(
            [len(x) for x in self.INDEX_TO_WORD], dtype=np.uint16
        )
        print("Found ", len(self.CONVERT_DICT), "words in corpus.")

        print("Vectorizing documents...")
        # Record the start time of the vectorization process
        start = pd.Timestamp.now()
        # Vectorize the documents using multiple threads
        self.VECTORIZED_DOCUMENTS, self.SUMMED_WORD_COUNTS = self.vectorize_documents(
            threads=threads, pattern = self.REGEX_PATTERN
        )
        # Create a DataFrame from the vectorized documents
        self.VECTORIZED_TEX_DF = pd.DataFrame(
            [{"u_id": x, "vectorized_text": y} for x, y in self.VECTORIZED_DOCUMENTS]
        )
        # Record the end time of the vectorization process
        end = pd.Timestamp.now()
        print("Vectorizing took ", end - start, " seconds.")

        # If a stop word file is provided, load the stop words from the file
        if stop_word_file:
            print(f"Loading stop words from file: {stop_word_file}")
            self.STOP_WORD_ARRAY = self.load_stop_words(stop_word_file)
        else:
            # If no stop word file is provided, initialize an array of zeros
            self.STOP_WORD_ARRAY = np.zeros(self.word_count, dtype=bool)

        # Count the n-grams in the corpus using multiple threads
        self.MERGED_NGRAMS, self.MERGED_COUNTER = self.threaded_ngram_counting(threads)

        # Filter out the n-grams that occur less than the minimum count
        self.ALLOWED_NGRAMS = self.MERGED_NGRAMS[
            self.MERGED_COUNTER >= minimum_ngram_count, :
        ]
        self.ALLOWED_COUNTER = self.MERGED_COUNTER[
            self.MERGED_COUNTER >= minimum_ngram_count
        ]
        
        # Print the percentage of n-grams that are being kept based on the minimum count criteria
        print(
            f"Keeping {(self.ALLOWED_NGRAMS.shape[0] / self.MERGED_NGRAMS.shape[0]) * 100:.2f}% of ngrams."
        )

        print("Calculating which words are involved in each column...")
        # Initialize a boolean array to keep track of which words are involved in each column of the n-grams
        self.ALLOWED_PER_COL = np.zeros(
            (len(self.CONVERT_DICT), self.NGRAM_LENGTH), dtype=bool
        )
        # For each column, mark the words that are involved in the n-grams
        for col in range(self.ALLOWED_PER_COL.shape[1]):
            self.ALLOWED_PER_COL[self.ALLOWED_NGRAMS[:, col], col] = True

        # Get the indexes of the n-grams for each word
        self.WORD_TO_NGRAM_INDEXES = self.get_word_to_ngram_index()

        print("Calculating valid ngrams per uid...")
        # Record the start time of the n-gram counting process
        start = pd.Timestamp.now()
        # Count the valid n-grams for each uid using multiple threads
        self.NGRAMS_PER_UID = self.threaded_valid_ngram_counting(threads=threads)
        # Record the end time of the n-gram counting process
        end = pd.Timestamp.now()
        print("Calculating valid ngrams took ", end - start, " seconds.")

        print("Creating sparse matrix...")
        # Create a sparse matrix from the n-gram counts
        self.NGRAM_SPARSE_COUNT = self.create_ngram_sparse_matrix()

        print("Calculate document frequency and total count...")
        # Calculate the document frequency, modified document frequency, and total count for each n-gram
        (
            self.DOCUMENT_FREQUENCY,
            self.INVERSE_DOCUMENT_FREQUENCY_MODIFIER,
            self.DOCUMENT_TOTAL_COUNT,
        ) = self.calculate_document_counts()

    def calculate_document_counts(self):
        # Convert the sparse matrix to COOrdinate format for efficient arithmetic and boolean operations
        sparse = self.NGRAM_SPARSE_COUNT.tocoo()

        # Calculate the document frequency (df) for each term (n-gram) in the corpus.
        # This is done by summing up the binary occurrences (0 or 1) of each term across all documents.
        # The result is a 1D numpy array where each element corresponds to the document frequency of a term.
        df = np.asarray(np.sum(sparse > 0, axis=0)).flatten()

        # Calculate the inverse document frequency modifier (df_m) for each term.
        # This is done by taking the reciprocal of the document frequency.
        # The result is a 1D numpy array where each element corresponds to the inverse document frequency of a term.
        # Note: If there were any terms that did not appear in any document, their document frequency would be zero,
        # and this line would cause a ZeroDivisionError. However, in this case, all terms that do not appear in any
        # document have already been filtered out, so every term should have a document frequency greater than zero.
        df_m = 1 / df

        # Calculate the total count (dtc) of each term in the corpus.
        # This is done by summing up the actual counts of each term across all documents.
        # The result is a 1D numpy array where each element corresponds to the total count of a term.
        dtc = np.asarray(np.sum(sparse, axis=0)).flatten()

        # Return the document frequency, inverse document frequency, and total count arrays.
        return df, df_m, dtc

    def create_ngram_sparse_matrix(self):
        """This creates a sparse matrix from the ngram counts.
        
        The size is (document_count, ngram_count).
        
        This can then be used to find out if a specific ngram exists in a specific document.

        Returns: dok_matrix (scipy.sparse.dok_matrix): A sparse matrix containing the ngram counts.
            
        """
        uid_ngram_sparse = dok_matrix(
            (len(self.NGRAMS_PER_UID), len(self.ALLOWED_NGRAMS)), dtype=np.uint16
        ).tolil()
        for idx, uid_ngrams in enumerate(self.NGRAMS_PER_UID):
            ngrams = uid_ngrams["ngrams"]
            if len(ngrams) == 0:
                continue
            d = defaultdict(int)
            for ngram in ngrams:
                d[ngram] += 1
            uid_ngram_sparse.data[idx] = list(d.values())
            uid_ngram_sparse.rows[idx] = list(d.keys())
        return uid_ngram_sparse

    def threaded_valid_ngram_counting(self, threads):
        """Returns the valid ngrams per uid. This is a list of dicts with uid and ngrams."""

        global count_valid_ngrams
        global numba_count_valid_ngrams

        @jit(nopython=True)
        def numba_count_valid_ngrams(
            data2d, allowed_ngram, word_to_ngram_indexes, allowed_per_col
        ):
            """
            Comment:
            The inefficient part is the "np.where" part which the previous version attempted to
            perform more efficiently by stopping at the end, as we expect them to be continuous.

            In general, this is not the step that takes a long time.

            Args:
                data2d (_type_): _description_
                allowed_ngram (_type_): _description_
                word_to_ngram_indexes (_type_): _description_
                allowed_per_col (_type_): _description_

            Returns:
                _type_: _description_
            """

            is_valid_ngram = np.ones(data2d.shape[0], dtype=np.bool_)
            # First we do a quick filter for allowed words
            for col in range(data2d.shape[1]):
                is_valid_ngram = (
                    is_valid_ngram & allowed_per_col[:, col][data2d[:, col]]
                )
            data2d = data2d[is_valid_ngram, :]
            found_ngram = np.ones(data2d.shape[0], np.uint32) * -1

            for idx, row in enumerate(data2d):
                start_index, end_index = word_to_ngram_indexes[row[0]]
                end_index += 1
                if start_index == -1:
                    continue
                # Merged is always sorted. So, limiting range this way should work.

                for col in range(
                    1, allowed_ngram.shape[1]
                ):  # Start at col2, we know col1 matches
                    row_val = row[col]
                    col_vals = allowed_ngram[
                        start_index:end_index, col
                    ]  # get current matches
                    where = np.where(row_val == col_vals)[0]
                    if len(where) == 0:
                        start_index = -1
                        break
                    s = where[0]
                    e = where[-1] + 1
                    end_index = start_index + e
                    start_index = start_index + s
                if start_index != -1:
                    found_ngram[idx] = start_index

                # else:
                # print('not found idx ', idx)
            return found_ngram[found_ngram > -1]

        def count_valid_ngrams(chunk):
            uids = self.VECTORIZED_TEX_DF.u_id.values[chunk]
            chunk_text = self.VECTORIZED_TEX_DF.vectorized_text.values[chunk]
            ngram_list = []
            for uid, data in zip(uids, chunk_text):
                if len(data) < self.NGRAM_LENGTH:
                    # Add dummy ngram
                    ngram_list.append(
                        {"uid": uid, "ngrams": np.array([], dtype=np.uint32)}
                    )
                    continue
                data2d = np.lib.stride_tricks.sliding_window_view(
                    data, window_shape=self.NGRAM_LENGTH
                )
                valid_ngrams = numba_count_valid_ngrams(
                    data2d,
                    self.ALLOWED_NGRAMS,
                    self.WORD_TO_NGRAM_INDEXES,
                    self.ALLOWED_PER_COL,
                )
                ngram_list.append({"uid": uid, "ngrams": valid_ngrams})
            return ngram_list

        chunks = np.array_split(range(len(self.VECTORIZED_TEX_DF)), threads)

        with Pool(threads) as pool:
            ngrams_per_uid = pool.map(count_valid_ngrams, chunks)
        del count_valid_ngrams
        del numba_count_valid_ngrams

        return_list = []
        for ng in ngrams_per_uid:
            return_list.extend(ng)
        return return_list

    def get_word_to_ngram_index(self):
        """The intent of this function is to get a mapping from the first word
        in an ngram and the indexes of all allowed ngrams starting with that word."""

        @jit(nopython=True)
        def ngram_indexes_from_first_word(ngram_list, total_words):
            val_to_index = np.ones((total_words, 2), dtype=np.uint32) * -1
            for i, val in enumerate(ngram_list[:, 0]):
                if val_to_index[val, 0] == -1:
                    val_to_index[val, :] = [i, i]
                else:
                    val_to_index[val, 1] = i
            return val_to_index

        return ngram_indexes_from_first_word(self.ALLOWED_NGRAMS, self.word_count)

    def allowed_ngrams_per_doc(self, data, allowed_per_col, ngram_length):
        if len(data) < ngram_length:
            return []
        data2d = np.lib.stride_tricks.sliding_window_view(
            data, window_shape=ngram_length
        )
        # print(data2d.shape)
        for col in range(ngram_length):
            data2d = data2d[allowed_per_col[data2d[:, col], col]]
        return data2d

    def load_stop_words(self, stop_word_file):
        df = pd.read_excel(stop_word_file, header=None)
        convert_dict = self.CONVERT_DICT

        df.columns = ["Stoppord"]
        stop_words = [
            convert_dict[x]
            for x in df.Stoppord.values
            if convert_dict.get(x, None) is not None
        ]
        # stop_set = set(stop_words)
        uncommon_words = self.SUMMED_WORD_COUNTS < self.MINIMUM_NGRAM_COUNT

        # Create a boolean array with all words that should not be used
        remove_array = uncommon_words
        for word in stop_words:
            remove_array[word] = True
        remove_count = np.sum(remove_array)
        print(
            f"Words to remove: {remove_count} ({(remove_count / len(remove_array)) * 100:.2f}%)"
        )

        return remove_array

    def threaded_find_words(self, threads):
        global worker_fun

        def worker_fun(texts):
            word_set = set()
            for text in self.SPEECH_INDEX.text_merged.values[texts]:
                word_set.update(
                    (x for x in nltk.regexp_tokenize(text, pattern=self.REGEX_PATTERN))
                )
            return word_set

        chunks = np.array_split(range(len(self.SPEECH_INDEX)), threads)
        with Pool(threads) as pool:
            workers = pool.map_async(worker_fun, chunks)
            all_words = set()
            for d in workers.get():
                all_words.update(d)
        del worker_fun

        convert_dict = {k: v for k, v in zip(all_words, range(len(all_words)))}
        index_to_word = [k for k, v in convert_dict.items()]
        return convert_dict, index_to_word

    def vectorize_documents(self, threads: int, pattern=R"(?u)\b\w\w+\b"):
        global worker_fun

        def worker_fun(chunk):
            df = self.SPEECH_INDEX
            vectors = []
            convert_dict = self.CONVERT_DICT
            word_count = np.zeros(len(convert_dict))
            for idx in chunk:
                uid, text = df.u_id.values[idx], df.text_merged.values[idx]
                vectorized_text = np.array(
                    [
                        convert_dict[x]
                        for x in nltk.regexp_tokenize(text, pattern=pattern)
                    ],
                    dtype=np.uint32,
                )
                np.add.at(word_count, vectorized_text, 1)
                vectors.append((uid, vectorized_text))
            return (vectors, word_count)

        chunks = np.array_split(range(len(self.SPEECH_INDEX)), threads)
        total_word_counts = []
        with Pool(threads) as pool:
            results = pool.map(worker_fun, chunks)
            all_vectors = []
            for d, word_counts in results:
                all_vectors.extend(d)
                total_word_counts.append(word_counts)
        del worker_fun

        # Sum the results from all threads
        summed_word_counts = np.sum(
            np.vstack(total_word_counts), axis=0, dtype=np.uint32
        )
        return all_vectors, summed_word_counts

    def threaded_ngram_counting(self, threads):
        """
        TODO: make this thread better.
        The previous version was way faster but also quite wrong.

        Args:
            threads (_type_): _description_

        Returns:
            _type_: _description_
        """
        global worker_fun

        def worker_fun(chunk):
            all_data = self.VECTORIZED_TEX_DF.vectorized_text.values[chunk]
            filtered_data = []
            for data in all_data:
                if len(data) < self.NGRAM_LENGTH:
                    continue
                has_remove_word = self.STOP_WORD_ARRAY[data]

                data2d = np.lib.stride_tricks.sliding_window_view(
                    data, window_shape=self.NGRAM_LENGTH
                )
                has_remove_word2d = np.lib.stride_tricks.sliding_window_view(
                    has_remove_word, window_shape=self.NGRAM_LENGTH
                )
                remove_rows = np.any(has_remove_word2d, axis=1)
                data2d = data2d[~remove_rows, :]
                filtered_data.append(data2d)

            unique, count = np.unique(
                np.concatenate(filtered_data), axis=0, return_counts=True
            )

            return (unique, count)

        chunks = np.array_split(range(len(self.SPEECH_INDEX)), threads)
        with Pool(threads) as pool:
            workers = pool.map_async(worker_fun, chunks)
            all_counters = []
            for d in workers.get():
                all_counters.append(d)
        del worker_fun

        global is_sorted
        global merge_sorted_arrays

        @jit(nopython=True)
        def is_sorted(a, b):
            """Checks if two arrays are sorted. Returns 1 if a is sorted before b, -1 if b is sorted before a, 0 if they are the same."""
            for idx in range(len(a)):
                if a[idx] < b[idx]:
                    return 1
                if a[idx] > b[idx]:
                    return -1
            return 0

        @jit(nopython=True)
        def merge_sorted_arrays(a, b, ac, bc):
            """Merge two sorted arrays into one sorted array."""
            if a.shape[1] != b.shape[1]:
                raise ValueError("Error! Shapes not same!")
            merged = np.zeros((len(a) + len(b), a.shape[1]), dtype=np.uint32)
            merged_c = np.zeros(len(a) + len(b), dtype=np.uint32)
            m_idx = 0
            a_idx = 0
            b_idx = 0
            a_max = a.shape[0]
            b_max = b.shape[0]
            while a_idx < a_max and b_idx < b_max:
                res = is_sorted(a[a_idx, :], b[b_idx, :])
                if res == 0:
                    # Same row, merge count
                    merged_c[m_idx] = ac[a_idx] + bc[b_idx]
                    merged[m_idx, :] = a[a_idx, :]
                    a_idx += 1
                    b_idx += 1
                elif res == -1:
                    # b should be before a
                    merged_c[m_idx] = bc[b_idx]
                    merged[m_idx, :] = b[b_idx, :]
                    b_idx += 1
                else:
                    # a should be before b
                    merged_c[m_idx] = ac[a_idx]
                    merged[m_idx, :] = a[a_idx, :]
                    a_idx += 1
                m_idx += 1
            if a_idx < a_max:
                remaining = a_max - a_idx
                merged[m_idx : m_idx + remaining, :] = a[a_idx:, :]
                merged_c[m_idx : m_idx + remaining] = ac[a_idx:]
                m_idx += remaining
            if b_idx < b_max:
                remaining = b_max - b_idx
                merged[m_idx : m_idx + remaining, :] = b[b_idx:, :]
                merged_c[m_idx : m_idx + remaining] = bc[b_idx:]
                m_idx += remaining
            return merged[:m_idx, :], merged_c[:m_idx]

        # Merge all counters. Note that they are already sorted.
        # TODO: This is a slow single-threaded step. We can merge all pairs
        # of similar size at the same time, and then merge the results of them and so on.
        # Perhaps possible using map reduce?
        print("Merging counters into one array...")
        merged_ngram, merged_counter = None, None
        while all_counters:
            ngrams, counter = all_counters.pop()
            if merged_ngram is None:
                merged_ngram = ngrams
                merged_counter = counter
                continue
            merged_ngram, merged_counter = merge_sorted_arrays(
                merged_ngram, ngrams, merged_counter, counter
            )

        return merged_ngram, merged_counter

    def verify_allowed_in_columns(self):
        for i, col in enumerate(
            np.sum(self.ALLOWED_PER_COL, axis=0) / self.ALLOWED_PER_COL.shape[0] * 100
        ):
            print("Allowed in column ", i, ": ", f"{col:.2f}%")

    def verify_sparse(self):
        """Verify if sparse matrix seems reasonable

        Returns:
            _type_: _description_
        """

        @jit(nopython=True)
        def my_search(arr, seq):
            start = seq[0]
            seq_length = len(seq)
            for idx, val in enumerate(arr[: -seq_length + 1]):
                if val == start:
                    if np.all(arr[idx : idx + seq_length] == seq):
                        return True
            return False

        weird_ngrams = np.where(
            np.asarray(
                np.sum(self.NGRAM_SPARSE_COUNT, axis=0) < self.MINIMUM_NGRAM_COUNT
            ).flatten()
        )[0]

        for ngram in tqdm(weird_ngrams):
            print(ngram)
            ngram_data = self.ALLOWED_NGRAMS[ngram, :]
            ngram_expected_count = self.ALLOWED_COUNTER[ngram]
            # print(ngram_data)
            print("Expected count: ", ngram_expected_count)
            print(" ".join([self.INDEX_TO_WORD[x] for x in ngram_data]))
            found_in = []
            for idx, vector in enumerate(self.VECTORIZED_TEX_DF.vectorized_text.values):
                if len(vector) < self.NGRAM_LENGTH:
                    continue
                found = my_search(vector, ngram_data)
                if found:
                    uid = self.VECTORIZED_TEX_DF.iloc[idx].u_id
                    found_in.append(uid)
            if len(found_in) > 0:
                print("Found in ", len(found_in))
            else:
                print("Not found!")
            print("---")

    def verity_sparse_matrix(self):
        mat = self.NGRAM_SPARSE_COUNT
        # Get random row from mat:
        random_index = np.random.randint(0, mat.shape[0])
        print("Selected index ", random_index)
        random_row = mat[random_index, :].todense()
        # get text for this uid
        vectorized_text = self.VECTORIZED_TEX_DF.iloc[random_index].vectorized_text
        # get all ngrams matching the random_row values:
        assumed_ngrams = self.ALLOWED_NGRAMS[random_row, :]

        def find_in_text(text, ngram):
            for idx in range(len(text) - len(ngram)):
                if np.all(text[idx : idx + len(ngram)] == ngram):
                    return idx
            return -1

        for ngram in assumed_ngrams:
            print(" ".join([self.INDEX_TO_WORD[x] for x in ngram]))
            if find_in_text(vectorized_text, ngram) == -1:
                print("Not found!")

    @staticmethod
    @jit(nopython=True)
    def _count_words(word_array, maximum_words):
        counted_words = np.zeros(maximum_words, dtype=np.uint32)
        for i in word_array:
            counted_words[i] += 1
        return counted_words

    def count_words_per_group(self, groupby_argument):
        group_counts = {}
        for name, group in tqdm(self.SPEECH_INDEX.groupby(groupby_argument)):
            vectorized_text = np.concatenate(
                self.VECTORIZED_TEX_DF.iloc[group.index].vectorized_text.values
            )
            word_counts = self.__class__._count_words(vectorized_text, self.word_count)
            group_counts[name] = word_counts
        return group_counts

    def calculate_lix(self, threads: int = 24):
        """Calculate LIX. This is a measure of readability.

        The formula is:

        LIX = (O/M) + ((L * 100)/O)
        where
        O = Number of words in the text
        M = Number of sentences in the text
        L = Number of words with more than 6 letters

        Args:
            threads (int): Number of threads to use
        Returns: A list of LIX values, one for each speech in SPEECH_INDEX (in order),
            as well as a list of invalid indices (where the speech was empty or invalid)
        """
        global worker_fun

        def worker_fun(chunk):
            df = self.SPEECH_INDEX
            lix_results = np.zeros(len(chunk), dtype=np.float64)
            for i, idx in enumerate(chunk):
                text = df.text_merged.values[idx]
                #  We only do this to get the number of sentences
                sent_text = nltk.sent_tokenize(text, language="swedish")
                sent_count = len(sent_text)
                if sent_count == 0:
                    lix_results[i] = np.nan
                    continue
                # We do not use the previously vectorized text; we keep single letter words.
                tokenized_text = nltk.regexp_tokenize(text, pattern=R"(?u)\b\w+\b")
                word_count = len(tokenized_text)
                if word_count == 0:
                    lix_results[i] = np.nan
                    continue
                word_lengths = np.array(
                    [len(x) for x in tokenized_text], dtype=np.uint16
                )
                # Calculate LIX
                lix_results[i] = (word_count / sent_count) + (
                    (100 * (np.sum(word_lengths > 6)) / word_count)
                )
            return lix_results

        chunks = np.array_split(range(len(self.SPEECH_INDEX)), threads)

        with Pool(threads) as pool:
            results = pool.map(worker_fun, chunks)
            lix_per_document = np.concatenate(results)
        del worker_fun

        invalid_documents = np.where(np.isnan(lix_per_document))[0]
        lix_per_document[invalid_documents] = -1

        return lix_per_document, invalid_documents

    def get_most_used_words_per_group(self, groupby_argument, n=10000):
        """Gets the most used words per group
        TODO: Thread? Should be easy to thread this.

        Args:
            groupby_argument: The groupby argument to use on the SPEECH_INDEX
            n (int, optional): _description_. Defaults to 10000.

        Returns:
            _type_: _description_
        """
        most_used = {}
        word_counts = self.count_words_per_group(groupby_argument)
        for name, word_count in word_counts.items():
            sort_order = np.argsort(word_count)[::-1][:n]  # Largest first
            top_words = [self.INDEX_TO_WORD[i] for i in sort_order]
            top_words_count = word_count[sort_order]

            most_used[name] = {
                "top_words": top_words,
                "top_words_count": top_words_count,
                "group_total_words": np.sum(word_count),
            }
        return most_used

    @staticmethod
    def word_use_cumulative_count(most_used: dict):
        group_cumulative_sums = {}

        for k, v in most_used.items():
            twc = v["top_words_count"]
            tw = v["top_words"]

            cumulative_percentage = (np.cumsum(twc) / v["group_total_words"]) * 100
            group_cumulative_sums[k] = {
                "Word": tw,
                "Count": twc,
                "Cumulative percentage of group": cumulative_percentage,
            }
        return group_cumulative_sums

    @staticmethod
    def save_cumulative_sums_to_excel(group_cumulative_sums: dict, output_path: str):
        output_directory = f"{output_path}/word_lengths/"
        # Ensure that output_directory exists:
        import os

        os.makedirs(output_directory, exist_ok=True)
        for name, group_data in group_cumulative_sums.items():
            party, year = name
            df = pd.DataFrame(group_data)

            df.to_excel(f"{output_directory}{party}_{year}.xlsx")

    @staticmethod
    @jit(nopython=True)
    def _count_word_lengths(count, word_lengths):
        """numba function to count word lengts.

        This makes things faster (as it's in numba)
        returns: the count of words of the different lengths
        """
        word_length_counter = np.zeros(np.max(word_lengths), np.uint32)
        for idx, c in enumerate(count):
            word_length_counter[word_lengths[idx]] += c
        return word_length_counter

    def count_group_word_lengths(self, group_counts):
        """Uses the group_counts to count the word lengths

        group_counts can be gotten from count_words_per_group
        """
        word_lengths = self.WORD_LENGTH
        group_word_lengths = {}
        for name, count in group_counts.items():
            group_word_lengths[name] = self.__class__._count_word_lengths(
                count, word_lengths
            )
        return group_word_lengths

    @staticmethod
    def group_word_length_to_excel(group_word_lengths, output_directory: str):
        for name, group_lengths in group_word_lengths.items():
            party, year = name
            df = pd.DataFrame(
                {"Word length": range(len(group_lengths)), "Count": group_lengths}
            )
            total_words = df.Count.sum()
            df["Percentage"] = 100 * df.Count / total_words
            import os

            os.makedirs(f"{output_directory}/word_length_counts/", exist_ok=True)
            filename = f"{output_directory}/word_length_counts/{party}_{year}.xlsx"
            df.to_excel(filename, index=False)

    @staticmethod
    @jit(nopython=True)
    def _calc_ttr(res, vec_words):
        for w in vec_words:
            res[w] += 1
        unique = np.sum(res > 0)
        count = np.sum(res)
        # TTR = (number of unique words / total number of words) x 100
        if count > 0:
            return (unique / count) * 100
        else:
            return -1

    def ttr_function(self, threads=24):
        """Using the vectorized text values, calculate the TTR for each speech

        Note that this will ignore short words (less than 2 characters by default)

        Args:
            threads (int, optional): Number of threads to use. Defaults to 24.

        Returns:
            np.ndarray: Array of TTR values, one for each speech
        """
        word_count = self.word_count
        global worker_fun

        def worker_fun(chunk):
            texts = self.VECTORIZED_TEX_DF.vectorized_text.values[chunk]

            ttr_results = np.zeros(len(chunk), dtype=np.float64)
            res = np.zeros(word_count, dtype=np.uint16)
            for i, text in enumerate(texts):
                ttr_results[i] = self.__class__._calc_ttr(res, text)
                res[:] = 0

            return ttr_results

        chunks = np.array_split(range(len(self.VECTORIZED_TEX_DF)), threads)
        with Pool(threads) as pool:
            results = pool.map(worker_fun, chunks)
        del worker_fun

        return np.concatenate(results)
