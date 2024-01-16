"""
This module contains the `CorpusParser` class.

The `CorpusParser` class is designed to parse a corpus and create a dataframe with the 
speeches.

Classes:
    CorpusParser: This class parses the corpus and creates a dataframe with the speeches

"""
import glob
import json
import os

from multiprocessing import Pool

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
import json

class CorpusParser:
    """This class is used to parse the corpus and create a dataframe with the speeches.

    This class uses multiprocessing to parse the corpus in parallel, as quickly as 
    possible.
    This can lead to a high memory usage, so be careful.

    Raises:
        Exception: If more than one speaker is detected in a sequence of utterances.

    Returns:
        Either a pandas dataframe or a list of dictionaries,
        depending on the as_dataframe parameter.
    """

    speech_dataframe: pd.DataFrame
    empty_speeches: pd.DataFrame

    def __init__(self, corpus_directory, database_file):
        self.corpus_directory = corpus_directory
        self.database_file = database_file
        self.xml_files = list(
            glob.iglob(corpus_directory + "/**/*.xml", recursive=True)
        )

    @staticmethod
    def get_speeches_from_soup(soup, as_dataframe=True):
        """
        Extracts speeches from a BeautifulSoup object, which is expected to
        represent a structured document.

        Parameters:
        soup (BeautifulSoup): A BeautifulSoup object representing the
            structured document from which speeches are to be extracted.
        as_dataframe (bool, optional): If True, the function returns a
            pandas DataFrame. If False, it returns a list of dictionaries.
            Default is True.

        Returns:
        pandas.DataFrame or list: If as_dataframe is True, returns a
            DataFrame where each row represents a speech, with columns for
            the speaker, date, protocol, number of tokens, and other
            relevant information. If as_dataframe is False, returns a list
            of dictionaries, where each dictionary contains the same
            information for a single speech.

        Raises:
        Exception: If more than one speaker is detected in a single
            utterance sequence.

        Note:
        The function is designed to work with a specific document
            structure, and may not work as expected if the input
            BeautifulSoup object does not conform to this structure.
        """
        date = None
        try:
            date = pd.to_datetime(soup.find(lambda x: x.name == "docDate").text)
        except:
            pass
        protocol = None
        try:
            protocol = (
                soup.find(
                    lambda x: (x.name == "div") and x.get("type", None) == "preface"
                )
                .find(lambda x: x.name == "head")
                .text
            )
        except:
            pass

        tags = soup.findAll(
            lambda x: (x.name == "u")
            | (x.name == "note" and "type" in x.attrs and x.attrs["type"] == "speaker")
        )

        speeches = []
        current_speech = []
        all_errors = []
        for x in tags:
            if x.name == "note":
                if len(current_speech) > 0:
                    speeches.append(current_speech)
                current_speech = [x]
                continue
            current_speech.append(x)
        if len(current_speech) > 0:
            speeches.append(current_speech)
        all_speeches = []
        for speech in speeches:
            first = next(iter(speech), None)
            who = "unknown"  # Default, change if found
            who_intro = ""
            who_intro_id = ""
            if first.name == "note":
                who_intro_id = first["xml:id"]
                who_intro = first.text.strip()
            possible_who = {x.get("who") for x in speech if x.name == "u"}
            if len(possible_who) > 1:
                # print("More than one who detected in utterence sequence! (number:", len(possible_who), ")")
                # print(possible_who)
                # print(speech)
                # NOTE ! ! ! ! THIS IS A TEMPORARY WORKAROUND AND SHOULD NOT REMAIN IN THE CODE
                all_errors.append((protocol,possible_who,speech))
                possible_who = set(list(possible_who)[0])
                # raise ValueError("More than one who detected in utterence sequence!")
            if len(possible_who) == 1:
                who = list(possible_who)[0]
            # Consider counting tokens here!
            text = [" ".join(x.text.strip().split()) for x in speech if x.name == "u"]
            n_tokens = np.sum([len(word_tokenize(x)) for x in text])
            u_ids = [x["xml:id"] for x in speech if x.name == "u"]
            all_speeches.append(
                {
                    "who": who,
                    "who_intro_id": who_intro_id,
                    "who_intro": who_intro,
                    "date": date,
                    "protocol": protocol,
                    "n_tokens": n_tokens,
                    "u_id": next(iter(u_ids), ""),
                    "u_ids_json": json.dumps(u_ids, ensure_ascii=False),
                    "text_json": json.dumps(text, ensure_ascii=False),
                }
            )
        if as_dataframe:
            return pd.DataFrame(all_speeches), all_errors
        return all_speeches, all_errors

    @staticmethod
    def _get_speakers_df_from_file(xml_file):
        """Opens an xml file and returns a dataframe with the speeches.

        Args:
            xml_file (_type_): The xml file to open

        Returns:
            pandas.DataFrame: A dataframe with the speeches
        """
        with open(xml_file, encoding="utf-8") as open_file:
            text = open_file.read()
            soup = BeautifulSoup(text, features="xml")
            try:
                df, errors = CorpusParser.get_speeches_from_soup(soup)
            except Exception as e: # noqa
                print(f"Error in file {xml_file}")
                # Print error messagE:
                error_message = str(e)
                print(error_message)
                raise e
            df["file_name"] = xml_file
        if len(errors) > 0:
            
            def save_to_json(data, filename):
                # Convert list of tuples to list of dictionaries
                dict_data = []
                for protocol, possible_who, speech in data:
                    # Convert each Tag object in speech to string and join them
                    speech_str = ' '.join(tag.get_text() for tag in speech)
                    dict_data.append({'protocol': protocol, 'possible_who': list(possible_who), 'speech': speech_str})
                
                # Write to JSON file
                with open(filename, 'w') as f:
                    json.dump(dict_data, f)
            save_to_json(errors, f'{os.path.basename(xml_file)}_multiple_who_errors.json')            
        return df

    def perform_threaded_parsing(self, threads=26):
        """Perform threaded parsing of the corpus.

        This uses the multiprocessing module to parse the corpus in parallel.

        Args:
            threads (int, optional): Number of threads to use. Defaults to 26.

        Returns:
            _type_: A list of dataframes with the speeches
        """
        with Pool(threads) as pool:
            return pool.map(self._get_speakers_df_from_file, self.xml_files)

    def read_speech_dataframe_from_disk(self):
        """Reads the speech dataframe from disk.

        For this we use the feather format.

        Returns:
            pandas.DataFrame: A dataframe with the speeches
        """
        df = pd.read_feather(self.database_file)
        for col in [x for x in df.columns if x.endswith("_json")]:
            df[col.removesuffix("_json")] = df[col].apply(json.loads)
            df = df.drop(columns=col)
        df.date = pd.to_datetime(df.date)  # Ensure date is a datetime object
        return df

    def initialize(self, threads=26, force_update=False):
        """Initialize the corpus parser. This will parse the corpus and save the result
        to disk.

        Args:
            threads (int, optional): Number of threads to use. Defaults to 26.
            force_update (bool, optional): Forces an update, even if the file
                    already exists. Defaults to False.
        """
        if not os.path.exists(self.database_file) or force_update:
            per_xml_dataframes = self.perform_threaded_parsing(threads=threads)
            if os.path.exists(self.database_file):
                os.remove(self.database_file)
            df = pd.concat(per_xml_dataframes, ignore_index=True)
            df.to_feather(self.database_file)
        # Even if we just saved it, we load it to make sure it works
        self.speech_dataframe = self.read_speech_dataframe_from_disk()
        # Drop empty speeches from dataframe
        self.empty_speeches = self.speech_dataframe[self.speech_dataframe.n_tokens == 0]
        self.speech_dataframe = self.speech_dataframe.drop(self.empty_speeches.index)
