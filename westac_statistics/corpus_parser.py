import glob
import json
import os

# import sqlite3
from multiprocessing import Pool

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize


class CorpusParser:
    """This class is used to parse the corpus and create a dataframe with the speeches.
    
    This class uses multiprocessing to parse the corpus in parallel, as quickly as possible.
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
        # Find date
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
            possible_who = set([x.get("who") for x in speech if x.name == "u"])
            if len(possible_who) > 1:
                raise Exception("More than one who detected in utterence sequence!")
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
            return pd.DataFrame(all_speeches)
        return all_speeches

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
            df = CorpusParser.get_speeches_from_soup(soup)
            df["file_name"] = xml_file
        return df

    def perform_threaded_parsing(self, threads=26):
        """Perform threaded parsing of the corpus.
        
        This uses the multiprocessing module to parse the corpus in parallel.

        Args:
            threads (int, optional): Number of threads to use. Defaults to 26.

        Returns:
            _type_: A list of dataframes with the speeches
        """
        with Pool(threads) as p:
            return p.map(self._get_speakers_df_from_file, self.xml_files)

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
        """Initialize the corpus parser. This will parse the corpus and save the result to disk.

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
