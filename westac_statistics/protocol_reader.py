import os
from pathlib import Path
from typing import List

import pandas as pd
from pandas import DataFrame

import numpy as np


class ProtocolReader:
    all_documents: List[str]
    document_dir: str
    token2id: DataFrame
    id2token: dict

    PD_SUC_PoS_tags: DataFrame

    def load_speech_index(self) -> pd.DataFrame:
        index_path = os.path.join(self.document_dir, "document_index.feather")
        members_path = os.path.join(self.document_dir, "person_index.csv")

        """Load speech index. Merge with person index (parla. members, ministers, speakers)"""
        speech_index: pd.DataFrame = pd.read_feather(index_path)
        #     members: pd.DataFrame = pd.read_csv(members_path)
        members: pd.DataFrame = pd.read_csv(members_path, delimiter="\t").set_index(
            "id"
        )
        speech_index["protocol_name"] = speech_index.filename.str.split("_").str[0]
        speech_index = speech_index.merge(
            members, left_on="who", right_index=True, how="inner"
        ).fillna("")
        speech_index.loc[speech_index["gender"] == "", "gender"] = "unknown"
        return speech_index, members

    def __init__(self, document_dir: str):
        self.document_dir = document_dir
        self.all_documents = self.get_all_documents(document_dir)
        self.token2id = pd.read_feather(
            os.path.join(self.document_dir, "token2id.feather")
        )
        self.id2token = dict(self.token2id.iloc[:, ::-1].values)

        self.speech_index, self.members = self.load_speech_index()

        self.PD_SUC_PoS_tags = (
            pd.DataFrame(
                data=[
                    ("AB", "Adverb", "Adverb"),
                    ("DT", "Pronoun", "Determinator"),
                    ("HA", "Adverb", "Adverbs (inq.)"),
                    ("HD", "Pronoun", "Det. (inq.)"),
                    ("HP", "Pronoun", "Pronoun (inq.)"),
                    ("HS", "Pronoun", "Pronoun (inq.)"),
                    ("IE", "Adverb", "Inf. mark"),
                    ("IN", "Adverb", "Interjection"),
                    ("JJ", "Adjective", "Adjectiv"),
                    ("KN", "Conjunction", "Conjunction"),
                    ("NN", "Noun", "Noun"),
                    ("PC", "Verb", "Participle"),
                    ("PL", "Adverb", "Particle"),
                    ("PM", "Noun", "Proper noun"),
                    ("PN", "Pronoun", "Pronoun"),
                    ("PP", "Preposition", "Preposition"),
                    ("PS", "Pronoun", "Poss. pron."),
                    ("RG", "Numeral", "Numeral"),
                    ("RO", "Numeral", "Numeral"),
                    ("SN", "Conjunction", "Subjuncion"),
                    ("UO", "Other", "Foreign ord"),
                    ("VB", "Verb", "Verb"),
                    ("MAD", "Delimiter", "Delimiter"),
                    ("MID", "Delimiter", "Delimiter"),
                    ("PAD", "Delimiter", "Delimiter"),
                ],
                columns=["tag", "tag_group_name", "description"],
            )
            .rename_axis("pos_id")
            .reset_index()
            .set_index("tag")
            .assign(tag=lambda x: x.index)
        )

    def decode_pos_id(self, df: DataFrame):
        return df.merge(self.PD_SUC_PoS_tags, on="pos_id")

    def get_all_documents(self, document_dir: str):
        p = Path(document_dir)
        if not p.exists():
            raise FileNotFoundError(f'No such directory "{document_dir}"')
        return list(p.glob("*/*.feather"))

    def get_protocol_path(self, year: int, protocol_name: str):
        expected_path = os.path.join(
            self.document_dir, str(year), f"{protocol_name}.feather"
        )
        if not Path(expected_path).exists():
            raise FileNotFoundError("Could not find the relevant document")
        return expected_path

    def read_protocol(self, year: int, protocol_name: str, decode=True):
        path = self.get_protocol_path(year, protocol_name)
        return self._read_protocol_path(path, decode)

    def _read_protocol_path(self, path: str, decode=True):
        df = pd.read_feather(path)
        if decode:
            df["lemma"] = df.lemma_id.map(lambda x: self.id2token[x])
            # Consider if to decode POS here or not

        return df

    @property
    def token_count(self):
        return len(self.id2token)

    @property
    def protocol_iterator(self):
        return ProtocolIterator(self)

    @property
    def document_iteractor(self):
        return DocumentIterator(ProtocolIterator(self), self)


class ProtocolIterator:
    all_documents: List[str]
    reader: ProtocolReader
    index: int

    def __init__(self, reader: ProtocolReader):
        self.reader = reader
        self.all_documents = reader.all_documents
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        try:
            val = self.reader._read_protocol_path(self.all_documents[self.index])
        except IndexError:
            raise StopIteration
        self.index += 1
        return val


class DocumentIterator:
    reader: ProtocolReader
    doc_iterator: ProtocolIterator
    curr_protocol: DataFrame
    remaining_documents: List[int]

    def __init__(self, doc_iterator: ProtocolIterator, reader: ProtocolReader):
        self.reader = reader
        self.doc_iterator = doc_iterator
        # self.speeches = reader.speech_index.document_id

    def __iter__(self):
        return self

    def __next__(self):
        if self.curr_protocol is None:
            self.curr_protocol = next(self.doc_iterator, None)
            if self.curr_protocol is None:
                raise StopIteration
            self.remaining_documents = np.unique(self.curr_protocol.document_id)
        return_document_id = self.remaining_documents.pop()
        return_df = self.curr_protocol[
            self.curr_protocol.document_id == return_document_id
        ]

        if len(self.remaining_documents == 0):
            self.curr_protocol = None

        return return_df
