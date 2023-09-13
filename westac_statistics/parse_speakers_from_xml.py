import glob
import sqlite3
import uuid
from contextlib import closing
from os import path
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from tqdm.auto import tqdm

import warnings
warnings.warn("The ParseSpeakersFromXML class is deprecated and will be removed in future versions.", DeprecationWarning)


@DeprecationWarning
class ParseSpeakersFromXML:
    dataframes: pd.DataFrame
    database_file: str

    def __init__(self, corpus_directory: str, database_file):
        self.xml_files = list(
            glob.iglob(corpus_directory + "/**/*.xml", recursive=True)
        )
        self.database_file = database_file

        if ~path.exists(self.database_file):
            with closing(sqlite3.connect(self.database_file)) as con:
                with closing(con.cursor()) as cur:
                    cur.execute(
                        """CREATE TABLE IF NOT EXISTS speakers
                           (u_id text,
                           file text,
                           speaker text,
                           next_u text,
                           first_seg text,
                           PRIMARY KEY(u_id)
                           )"""
                    )
                    cur.execute(
                        """CREATE TABLE IF NOT EXISTS speaker_note
                                   (speaker_note_id text,
                                   who text,
                                   note_text text,
                                   PRIMARY KEY(speaker_note_id)
                                   )"""
                    )
        self.dataframes = dict()

    def is_initialized(self):
        if not Path(self.database_file).is_file():
            return False
        if len(self.db_execute("SELECT * from speakers limit 1")) == 2:
            return True
        return False

    def is_df_initialized(self):
        val, _ = self.db_execute('SELECT name FROM sqlite_master where type="table"')
        dataframe_names = [x[0] for x in val if x[0].endswith("_df")]
        return len(dataframe_names) > 0

    def initialize(self, force_init=False):
        if not self.is_initialized() or force_init:
            print("Corpus XML not parsed! Parsing...")
            self.initialize_speakers()
        if not self.is_df_initialized():
            print("Dataframes not in database! Creating and adding...")
            self.load_dataframes()
        else:
            with closing(sqlite3.connect(self.database_file)) as con:
                val, _ = self.db_execute(
                    'SELECT name FROM sqlite_master where type="table"'
                )
                dataframe_names = [x[0] for x in val if x[0].endswith("_df")]
                for name in dataframe_names:
                    self.dataframes[name] = pd.read_sql(
                        sql=f"SELECT * FROM {name}", con=con
                    )

    # Only keep speakers if they have a following utterance
    @staticmethod
    def _get_valid_speakers(soup):
        tags = soup.findAll(
            lambda x: (
                x.name == "note" and "type" in x.attrs and x["type"] == "speaker"
            )
            or x.name == "u"
        )
        tags = list(tags)
        valid_tags = []
        for i, tag in enumerate(tags[:-2]):
            next_tag = tags[i + 1]
            if tag.name == "note" and next_tag.name == "u":
                valid_tags.append(tag)
        return valid_tags

    def initialize_speakers(self):
        for xml_file in tqdm(self.xml_files):
            with open(xml_file, encoding="utf-8") as open_file:
                with closing(sqlite3.connect(self.database_file)) as con:
                    with closing(con.cursor()) as cur:
                        text = open_file.read()
                        soup = BeautifulSoup(text, features="xml")

                        speakers = self._get_valid_speakers(soup)
                        for speaker in speakers:
                            next_u = speaker.findNext(lambda x: x.name == "u")
                            if next_u is not None:
                                first_seg = next_u.findNext(lambda x: x.name == "seg")
                                u_id = next_u["xml:id"]
                            else:
                                next_u = None
                                first_seg = None
                                u_id = str(uuid.uuid4())
                            try:
                                cur.execute(
                                    "INSERT INTO speakers (u_id, file, speaker, next_u, first_seg) VALUES (?,?,?,?,?)",
                                    (
                                        u_id,
                                        xml_file,
                                        str(speaker),
                                        str(next_u),
                                        str(first_seg),
                                    ),
                                )
                            except:
                                con.commit()
                                raise sqlite3.DatabaseError(
                                    f"DB insert failed, u_id: {u_id}"
                                )
                        con.commit()

    def db_execute(self, sql_query: str, *args):
        with closing(sqlite3.connect(self.database_file)) as con:
            with closing(con.cursor()) as cur:
                res = cur.execute(sql_query, args)
                return res.fetchall(), [x[0] for x in cur.description]

    def get_all_from_db_who(self, who: str):
        return self.db_execute("SELECT * FROM speakers WHERE speaker == (?)", who)

    def get_all_from_db_id(self, u_id: str):
        return self.db_execute("SELECT * FROM speakers WHERE xml_id IS (?)", u_id)

    @staticmethod
    def _join_on(
        left_df: pd.DataFrame, right_df: pd.DataFrame, column, delete_join=False
    ):
        len_before = len(left_df)
        combined_df = (
            left_df.set_index(column)
            .join(right_df.set_index(column), how="left")
            .reset_index()
        )
        if len(combined_df) != len_before:
            print(
                f"Number of rows changed, before: {len_before:10}, after: {len(combined_df):10}, difference: {(len(combined_df) - len_before):10}"
            )
        if delete_join:
            return combined_df.drop(columns=column)
        return combined_df

    def get_text_from_uid(self, u_id, strip_whitespace=True):
        full_df = self.dataframes["full_df"]
        uid_df = full_df[full_df.u_id == u_id]
        file = uid_df.file.values[0]
        who = uid_df.who.values[0]

        with open(file, encoding="utf-8") as open_file:
            text = open_file.read()
            soup = BeautifulSoup(text, features="xml")

        tags = soup.findAll(
            lambda x: (x.name == "u" and "prev" in x.attrs and x["prev"] == u_id)
            | (x.name == "u" and "xml:id" in x.attrs and x["xml:id"] == u_id)
        )
        # print(tags)
        result = []

        # If unknown, just append all u tags until the next note with a speaker appears
        if who == "unknown" and len(tags) == 1:
            tag = tags[0]
            for x in tag.find_next_siblings({"u", "note"}):
                # print(x)
                if (
                    x.name == "note"
                    and "type" in x.attrs
                    and x.attrs["type"] == "speaker"
                ):
                    break
                if x.name == "u":
                    tags.append(x)

        for tag in tags:
            seg_list = tag.findAll(lambda x: (x.name == "seg"))
            for seg in seg_list:
                if strip_whitespace:
                    result.append(seg.text.strip())
                else:
                    result.append(seg.text)
        if strip_whitespace:
            return [x for x in [" ".join(x.split()) for x in result]]
        return result

    def load_dataframes(self):
        results, columns = self.db_execute("SELECT u_id, speaker, next_u FROM speakers")

        who_and_xml_id = []
        for u_id, speaker, next_u in tqdm(results):
            next_u_soup = BeautifulSoup(next_u, features="xml")
            speaker_soup = BeautifulSoup(speaker, features="xml")

            who_and_xml_id.append(
                {
                    "u_id": u_id,
                    "who": next_u_soup.find("u").get("who"),
                    "speaker_note_id": speaker_soup.find("note").get("xml:id"),
                    "note_text": speaker_soup.text.strip(),
                }
            )
        self.dataframes["speaker_df"] = pd.DataFrame(who_and_xml_id)

        speakers, columns = self.db_execute("SELECT * FROM speakers")
        self.dataframes["speaker_note_df"] = pd.DataFrame(
            [{y: x for x, y in zip(speaker, columns)} for speaker in speakers]
        )
        self.dataframes["full_df"] = self._join_on(
            self.dataframes["speaker_df"],
            self.dataframes["speaker_note_df"],
            column="u_id",
        )

        with closing(sqlite3.connect(self.database_file)) as con:
            for df_name in self.dataframes.keys():
                self.dataframes[df_name].to_sql(
                    name=df_name,
                    con=con,
                    if_exists="replace",
                    index=False,
                    dtype="text",
                )
