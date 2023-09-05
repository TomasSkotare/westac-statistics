import glob
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from tqdm.auto import tqdm


class MetadataParser:
    def __init__(self, metadata_directory: str):
        self.csv_files = list(
            glob.iglob(metadata_directory + "/**/*.csv", recursive=True)
        )
        self.metadata = {}

    def initialize(self):
        for x in self.csv_files:
            self.metadata[Path(x).stem] = pd.read_csv(x)
        aff = self.metadata["party_affiliation"]
        aff["start_dt"] = aff.start.apply(pd.to_datetime)
        aff["end_dt"] = aff.end.apply(pd.to_datetime)
        self.metadata["party_affiliation"] = aff

        self.__fix_name_metadata()
        self.__fix_person_metata()

    @staticmethod
    def compare_dates(a, b):
        if pd.isnan(a) or pd.isnan(b):
            return np.inf
        return

    @staticmethod
    def get_closest_to_date(df, date):
        # These can crash in the case that *all* end or start dates are not a time, just account for that
        try:
            start_closest = np.argmin((df.start_dt - date).abs())
        except:
            start_closest = 0
        try:
            end_closest = np.argmin((df.end_dt - date).abs())
        except:
            end_closest = 0
        return df.iloc[
            start_closest
            if df.start_dt.values[start_closest] > df.end_dt.values[end_closest]
            else end_closest
        ].party

    def get_affiliation_from_dates(self, who, dates):
        """
        Gets affiliation from a dataframe.
        This includes a "uncertain" flag, which specifies if the party affiliation is an approximation
        or not.
        note that "unknown" and "unknown missing" are two valid options that are not uncertain.
        Uncertain values are when the specified date is not within the range of any specific start-stop date.
        In such cases, we have decided that the closest start/stop date from the other options should decide,
        and the uncertain flag is set to True.
        """
        aff = self.metadata["party_affiliation"]
        if who == "unknown":
            return [("unknown", False, "Unknown person") for x in range(len(dates))]
        df = aff[aff.wiki_id == who]
        if len(df) == 0:
            print(f"{who} has no entry in party_affiliation!")
            return [("unknown_missing", False, "No listing") for x in range(len(dates))]
        if len(df) == 1:
            # Regardless of start/stop date, if there is only one option we choose it
            # NOTE: We consider this certain, even though it may not be in reality.
            # The reason for this choice is that the vast majority of politicians only
            # have one party, and are missing start and/or stop dates. Leaving this uncertain
            # will lead to a close to 100% overall count of uncertain values.
            return [
                (df.party.values[0], False, "Only one match") for x in range(len(dates))
            ]

        n_unique_parties = df.party.nunique()
        results = []
        for date in dates:
            result = df[(df.start_dt <= date) & (df.end_dt >= date)]
            if len(result) == 1:  # Perfect match found!
                results.append((result.party.values[0], False, "Perfect match found"))
                continue
            if n_unique_parties == 1:
                results.append(
                    (
                        df.party.values[0],
                        False,
                        "Only one unique party possible, no matching date",
                    )
                )
                continue
            if (
                len(result) == 0
            ):  # No certain match!In this case, the uncertain flag is true!
                results.append(
                    (
                        MetadataParser.get_closest_to_date(df, date),
                        True,
                        f"Estimated closest match ({n_unique_parties} possible parties)",
                    )
                )
                continue

            # If more than 1 match is found, take the last and set uncertain flag to True
            results.append(
                (
                    result.party.values[-1],
                    True,
                    "More than one possible party found, had to guess",
                )
            )
        return results

    @staticmethod
    def join_on(df1, df2, column, delete_join=False):
        len_before = len(df1)
        df = df1.set_index(column).join(df2.set_index(column), how="left").reset_index()
        if len(df) != len_before:
            print(
                f"Number of rows changed, before: {len_before:10}, after: {len(df):10}, difference: {(len(df) - len_before):10}"
            )
        if delete_join:
            return df.drop(columns=column)
        else:
            return df

    def __fix_name_metadata(self):
        names = self.metadata["name"]
        # Sort, keeping primary name first
        names = names.sort_values(by=["wiki_id", "primary_name"], ascending=False)
        primary_names = names[["wiki_id", "name"]].drop_duplicates(
            subset="wiki_id", ignore_index=True, keep="first"
        )
        all_names = names.groupby("wiki_id").agg({"name": list})

        combined_names = self.join_on(
            primary_names,
            all_names.rename(columns={"name": "alternative_names"}).reset_index(),
            column="wiki_id",
        )

        self.metadata["name"] = combined_names

    @staticmethod
    def __convert_date(date: str):
        """Converts a date string to a datetime object.
        
        Is supposed to handle different types of date formats.
        This was default behavious in earlier versions of Pandas but has since been removed.
        
        This handles variants of types:
            - 2020
            - 2020-01
            - 2020-01-01

        Args:
            date (str): The date, as a string

        Raises:
            ValueError: If no valid date format is found

        Returns:
            DateTime: A pandas datetime object
        """
        for fmt in ('%Y', '%Y-%m-%d', '%Y-%m'):
            try:
                return pd.to_datetime(date, format=fmt)
            except ValueError:
                pass
        raise ValueError(f'no valid date format found for string {date}')     

    def __fix_person_metata(self):
        df = self.metadata["person"]
        df.born = df.born.astype(str)
        df.dead = df.dead.astype(str)

        df_g = df.groupby("wiki_id").agg(list)

        df_g.born = df_g.born.apply(
            lambda x: sorted(x, key=len, reverse=False)[0]
        )  # Get longest possible date
        df_g.born.apply(self.__convert_date)

        df_g.dead = df_g.dead.apply(
            lambda x: sorted(x, key=len, reverse=False)[0]
        )  # Get longest possible date
        df_g.dead.apply(self.__convert_date)

        # This ensures that there is no instance where a column has more than one unique value
        # in the "name" column, e.g. to ensure that a person has at most one guid and id
        def _ensure_only_one(df, name):
            if np.any(
                df[name].apply(lambda x: len(np.unique(x))) == 0
            ):  # TEMPORARY WORKAROUND DUE TO MORE THAN ONE GUID
                raise Exception(f"Unexpected {name}, verify!")
            df[name] = df[name].apply(lambda x: x[0])
            return df

        df_g = _ensure_only_one(df_g, "gender")
        # df_g = _ensure_only_one(df_g, "riksdagen_guid") # This was removed in 0.5.0
        df_g = _ensure_only_one(df_g, "riksdagen_id")

        self.metadata["person"] = df_g.reset_index()

    def add_affiliation(self, speech_dataframe):
        """
        TODO: Add some threading here to improve time
        """
        speech_dataframe["party_affiliation"] = "unknown"
        for name, group in tqdm(speech_dataframe[["who", "date"]].groupby("who")):
            date_grouping = (
                group.reset_index()
                .groupby("date")
                .agg({"index": list})
                .rename(columns={"index": "indexes"})
            )
            dates = date_grouping.index.values
            try:
                affiliation_per_date = self.get_affiliation_from_dates(name, dates)
            except Exception as e:
                print(f"Exception at {name}")
                raise e
            for idx, (affil, uncertain, message) in enumerate(affiliation_per_date):
                indexes = date_grouping.iloc[idx].values[0]
                speech_dataframe.loc[indexes, "party_affiliation"] = affil
                speech_dataframe.loc[indexes, "party_affiliation_uncertain"] = uncertain
                speech_dataframe.loc[indexes, "party_affiliation_message"] = message
                
                
           