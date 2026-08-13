import json
import os
import pickle

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .config import CorpusConfig, load_config


# Fallback abbreviations for parties missing from party_abbreviation.csv.
# Maps full party name (as it appears in party_affiliation.csv) to abbreviation.
# Historical parties without standard abbreviations are omitted; they fall
# through to using the full party name as the abbreviation.
PARTY_ABBREV_FALLBACK = {
    # Modern parties with known abbreviations
    "Moderata samlingspartiet": "M",
    "Folkpartiet liberalerna": "L",
    "Feministiskt initiativ": "FI",
    "Alternativ för Sverige": "AfS",
    "Klimatalliansen": "KA",
    "MoD": "MoD",
    "Utan partibeteckning": "np",
    # Historical parties with known abbreviations
    "Högerpartiet": "H",
    "Högerpartiet de konservativa": "H",
    "Högerns riksdagsgrupp": "H",
    "Lantmannapartiet": "Lp",
    "Gamla lantmannapartiet": "Lp",
    "Nya lantmannapartiet": "Lp",
    "Frisinnade folkpartiet": "FrFP",
    "Frisinnade landsföreningen": "FrL",
    "Frisinnade försvarsvänner": "FrF",
    "Liberala samlingspartiet": "LS",
    "Liberala riksdagspartiet": "LR",
    "Folkpartiet (1895\u20131900)": "L",
    "Vänsterpartiet Kommunisterna": "VK",
    "vänstern": "V",
    "Kommunistiska partiet": "K",
    "Sveriges kommunistiska parti": "SKP",
    "Sverges kommunistiska parti": "SKP",
    "Socialistiska partiet": "Sp",
    "Socialdemokratiska vänstergruppen": "SDV",
    "Centern (partigrupp 1873-1882)": "C",
    "Centern (partigrupp 1885-1887)": "C",
    "Nya centern (partigrupp 1883-1887)": "C",
    "nya centern (partigrupp 1895-1896)": "C",
    "Andra kammarens center": "C",
    "Andra kammarens vänster": "V",
    "Andra kammarens frihandelsparti": "Fr",
    "Det förenade högerpartiet": "FH",
    "De moderata reformvännernas grupp": "MR",
    "Första kammarens moderata parti": "Fm",
    "Första kammarens konservativa grupp": "Fk",
    "Första kammarens nationella parti": "Fn",
    "Första kammarens protektionistiska parti": "Fp",
    "Första kammarens ministeriella grupp": "Fm",
    "Första kammarens minoritetsparti": "Fm",
    "Center-högern": "CH",
    "Frihandelsvänliga centern": "FC",
    "Lantmanna- och borgarepartiet inom andrakammaren": "LB",
    "Medborgerlig samling (1964\u20131968)": "MS",
    "Nyliberala partiet": "NyL",
    "Kristdemokratiska Samhällspartiet": "KD",
    "Kristdemokrater i Svenska kyrkan": "KD",
    "Miljöpartister i Svenska kyrkan De Gröna": "MP",
    "Socialdemokrater för tro och solidaritet": "S",
    "Partiet Vändpunkt": "VP",
    "Partipolitiskt obundna i Svenska kyrkan": "np",
    "Bondeska diskussionsklubben": "BD",
    "Friesenska diskussionsklubben": "FD",
    "Centerpartiets ungdomsförbund": "CU",
    "Jordbrukarnas fria grupp": "JG",
    "Junkerpartiet": "J",
    "Kilbomspartiet": "K",
    "Mannerheimska partiet": "M",
    "Nationella framstegspartiet": "NF",
    "Skånska partiet": "Sk",
    "Stockholmsbänken": "SB",
    "Sveriges nationella förbund": "SN",
    "borgmästarepartiet": "BP",
    "ministeriella partiet": "MP",
    "Ehrenheimska partiet": "EP",
    "Europeiska socialdemokratiska partiet": "ESP",
    "National Organization of the Right": "NOR",
}


def _parse_date(v):
    if pd.isna(v) or v == "" or v == "nan":
        return pd.NaT
    s = str(v).strip()
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return pd.Timestamp(s)
        except ValueError:
            continue
    return pd.NaT


def _build_lookup_tables(persons_path: str):
    names_df = pd.read_csv(os.path.join(persons_path, "name.csv"))
    party_aff_df = pd.read_csv(os.path.join(persons_path, "party_affiliation.csv"))
    party_abbrev_df = pd.read_csv(os.path.join(persons_path, "party_abbreviation.csv"))
    person_df = pd.read_csv(os.path.join(persons_path, "person.csv"))

    primary_sorted = names_df.sort_values(
        by=["person_id", "primary_name"], ascending=[True, False]
    )
    name_lookup = {}
    alt_names_lookup = {}
    for _, row in primary_sorted.iterrows():
        pid = row["person_id"]
        if pid not in name_lookup:
            name_lookup[pid] = row["name"]
            alt_names_lookup[pid] = []
        elif not row["primary_name"]:
            alt_names_lookup.setdefault(pid, []).append(row["name"])

    abbrev_map = dict(PARTY_ABBREV_FALLBACK)
    for _, row in party_abbrev_df.iterrows():
        abbrev_map[row["party"]] = row["abbreviation"]

    aff_df = party_aff_df.copy()
    aff_df["start_dt"] = aff_df["start"].apply(_parse_date)
    aff_df["end_dt"] = aff_df["end"].apply(_parse_date)
    aff_df["abbrev"] = aff_df["party"].map(lambda p: abbrev_map.get(p, p))
    aff_by_person = {}
    for pid, group in aff_df.groupby("person_id"):
        aff_by_person[pid] = group.sort_values("start_dt", na_position="first")

    gender_lookup = {}
    born_lookup = {}
    dead_lookup = {}
    for _, row in person_df.iterrows():
        pid = row["person_id"]
        g = row.get("gender")
        if pd.notna(g) and pid not in gender_lookup:
            gender_lookup[pid] = str(g)
        b = row.get("born")
        if pd.notna(b) and pid not in born_lookup:
            born_lookup[pid] = _parse_date(b)
        d = row.get("dead")
        if pd.notna(d) and pid not in dead_lookup:
            dead_lookup[pid] = _parse_date(d)

    return (
        name_lookup,
        alt_names_lookup,
        aff_by_person,
        abbrev_map,
        gender_lookup,
        born_lookup,
        dead_lookup,
    )


def _get_closest_to_date(df, date):
    """Original get_closest_to_date logic, ported."""
    try:
        start_closest = np.argmin((df.start_dt - date).abs())
    except Exception:
        start_closest = 0
    try:
        end_closest = np.argmin((df.end_dt - date).abs())
    except Exception:
        end_closest = 0
    return df.iloc[
        start_closest
        if df.start_dt.values[start_closest] > df.end_dt.values[end_closest]
        else end_closest
    ]


class MetadataLoader:
    def __init__(self, config: CorpusConfig | None = None):
        self.config = config or load_config()
        self.cache_file = os.path.join(
            self.config.cache_dir,
            f"metadata_{self.config.persons_version.replace('.', '_')}.pkl",
        )
        self.name_lookup = {}
        self.alt_names_lookup = {}
        self.aff_by_person = {}
        self.abbrev_map = {}
        self.gender_lookup = {}
        self.born_lookup = {}
        self.dead_lookup = {}

    def _cache_valid(self) -> bool:
        if not os.path.exists(self.cache_file):
            return False
        meta_file = self.cache_file + ".meta"
        if not os.path.exists(meta_file):
            return False
        try:
            with open(meta_file) as f:
                meta = json.load(f)
            return meta.get("persons_version") == self.config.persons_version
        except Exception:
            return False

    def initialize(self):
        if self._cache_valid():
            with open(self.cache_file, "rb") as f:
                (
                    self.name_lookup,
                    self.alt_names_lookup,
                    self.aff_by_person,
                    self.abbrev_map,
                    self.gender_lookup,
                    self.born_lookup,
                    self.dead_lookup,
                ) = pickle.load(f)
            return

        (
            self.name_lookup,
            self.alt_names_lookup,
            self.aff_by_person,
            self.abbrev_map,
            self.gender_lookup,
            self.born_lookup,
            self.dead_lookup,
        ) = _build_lookup_tables(self.config.persons_path)
        os.makedirs(os.path.dirname(self.cache_file) or ".", exist_ok=True)
        with open(self.cache_file, "wb") as f:
            pickle.dump(
                (
                    self.name_lookup,
                    self.alt_names_lookup,
                    self.aff_by_person,
                    self.abbrev_map,
                    self.gender_lookup,
                    self.born_lookup,
                    self.dead_lookup,
                ),
                f,
                protocol=5,
            )
        meta = {
            "persons_version": self.config.persons_version,
            "name_count": len(self.name_lookup),
            "aff_count": len(self.aff_by_person),
        }
        with open(self.cache_file + ".meta", "w") as f:
            json.dump(meta, f)

    def resolve_speaker_name(self, who_id: str) -> str:
        if who_id == "unknown" or who_id == "unknown_no_xml_id":
            return "unknown"
        return self.name_lookup.get(who_id, "unknown_not_in_metadata")

    def resolve_party_for_speech(self, who_id: str, date) -> tuple:
        if who_id == "unknown" or who_id == "unknown_no_xml_id":
            return "unknown", "unknown", False, "Unknown speaker"
        aff = self.aff_by_person.get(who_id)
        if aff is None or len(aff) == 0:
            return "unknown_missing", "unknown_missing", False, "No affiliation data"
        if len(aff) == 1:
            row = aff.iloc[0]
            return row["party"], row["abbrev"], False, "Only one match"
        if pd.isna(date):
            row = aff.iloc[0]
            return row["party"], row["abbrev"], True, "No date, used first party"
        if isinstance(date, str):
            date = pd.Timestamp(date)
        n_unique = aff.party.nunique()
        result = aff[(aff.start_dt <= date) & (aff.end_dt >= date)]
        if len(result) == 1:
            row = result.iloc[0]
            return row["party"], row["abbrev"], False, "Perfect match found"
        if n_unique == 1:
            row = aff.iloc[0]
            return (
                row["party"],
                row["abbrev"],
                False,
                "Only one unique party possible, no matching date",
            )
        if len(result) == 0:
            closest = _get_closest_to_date(aff, date)
            return (
                closest["party"],
                closest["abbrev"],
                True,
                f"Estimated closest match ({n_unique} possible parties)",
            )
        row = result.iloc[-1]
        return (
            row["party"],
            row["abbrev"],
            True,
            "More than one possible party found, had to guess",
        )

    def _get_affiliation_from_dates(self, who_id: str, dates) -> list:
        """Port of old get_affiliation_from_dates — vectorized per-speaker masks."""
        if who_id == "unknown" or who_id == "unknown_no_xml_id":
            return [("unknown", "unknown", False, "Unknown person") for _ in dates]

        aff = self.aff_by_person.get(who_id)
        if aff is None or len(aff) == 0:
            return [
                ("unknown_missing", "unknown_missing", False, "No listing")
                for _ in dates
            ]

        if len(aff) == 1:
            row = aff.iloc[0]
            return [
                (row["party"], row["abbrev"], False, "Only one match") for _ in dates
            ]

        n_unique_parties = aff.party.nunique()
        results = []
        for date in dates:
            dt = pd.Timestamp(date) if not isinstance(date, pd.Timestamp) else date
            result = aff[(aff.start_dt <= dt) & (aff.end_dt >= dt)]
            if len(result) == 1:
                row = result.iloc[0]
                results.append(
                    (row["party"], row["abbrev"], False, "Perfect match found")
                )
                continue
            if n_unique_parties == 1:
                row = aff.iloc[0]
                results.append(
                    (
                        row["party"],
                        row["abbrev"],
                        False,
                        "Only one unique party possible, no matching date",
                    )
                )
                continue
            if len(result) == 0:
                closest = _get_closest_to_date(aff, dt)
                results.append(
                    (
                        closest["party"],
                        closest["abbrev"],
                        True,
                        f"Estimated closest match ({n_unique_parties} possible parties)",
                    )
                )
                continue
            row = result.iloc[-1]
            results.append(
                (
                    row["party"],
                    row["abbrev"],
                    True,
                    "More than one possible party found, had to guess",
                )
            )
        return results

    def add_affiliation(self, speech_dataframe: pd.DataFrame) -> pd.DataFrame:
        speech_dataframe = speech_dataframe.copy()
        speech_dataframe["party_affiliation"] = "unknown"
        speech_dataframe["party_abbrev"] = "unknown"
        speech_dataframe["party_affiliation_uncertain"] = True
        speech_dataframe["party_affiliation_message"] = ""
        speech_dataframe["name"] = "unknown"
        speech_dataframe["alternative_names"] = [[] for _ in range(len(speech_dataframe))]
        speech_dataframe["gender"] = "unknown"

        # Resolve name/alt_names/gender per unique speaker (fast dict lookup)
        unique_who = speech_dataframe["who"].unique()
        name_map = {}
        alt_map = {}
        gender_map = {}
        for who_id in tqdm(unique_who, desc="Resolving names"):
            name_map[who_id] = self.resolve_speaker_name(who_id)
            alt_map[who_id] = self.alt_names_lookup.get(who_id, [])
            gender_map[who_id] = self.gender_lookup.get(who_id, "unknown")

        speech_dataframe["name"] = speech_dataframe["who"].map(name_map).fillna(
            "unknown"
        )
        speech_dataframe["alternative_names"] = speech_dataframe["who"].apply(
            lambda w: alt_map.get(w, [])
        )
        speech_dataframe["gender"] = speech_dataframe["who"].map(gender_map).fillna(
            "unknown"
        )

        # Resolve party affiliation — collect all assignments, then bulk .loc
        all_indexes = []
        all_party = []
        all_abbrev = []
        all_uncertain = []
        all_message = []

        for who_id, group in tqdm(
            speech_dataframe[["who", "date"]].groupby("who"),
            desc="Resolving affiliations",
        ):
            date_grouping = (
                group.reset_index()
                .groupby("date")
                .agg({"index": list})
                .rename(columns={"index": "indexes"})
            )
            dates = date_grouping.index.values
            try:
                affiliation_per_date = self._get_affiliation_from_dates(who_id, dates)
            except Exception as e:
                print(f"Exception at {who_id}")
                raise e
            for idx_pos, (affil, abbrev, uncertain, message) in enumerate(
                affiliation_per_date
            ):
                indexes = date_grouping.iloc[idx_pos].values[0]
                all_indexes.extend(indexes)
                all_party.extend([affil] * len(indexes))
                all_abbrev.extend([abbrev] * len(indexes))
                all_uncertain.extend([uncertain] * len(indexes))
                all_message.extend([message] * len(indexes))

        speech_dataframe.loc[all_indexes, "party_affiliation"] = all_party
        speech_dataframe.loc[all_indexes, "party_abbrev"] = all_abbrev
        speech_dataframe.loc[all_indexes, "party_affiliation_uncertain"] = all_uncertain
        speech_dataframe.loc[all_indexes, "party_affiliation_message"] = all_message

        return speech_dataframe

    def enrich_speech_dataframe(self, speech_dataframe: pd.DataFrame) -> pd.DataFrame:
        speech_dataframe = self.add_affiliation(speech_dataframe)

        # Add born/dead from person metadata (fast dict lookup per unique speaker)
        unique_who = speech_dataframe["who"].unique()
        born_map = {w: self.born_lookup.get(w) for w in unique_who}
        dead_map = {w: self.dead_lookup.get(w) for w in unique_who}
        speech_dataframe["born"] = speech_dataframe["who"].map(born_map)
        speech_dataframe["dead"] = speech_dataframe["who"].map(dead_map)

        speech_dataframe["year"] = speech_dataframe["date"].dt.year
        speech_dataframe["decade"] = (
            speech_dataframe["year"] - speech_dataframe["year"] % 10
        )
        speech_dataframe["text_merged"] = speech_dataframe["text"].apply(
            lambda x: " ".join(iter(x)).lower() if isinstance(x, list) else str(x).lower()
        )
        if "u_id" not in speech_dataframe.columns:
            speech_dataframe["u_id"] = range(len(speech_dataframe))
        return speech_dataframe
