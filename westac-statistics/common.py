import pandas as pd


def load_speech_index(index_path: str, members_path: str) -> pd.DataFrame:
    """Load speech index. Merge with person index (parla. members, ministers, speakers)"""
    speech_index: pd.DataFrame = pd.read_feather(index_path)
    #     members: pd.DataFrame = pd.read_csv(members_path)
    members: pd.DataFrame = pd.read_csv(members_path, delimiter="\t").set_index("id")
    speech_index["protocol_name"] = speech_index.filename.str.split("_").str[0]
    speech_index = speech_index.merge(
        members, left_on="who", right_index=True, how="inner"
    ).fillna("")
    speech_index.loc[speech_index["gender"] == "", "gender"] = "unknown"
    return speech_index, members
