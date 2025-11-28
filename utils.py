import copy
import polars as pl


def make_smartnoise_metadata(metadata):
    metadata_dict = copy.deepcopy(metadata)
    for name, val in metadata_dict["columns"].items():
        already_in.append(name)
        if val["private_id"] or val["type"] == "datetime":
            for k in ["lower", "upper"]:
                if val.get(k) is not None:
                    del val[k]
    metadata_dict.update(metadata_dict["columns"])
    del metadata_dict["columns"]
    metadata_dict = {"": {"": {"df": metadata_dict}}}
    return metadata_dict


def get_df_types_from_metadata(metadata):
    dtypes = {}
    datetime_columns = []
    for col_name, data in metadata["columns"].items():
        if data["type"] == "datetime":
            dtypes[col_name] = "string"
            datetime_columns.append(col_name)
        elif hasattr(data, "precision"):
            dtypes[col_name] = f"{data["type"]}{data["precision"]}"
        else:
            dtypes[col_name] = data["type"]
    return dtypes, datetime_columns


def get_lf_from_df(df):
    df_copy = df.copy()
    for col in df_copy.select_dtypes(include=["datetime"]):
        df_copy[col] = df_copy[col].astype("string[python]")
    lf_dummy = pl.from_pandas(df_copy).lazy()
    return lf_dummy