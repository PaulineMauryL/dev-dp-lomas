import io
import logging
import os
import re

import opendp as dp
import polars as pl
from opendp._lib import lib_path
from opendp.metrics import metric_distance_type, metric_type
from opendp.mod import enable_features
from opendp import measures as ms, typing as tp
from enum import IntEnum, StrEnum
from opendp.mod import enable_features
enable_features("contrib")
from opendp import measures as ms, typing as tp


OPENDP_TYPE_MAPPING = {
    "int32": tp.i32,
    "float32": tp.f32,
    "int64": tp.i64,
    "float64": tp.f64,
    "string": tp.String,
    "boolean": bool,
}

class OpenDpMechanism(StrEnum):
    """Name of OpenDP mechanisms."""

    LAPLACE = "laplace"
    GAUSSIAN = "gaussian"

OPENDP_OUTPUT_MEASURE: dict[OpenDpMechanism, tp.Measure] = {
    OpenDpMechanism.LAPLACE: ms.max_divergence(),
    OpenDpMechanism.GAUSSIAN: ms.zero_concentrated_divergence(),
}

class MetadataColumnType(StrEnum):
    """Column types for metadata."""

    STRING = "string"
    INT = "int"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    # These two are only used by pydantic to select the model to parse.
    # The pydantic models for the metadata columns never set their type to either one of these values.
    CAT_INT = "categorical_int"
    CAT_STRING = "categorical_string"



def get_lf_domain(metadata_dict: dict, plan: pl.LazyFrame) -> dp.mod.Domain:
    """
    Returns the OpenDP LazyFrame domain given a metadata dictionary.

    Args:
        metadata_dict (dict): The metadata dictionary
        plan (LazyFrame): The polars query plan as a Polars LazyFrame
    Raises:
        Exception: If there is missing information in the metadata.
    Returns:
        dp.mod.Domain: The OpenDP domain for the metadata.
    """
    # Get raw lf domain (without margins)
    lf_domain = get_raw_lf_domain(metadata_dict)

    # Add global margin to domain (for by=[])
    lf_domain = add_global_margin(lf_domain, metadata_dict)

    # Add group-by margin (if any)
    lf_domain = add_group_by_margin(plan, lf_domain, metadata_dict)

    return lf_domain


def get_raw_lf_domain(metadata_dict: dict):
    """
    Builds the "raw" lf domain from the metadata.

    The domain in considered "raw" because it does not contain any margin.
    The domain is built by putting together series domains from each column.
    """
    series_domains = []
    # Series domains
    for name, series_info in metadata_dict["columns"].items():
        series_bounds = None
        if series_info["type"] in [MetadataColumnType.FLOAT, MetadataColumnType.INT]:
            series_type = f"{series_info['type']}{series_info['precision']}"
            if hasattr(series_info, "lower") and hasattr(series_info, "upper"):
                series_bounds = (series_info["lower"], series_info["upper"])
        # TODO 392: release opendp 0.12 (adapt with type date)
        elif series_info["type"] == MetadataColumnType.DATETIME:
            series_type = MetadataColumnType.STRING
        else:
            series_type = series_info["type"]

        if series_type not in OPENDP_TYPE_MAPPING:
            # For valid metadata, only datetime would fail here
            raise InvalidQueryException(
                f"Column type {series_type} not supported by OpenDP. "
                f"Type must be in {OPENDP_TYPE_MAPPING.keys()}"
            )

        # Note: Same as using option_domain (at least how I understand it)
        series_nullable = (
            series_info["nullable_proportion"] > 0.0 and series_type != MetadataColumnType.STRING
        )
        series_type = OPENDP_TYPE_MAPPING[series_type]

        series_domain = dp.domains.series_domain(
            name,
            dp.domains.atom_domain(T=series_type, nan=series_nullable, bounds=series_bounds),
        )
        series_domains.append(series_domain)

    # Build domain from series domain
    raw_lf_domain = dp.domains.lazyframe_domain(series_domains)

    return raw_lf_domain


def add_global_margin(lf_domain, metadata: dict):
    """
    Builds the "global" (by = []) margin from the metadata
    """
    lf_domain = dp.domains.with_margin(
        lf_domain, 
        by = [],
        public_info = "lengths",
        max_partition_length = metadata["rows"],
        max_num_partitions = None,
        max_partition_contributions = metadata["max_ids"],
        max_influenced_partitions = None
    )
    return lf_domain


def get_global_params(metadata: dict) -> dict:
    """Get global parameters for margin.

    Args:
        metadata (dict): The metadata dictionary
    Returns:
        dict: Parameters for margin
    """
    margin_params = {}
    margin_params["max_num_partitions"] = 1
    margin_params["max_partition_length"] = metadata["rows"]

    return margin_params


def add_group_by_margin(plan, lf_domain, metadata: dict) -> dict:
    """
    Adds a margin for the columns in the by_configs

    Args:
        metadata (dict): The metadata dictionary.
        by_config (list): List of columns used for grouping.
    """
    # Only works with single group-by! See issue 323.
    
    # If grouping in the query, we add a margin for the group-by columns
    by_config = extract_group_by_columns(plan.explain())
    if len(by_config)==0:
        return lf_domain
            
    # Initialize max_numpartitions/max_partition_length to 1
    margin_params = {}
    margin_params["max_num_partitions"] = 1
    margin_params["max_partition_length"] = metadata["rows"]

    for column in by_config:
        series_info = metadata["columns"][column]

        # max_partitions_length logic:
        # When two columns in the grouping
        # We use as max_partition_length the smaller value
        # at the column level. If None are defined, dataset length is used.

        # Get max_partition_length from series_info, defaulting to metadata["rows"] if not set
        series_max_partition_length = (
            series_info["max_partition_length"]
            if series_info["max_partition_length"] is not None
            else metadata["rows"]
        )

        # Update the max_partition_length
        margin_params["max_partition_length"] = min(
            margin_params["max_partition_length"], series_max_partition_length
        )

        # max_num_partitions logic:
        # We multiply the cardinality defined in each column
        # If None are defined, max_num_partitions is equal to None
        if "cardinality" in series_info:
            if series_info["cardinality"]:
                margin_params["max_num_partitions"] *= series_info["cardinality"]

        # max_influenced_partitions logic:
        # We multiply the max_influenced_partitions defined in each column
        # If None are defined, max_influenced_partitions is equal to None
        if series_info["max_influenced_partitions"]:
            margin_params["max_influenced_partitions"] = (
                margin_params.get("max_influenced_partitions", 1) * series_info["max_influenced_partitions"]
            )

        # max_partition_contributions logic:
        # We multiply the max_partition_contributions defined in each column
        # If None are defined, max_partition_contributions is equal to None
        if series_info["max_partition_contributions"]:
            margin_params["max_partition_contributions"] = (
                margin_params.get("max_partition_contributions", 1)
                * series_info["max_partition_contributions"]
            )

    # If max_influenced_partitions > max_ids:
    # Then max_influenced_partitions = max_ids
    if "max_influenced_partitions" in margin_params:
        margin_params["max_influenced_partitions"] = min(
            metadata["max_ids"], margin_params["max_influenced_partitions"]
        )

    # If max_partition_contributions > max_ids:
    # Then max_partition_contributions = max_ids
    if "max_partition_contributions" in margin_params:
        margin_params["max_partition_contributions"] = min(
            metadata["max_ids"],
            margin_params.get("max_partition_contributions"),
        )

    return dp.domains.with_margin(
        lf_domain,
        by = by_config,
        public_info = "keys",
        **margin_params
    )


def extract_group_by_columns(plan: str) -> list:
    """
    Extract column names used in the BY operation from the plan string.

    Parameters:
    plan (str): The polars query plan as a string.
    Returns:
    list: A list of column names used in the BY operation.
    """
    # Regular expression to capture the content inside BY []
    aggregate_by_pattern = r"AGGREGATE(?:.|\n)+?BY \[(.*?)\]"

    # Find the part of the plan related to the GROUP BY clause
    match = re.findall(aggregate_by_pattern, plan)

    if len(match) == 1:
        # Extract the columns part
        columns_part = match[0]
        # Find all column names inside col("...")
        column_names = re.findall(r'col\("([^"]+)"\)', columns_part)
        return column_names
    if len(match) > 1:
        raise InvalidQueryException(
            "Your are trying to do multiple groupings. "
            "This is currently not supported, please use one grouping"
        )
    return []

def prepare_opendp_pipe(plan, input_data, metadata):
    lf_domain = get_lf_domain(metadata, plan)
    opendp_pipe = dp.measurements.make_private_lazyframe(
        lf_domain, dp.metrics.symmetric_distance(), ms.max_divergence(), plan
    )
    expressions = []
    for col, val in metadata["columns"].items():
        if val["type"] in [MetadataColumnType.STRING, MetadataColumnType.DATETIME]:
            expressions.append(pl.col(col).fill_null("").alias(col))
    
    input_data = input_data.with_columns(expressions)

    return opendp_pipe, input_data
    

def get_lf_from_df(df_dummy):
    df_copy = df_dummy.copy()
    for col in df_copy.select_dtypes(include=["datetime"]):
        df_copy[col] = df_copy[col].astype("string[python]")
    lf_dummy = pl.from_pandas(df_copy).lazy()
    return lf_dummy

