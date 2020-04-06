import pandas as pd
from pathlib import Path

from imlib.pandas.misc import regex_remove_df_columns


def load_and_clean_dlc(dlc_files, regex_remove_columns=None):
    """
    Load N dlc files, and clean up the column names
    :param dlc_files: A list of dlc files, in order
    :param search_string_list: A list of regex strings to search for.
    Columns matching these will be removed
    :return: A single dataframe with all output, and informative column names
    """
    dlc_data = load_multiple_dlc(dlc_files)
    dlc_data = make_dlc_columns(dlc_data)
    if regex_remove_columns is not None:
        dlc_data = regex_remove_df_columns(dlc_data, regex_remove_columns)
    return dlc_data


def load_multiple_dlc(dlc_files):
    """
    Load multiple deeplabcut files (when the output is split between files
    :param dlc_files: A list of dlc files, in order
    :return: A single dataframe with all output
    """

    dlc_data = read_dlc(dlc_files[0])
    for dlc_file in dlc_files[1:]:
        tmp_df = read_dlc(dlc_file, drop_first_column=True)
        dlc_data = dlc_data.append(tmp_df)

    return dlc_data


def read_dlc(dlc_file, drop_first_column=False):
    """
    Read DeepLabCut output, regardless of format
    :param dlc_file: DeepLabCut output file
    :param drop_first_column: Remove the first column with timepoint index
    :return: Dataframe of the DLC results
    """

    file_type = Path(dlc_file).suffix
    if file_type == ".xlsx":
        df = pd.read_excel(dlc_file)
    elif file_type == ".csv":
        df = pd.read_csv(dlc_file)
    else:
        raise TypeError(f"Filetype: '{Path(file_type)} is not supported")

    if drop_first_column:
        df = df.iloc[2:]

    return df


def make_dlc_columns(
    dlc_df, remove_header_rows=True, reset_index=True, drop=True
):
    """
    Replaces the default column names (e.g 'DLC_resnet50_...'),
    with more useful names, combining rows 0 and 1.

    :param dlc_df: Dataframe loaded from DLC output
    :param remove_header_rows: Remove the two rows used to make the column
    names from the data
    :param reset_index: Reset the dataframe index (after removing the header
    rows)
    :param drop: When resetting the dataframe index, do not try to insert
    index into dataframe columns.

    """
    dlc_df.columns = dlc_df.iloc[0] + "_" + dlc_df.iloc[1]
    if remove_header_rows:
        dlc_df = dlc_df.iloc[2:]
        if reset_index:
            dlc_df = dlc_df.reset_index(drop=drop)

    return dlc_df
