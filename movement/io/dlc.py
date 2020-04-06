import pandas as pd
from pathlib import Path


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
