import pandas as pd


def columns_to_categorical_codes(
    df: pd.DataFrame, cols: list[str]
) -> pd.DataFrame:
    """Convert columns in a DataFrame to ordered categorical codes. The codes
    are integers corresponding to the unique values in the column,
    ordered by appearance."""
    new_df = df.copy()
    for col in cols:
        cat_dtype = pd.api.types.CategoricalDtype(
            categories=df[col].unique().tolist(), ordered=True
        )
        new_df[col] = df[col].astype(cat_dtype).cat.codes
    return new_df
