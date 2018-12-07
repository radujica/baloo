from pandas import read_csv as pd_read_csv

from ..core import DataFrame


# TODO: make this lazy ~ need small framework
def read_csv(filepath, sep=',', header='infer', names=None):
    """Read CSV into DataFrame.

    Eager implementation using pandas, i.e. entire file is read at this point.

    Parameters
    ----------
    filepath : str
    sep : str, optional
        Separator used between values.
    header : 'infer' or None, optional
        Whether to infer the column names from the first row or not.
    names : list of str, optional
        List of column names to use. Overrides inferred header.

    Returns
    -------
    DataFrame

    """
    pd_df = pd_read_csv(filepath,
                        sep=sep,
                        header=header,
                        names=names)

    return DataFrame.from_pandas(pd_df)


# TODO: should avoid going to Pandas
def to_csv(df, filepath, sep=',', header=True, index=True):
    """Save DataFrame as csv.

    Note data is expected to be evaluated.

    Currently delegates to Pandas.

    Parameters
    ----------
    df : DataFrame
    filepath : str
    sep : str, optional
        Separator used between values.
    header : bool, optional
        Whether to save the header.
    index : bool, optional
        Whether to save the index columns.

    Returns
    -------
    None

    """
    df.to_pandas().to_csv(filepath,
                          sep=sep,
                          header=header,
                          index=index)
