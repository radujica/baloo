from pandas import read_csv as pd_read_csv

from ..core import DataFrame


def read_csv(filepath, sep=',', header='infer', names=None, usecols=None, dtype=None, converters=None,
             skiprows=None, nrows=None):
    """Read CSV into DataFrame.

    Eager implementation using pandas, i.e. entire file is read at this point. Only common/relevant parameters
    available at the moment; for full list, could use pandas directly and then convert to baloo.

    Parameters
    ----------
    filepath : str
    sep : str, optional
        Separator used between values.
    header : 'infer' or None, optional
        Whether to infer the column names from the first row or not.
    names : list of str, optional
        List of column names to use. Overrides inferred header.
    usecols : list of (int or str), optional
        Which columns to parse.
    dtype : dict, optional
        Dict of column -> type to parse as.
    converters : dict, optional
        Dict of functions for converting values in certain columns.
    skiprows : int, optional
        Number of lines to skip at start of file.
    nrows : int, optional
        Number of rows to read.

    Returns
    -------
    DataFrame

    See Also
    --------
    pandas.read_csv : https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html

    """
    pd_df = pd_read_csv(filepath,
                        sep=sep,
                        header=header,
                        names=names,
                        usecols=usecols,
                        dtype=dtype,
                        converters=converters,
                        skiprows=skiprows,
                        nrows=nrows)

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

    See Also
    --------
    pandas.DataFrame.to_csv : https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_csv.html

    """
    df.to_pandas().to_csv(filepath,
                          sep=sep,
                          header=header,
                          index=index)
