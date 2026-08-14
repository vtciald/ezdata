import pandas as pd
import numpy as np
import prince
from .selector import Selector, ColumnSelector, PairSelector, GroupSelector
from collections.abc import Sequence
import warnings

def reduce_pca(
    df: pd.DataFrame,
    *,
    n_components: int,
    cols: Sequence[str] | ColumnSelector | None = None,
    n_iter: int = 4,
    rescale_with_mean: bool = True,
    rescale_with_std: bool = True,
    random_state: int = 0,
    print_summary: bool = False,
) -> pd.DataFrame:
    """Add PCA components to DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame
        cols (Sequence[str] | ColumnSelector | None): Column(s) to include. If None, includes all columns. Defaults to None.
        n_components (int): The number of components that are computed.
        n_iter (int, optional): The number of iterations used for computing the SVD. Defaults to 4.
        rescale_with_mean (bool, optional): If true, subtracts each column's mean from their values. Defaults to True.
        rescale_with_std (bool, optional): If true, divides each column by its standard deviation. Defaults to True.
        random_state (int, optional): A random-number-generator seed. Defaults to 0.
        print_summary (bool, optional): If true, prints the eigenvalue summary after fit. Defaults to False.

    Returns:
        pd.DataFrame: The DataFrame with `n_components` additional columns of the form 'pca_component_{n}'.
    """

    cols = Selector.resolve(df, cols)
    df = df.copy()
    X = df[cols]

    if df[cols].isna().sum(axis = 0).sum() > 0:
        warnings.warn(
            'There are NaNs present in columns intended for PCA. '
            'Offending rows will be NaN for PCA components.'
        )

        X = df[cols].dropna(axis = 0, how = 'any')

    pca = prince.PCA(
        n_components = n_components,
        n_iter = n_iter,
        rescale_with_mean = rescale_with_mean,
        rescale_with_std = rescale_with_std,
        check_input = True,
        random_state = random_state
    )

    pca = pca.fit(X)

    col_names = [f'pca_component_{n}' for n in range(0, n_components)]
    df[col_names] = pca.row_coordinates(X)

    if print_summary:
        print(pca.eigenvalues_summary)

    return df

# TODO: update random_state.... shouldn't default to 0 but a None-like