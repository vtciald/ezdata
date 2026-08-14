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
    random_state: int | None = None,
    print_summary: bool = False,
) -> pd.DataFrame:
    """Add PCA components to DataFrame.

    Used if all variables are numeric.

    Args:
        df (pd.DataFrame): The DataFrame
        cols (Sequence[str] | ColumnSelector | None): Column(s) to include. If None, includes all columns. Defaults to None.
        n_components (int): The number of components that are computed.
        n_iter (int, optional): The number of iterations used for computing the SVD. Defaults to 4.
        rescale_with_mean (bool, optional): If true, subtracts each column's mean from their values. Defaults to True.
        rescale_with_std (bool, optional): If true, divides each column by its standard deviation. Defaults to True.
        random_state (int | None, optional): A random-number-generator seed. Defaults to None.
        print_summary (bool, optional): If true, prints the eigenvalue summary after fit. Defaults to False.

    Returns:
        pd.DataFrame: The DataFrame with `n_components` additional columns of the form 'pca_component_{n}'.
    """

    cols = Selector.resolve(df, cols)
    X = _prep_reduce_X(df, cols, 'PCA')

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

def reduce_mca(
    df: pd.DataFrame,
    *,
    n_components: int,
    cols: Sequence[str] | ColumnSelector | None = None,
    n_iter: int = 4,
    random_state: int | None = None,
    print_summary: bool = False,
) -> pd.DataFrame:
    """Add MCA components to DataFrame.

    Used if all variables are categorical.

    Args:
        df (pd.DataFrame): The DataFrame
        cols (Sequence[str] | ColumnSelector | None): Column(s) to include. If None, includes all columns. Defaults to None.
        n_components (int): The number of components that are computed.
        n_iter (int, optional): The number of iterations used for computing the SVD. Defaults to 4.
        random_state (int | None, optional): A random-number-generator seed. Defaults to None.
        print_summary (bool, optional): If true, prints the eigenvalue summary after fit. Defaults to False.

    Returns:
        pd.DataFrame: The DataFrame with `n_components` additional columns of the form 'pca_component_{n}'.
    """

    cols = Selector.resolve(df, cols)
    X = _prep_reduce_X(df, cols, 'MCA')

    mca = prince.MCA(
        n_components = n_components,
        n_iter = n_iter,
        check_input = True,
        correction = 'greenacre',
        random_state = random_state
    )

    mca = mca.fit(X)

    col_names = [f'mca_component_{n}' for n in range(0, n_components)]
    df[col_names] = mca.row_coordinates(X)

    if print_summary:
        print(mca.eigenvalues_summary)

    return df

def _prep_reduce_X(
    df: pd.DataFrame,
    cols: list[str],
    method: str,
) -> pd.DataFrame:
    """Prepare inputs for dimension reduction.

    Args:
        df (pd.DataFrame): The DataFrame.
        cols (list[str]): The columns on which to operate.
        method (str): The method of dimension reduction.

    Returns:
        pd.DataFrame: The DataFrame input for dimension reduction.
    """
    
    if method not in {'PCA', 'MCA', 'MFA', 'FAMD'}:
        raise ValueError(f'Dimension reduction method \'{method}\' is not recognized.')

    df = df.copy()
    X = df[cols]

    if df[cols].isna().sum(axis = 0).sum() > 0:
        warnings.warn(
            f'There are NaNs present in columns intended for {method}. '
            f'Offending rows will be NaN for {method} components.'
        )

        X = df[cols].dropna(axis = 0, how = 'any')

    if method == 'PCA':
        X = _prep_numeric_X(X)

    elif method == 'MCA':
        X = _prep_one_hot_X(X, method)

    return X

def _prep_one_hot_X(
    X: pd.DataFrame,
    method: str,
    categorical_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Ensure categorical columns are one-hot encoded.

    Args:
        X (pd.DataFrame): The DataFrame.
        method (str): The method of dimension reduction.
        categorical_cols (list[str] | None, optional): Columns on which to operate. If None, includes all columns. Defaults to None.

    Returns:
        pd.DataFrame: The DataFrame with categorical columns one-hot encoded.
    """

    cols = categorical_cols if categorical_cols is not None else X.columns
    to_dummy_cols = []
    retain_cols = []

    for col in cols:
        if not X[col].isin([0, 1, '0', '1', False, True]).all():
            if pd.api.types.is_integer_dtype(X[col]):
                warnings.warn(
                    f'Column \'{col}\' was included for {method} as a '
                    'categorical column, but it has an integer dtype.'
                )

            elif pd.api.types.is_float_dtype(X[col]):
                warnings.warn(
                    f'Column \'{col}\' was included for {method} as a '
                    'categorical column, but it has a float dtype.'
                )
                
            to_dummy_cols.append(col)

        else:
            retain_cols.append(col)

    X = X[retain_cols].join(pd.get_dummies(X[to_dummy_cols]))

    X = X.astype(int)

    return X

def _prep_numeric_X(
    X: pd.DataFrame,
    numeric_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Ensure numeric columns are floats.

    Args:
        X (pd.DataFrame): The DataFrame.
        numeric_cols (list[str] | None, optional): Columns on which to operate. If None, includes all columns. Defaults to None.

    Returns:
        pd.DataFrame: The DataFrame with numeric columns as floats.
    """

    cols = numeric_cols if numeric_cols is not None else X.columns

    for col in cols:
        X[col] = X[col].astype(float)

    return X

# TODO: be sure to check how categorical cols are handled in MFA or FAMD... if so, may not want to one-hot encode?

# MFA -> Used if there are groups of categorical xor numeric variables.
# FAMD -> Used if there are both categorical and numeric variables.