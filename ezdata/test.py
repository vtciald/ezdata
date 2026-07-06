import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon, kruskal, mannwhitneyu, f_oneway, ttest_ind, chi2_contingency, fisher_exact, ttest_1samp, binomtest
from . import prep
from .selector import Selector, ColumnSelector, PairSelector
from collections.abc import Sequence
from itertools import combinations
from statsmodels.stats.contingency_tables import mcnemar, cochrans_q

def test_one_sample(
    df: pd.DataFrame,
    method: str,
    *,
    null: float = 0.0,
    alpha: float = 0.05,
    cols: Sequence[str] | str | ColumnSelector | None = None,
) -> pd.DataFrame:
    """Run a one-sample test.

    Args:
        df (pd.DataFrame): The DataFrame.
        method (str): The test method. Supported choices: 't', 'wilcoxon', 'sign'.
        null (float, optional): The value representing the central tendency of the null hypothesis. Defaults to 0.
        alpha (float, optional): The desired alpha. Defaults to 0.05.
        cols (Sequence[str] | str | ColumnSelector | None, optional): Column(s) to include. If None, includes all columns. Defaults to None.

    Raises:
        ValueError: If string argument for `method` isn't recognized.

    Returns:
        pd.DataFrame: A DataFrame with indices matching the columns specified in the column-selection parameters.
            Columns include:
            - 'test_statistic': A statistic based on the `method` used.
                * T statistic when `method = 't'`.
                * The estimate of the proportion of successes. when `method = 'sign'`.
                * The sum of the ranks of the differences above or below zero, whichever is smaller when `method = 'wilcoxon'.
            - 'p_value': The calculated p value.
            - 'stat_sig': A boolean flag indicating statistical significance.
            - 'count': The number of valid non-nan observations.
    """

    cols = Selector.resolve(df, cols)

    if method == 't':
        result = _one_sample_t(df, cols, null, alpha)

    elif method == 'wilcoxon':
        result = _one_sample_wilcoxon(df, cols, null, alpha)
    
    elif method == 'sign':
        result = _one_sample_sign(df, cols, null, alpha)

    else:
        raise ValueError(f'One-sample test method \'{method}\' is not recognized.')

    return result

def test_one_sample_proportion(
    df: pd.DataFrame,
    method: str,
    *,
    null: float = 0.5,
    alpha: float = 0.05,
    cols: Sequence[str] | str | ColumnSelector | None = None,
) -> pd.DataFrame:
    """Run a one-sample test.

    Args:
        df (pd.DataFrame): The DataFrame.
        method (str): The test method. Supported choices: 't', 'sign'.
        null (float, optional): The value representing the central tendency of the null hypothesis. Defaults to 0.5.
        alpha (float, optional): The desired alpha. Defaults to 0.05.
        cols (Sequence[str] | str | ColumnSelector | None, optional): Column(s) to include. If None, includes all columns. Defaults to None.

    Raises:
        ValueError: If string argument for `method` isn't recognized.

    Returns:
        pd.DataFrame: A DataFrame with indices matching the labels in `cols`.
            Columns include:
            - 'prop_diff': The absolute difference between observed and null proportions.
            - 'p_value': The calculated p value.
            - 'stat_sig': A boolean flag indicating statistical significance.
            - 'count': The number of valid non-nan observations.
    """

    cols = Selector.resolve(df, cols)

    if method == 't':
        result = _one_sample_t(df, cols, null, alpha)
    
    elif method == 'sign':
        result = _one_sample_sign(df, cols, null, alpha, proportion = True)

    else:
        raise ValueError(f'One-sample test method \'{method}\' is not recognized.')

    return result    

def test_independent_proportion(
    df: pd.DataFrame,
    method: str,
    *,
    group_col: Sequence[str] | str | ColumnSelector,
    target_cols: Sequence[str] | str | ColumnSelector | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run an independent-samples test.

    Args:
        df (pd.DataFrame): The DataFrame.
        method (str): The test method. Supported choices: 'chi_squared', 'fisher_exact'.
        group_col (Sequence[str] | str | ColumnSelector): Column(s) to use as the grouping variable. If one-hot encoded, will be converted to mutually exclusive categories.
        target_cols (Sequence[str] | str | ColumnSelector | None, optional): Column(s) to evaluate for differences on the basis of `group_col`. If None, includes all columns. Defaults to None.
        alpha (float, optional): The desired alpha. Defaults to 0.05.

    Raises:
        ValueError: If string argument for `method` isn't recognized.

    Returns:
        pd.DataFrame: A DataFrame with indices matching the labels in `target_cols`.
            Columns include:
            - 'test_statistic': A statistic based on the `method` used.
                * The Chi-squared test statistic when `method = 'chi_squared'`.
                * The prior odds ratio when `method = 'fisher_exact'`.
            - 'p_value': The calculated p value.
            - 'stat_sig': A boolean flag indicating statistical significance.
            - 'count': The number of valid non-nan observations.
    """

    df, group_col = prep.dummy_to_categorical(df, cols = group_col)
    target_cols = Selector.resolve(df, target_cols)

    if method == 'chi_squared':
        result = _independent_chi_sq(df, group_col, target_cols, alpha)
    
    elif method == 'fisher_exact':
        result = _independent_fisher_exact(df, group_col, target_cols, alpha)

    else:
        raise ValueError(f'Independent test method \'{method}\' is not recognized.')

    return result

def test_independent(
    df: pd.DataFrame,
    method: str,
    *,
    group_col: Sequence[str] | str | ColumnSelector,
    target_cols: Sequence[str] | str | ColumnSelector | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run an independent-samples test.

    Args:
        df (pd.DataFrame): The DataFrame.
        method (str): The test method. Supported choices: 't', 'mann_whitney', 'anova', 'kruskal_wallis'
        group_col (Sequence[str] | str | ColumnSelector): Column(s) to use as the grouping variable. If one-hot encoded, will be converted to mutually exclusive categories.
        target_cols (Sequence[str] | str | ColumnSelector | None, optional): Column(s) to evaluate for differences on the basis of `group_col`. If None, includes all columns. Defaults to None.
        alpha (float, optional): The desired alpha. Defaults to 0.05.

    Raises:
        ValueError: If string argument for `method` isn't recognized.

    Returns:
        pd.DataFrame: A DataFrame with indices matching the labels in `target_cols`.
            Columns include:
            - 'test_statistic': A statistic based on the `method` used.
                * T statistic when `method = 't'`.
                * U statistic when `method = 'mann_whitney'`.
                * F statistic when `method = 'anova'`.
                * H statistic when `method = 'kruskal_wallis'`.
            - 'p_value': The calculated p value.
            - 'stat_sig': A boolean flag indicating statistical significance.
            - 'count': The number of valid non-nan observations.
    """

    df, group_col = prep.dummy_to_categorical(df, cols = group_col)
    target_cols = Selector.resolve(df, target_cols)

    if method == 't':
        result = _independent_t(df, group_col, target_cols, alpha)
    
    elif method == 'mann_whitney':
        result = _independent_mann_whitney_u(df, group_col, target_cols, alpha)

    elif method == 'anova':
        result = _independent_one_way_anova(df, group_col, target_cols, alpha)

    elif method == 'kruskal_wallis':
        result = _independent_kruskal_wallis_h(df, group_col, target_cols, alpha)

    else:
        raise ValueError(f'Independent test method \'{method}\' is not recognized.')

    return result

def test_dependent(
    df: pd.DataFrame,
    method: str,
    *,
    target_cols: Sequence[str] | Sequence[Sequence[str]] | ColumnSelector | PairSelector | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run an dependent-samples test.

    Args:
        df (pd.DataFrame): The DataFrame.
        method (str): The test method. Supported choices: 't', 'wilcoxon'
        target_cols (Sequence[str] | Sequence[Sequence[str]] | ColumnSelector | PairSelector | None, optional): Column(s) to evaluate for differences on the basis of `group_col`. If None, includes all columns. Defaults to None.
        alpha (float, optional): The desired alpha. Defaults to 0.05.

    Note:
        If `target_cols` is a list or set of strings (or ColumnSelector), all combinations of columns will be tested.

    Raises:
        ValueError: If string argument for `method` isn't recognized.

    Returns:
        pd.DataFrame: A DataFrame with indices matching the labels in `target_cols`.
            Columns include:
            - 'test_statistic': A statistic based on the `method` used.
                * T statistic when `method = 't'`.
                * The sum of the ranks of the differences above or below zero, whichever is smaller when `method = 'wilcoxon'`.
            - 'p_value': The calculated p value.
            - 'stat_sig': A boolean flag indicating statistical significance.
            - 'count': The number of valid non-nan observations.
    """

    target_cols = Selector.resolve_pair(df, target_cols)

    if method == 't':
        result = _dependent_t(df, target_cols, alpha)
    
    elif method == 'wilcoxon':
        result = _dependent_wilcoxon(df, target_cols, alpha)

    else:
        raise ValueError(f'Independent test method \'{method}\' is not recognized.')

    return result

def test_dependent_proportion(
    df: pd.DataFrame,
    method: str,
    *,
    target_cols: Sequence[str] | Sequence[Sequence[str]] | ColumnSelector | PairSelector | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run an dependent-samples test.

    Args:
        df (pd.DataFrame): The DataFrame.
        method (str): The test method. Supported choices: 'mcnemar_exact', 'mcnemar_asymptotic', 'cochran'.
        target_cols (Sequence[str] | Sequence[Sequence[str]] | ColumnSelector | PairSelector | None, optional): Column(s) to evaluate for differences on the basis of `group_col`. If None, includes all columns. Defaults to None.
        alpha (float, optional): The desired alpha. Defaults to 0.05.

    Note:
        If `target_cols` is a list or set of strings (or ColumnSelector), all combinations of columns will be tested.

    Raises:
        ValueError: If string argument for `method` isn't recognized.

    Returns:
        pd.DataFrame: A DataFrame with indices matching the labels in `target_cols`.
            Columns include:
            - 'test_statistic': A statistic based on the `method` used.
                * T statistic when `method = 't'`.
                * The sum of the ranks of the differences above or below zero, whichever is smaller when `method = 'wilcoxon'`.
            - 'p_value': The calculated p value.
            - 'stat_sig': A boolean flag indicating statistical significance.
            - 'count': The number of valid non-nan observations.
    """

    

    if method == 'mcnemar_exact':
        target_cols = Selector.resolve_pair(df, target_cols)
        result = _dependent_mcnemar(df, target_cols, alpha, exact = True)

    elif method == 'mcnemar_asymptotic':
        target_cols = Selector.resolve_pair(df, target_cols)
        result = _dependent_mcnemar(df, target_cols, alpha, exact = False)
    
    elif method == 'cochran':
        raise NotImplementedError(f'Method \'{method}\' not yet implemented.')
        target_cols = Selector.resolve_pair(df, target_cols) # TODO: groups of cols?? or just list of cols?
        result = _dependent_cochran(df, target_cols, alpha)

    else:
        raise ValueError(f'Independent test method \'{method}\' is not recognized.')

    return result

def _dependent_cochran(
   df: pd.DataFrame,
   column_pairs: list[list[str]],
   alpha: float,
   exact: bool,
) -> pd.DataFrame:
    """Run a Cochran's Q test.

    Args:
        df (pd.DataFrame): The DataFrame.
        column_pairs (list[tuple[str, str]]): Pairs of column labels to compare.
        alpha (float): The desired alpha level.
        exact (bool): If true, the exact binomial distribution will be used. Otherwise, the chi2 approximation will be used.

    Returns:
        pd.DataFrame: A DataFrame with indices matching the labels in `target_cols`.
            Columns include:
            - 'test_statistic': Cochran's Q test statistic.
            - 'p_value': The calculated p value.
            - 'stat_sig': A boolean flag indicating statistical significance.
            - 'count': The number of valid non-nan observations.
    """
    
    index_tuples = []
    counts = []
    test_statistics = []
    p_values = []

    for col0, col1 in column_pairs:
        table = pd.crosstab(df[col0], df[col1])
        count = (df[col0].notna() & df[col1].notna()).sum()
        result = mcnemar(table, exact)

        # group_0, group_1
        index_tuples.append((col0, col1))
        counts.append(count)
        test_statistics.append(result.statistic) # type: ignore
        p_values.append(result.pvalue) # type: ignore

        # group_1, group_0 (so multi-index can be accessed both ways)
        index_tuples.append((col1, col0))      
        counts.append(count)
        test_statistics.append(result.statistic) # type: ignore
        p_values.append(result.pvalue) # type: ignore

    return _create_test_frame(
        index_tuples,
        np.array(test_statistics),
        np.array(p_values),
        np.array(counts), # type: ignore
        alpha,
    )

def _dependent_mcnemar(
   df: pd.DataFrame,
   column_pairs: list[list[str]],
   alpha: float,
   exact: bool,
) -> pd.DataFrame:
    """Run a McNemar test.

    Args:
        df (pd.DataFrame): The DataFrame.
        column_pairs (list[list[str]]): Pairs of column labels to compare.
        alpha (float): The desired alpha level.
        exact (bool): If true, the exact binomial distribution will be used. Otherwise, the chi2 approximation will be used.

    Returns:
        pd.DataFrame: A DataFrame with indices matching the labels in `target_cols`.
            Columns include:
            - 'test_statistic': The Chi-squared test statistic when `exact = False` otherwise the minimum of discordant-pair counts.
            - 'p_value': The calculated p value.
            - 'stat_sig': A boolean flag indicating statistical significance.
            - 'count': The number of valid non-nan observations.
    """
    
    index_tuples = []
    counts = []
    test_statistics = []
    p_values = []

    for col0, col1 in column_pairs:
        table = pd.crosstab(df[col0], df[col1])
        count = (df[col0].notna() & df[col1].notna()).sum()
        result = mcnemar(table, exact)

        # group_0, group_1
        index_tuples.append((col0, col1))
        counts.append(count)
        test_statistics.append(result.statistic) # type: ignore
        p_values.append(result.pvalue) # type: ignore

        # group_1, group_0 (so multi-index can be accessed both ways)
        index_tuples.append((col1, col0))      
        counts.append(count)
        test_statistics.append(result.statistic) # type: ignore
        p_values.append(result.pvalue) # type: ignore

    return _create_test_frame(
        index_tuples,
        np.array(test_statistics),
        np.array(p_values),
        np.array(counts), # type: ignore
        alpha,
    )

def _dependent_t(
   df: pd.DataFrame,
   column_pairs: list[list[str]],
   alpha: float,
) -> pd.DataFrame:
    """Run a dependent-samples T test.

    Args:
        df (pd.DataFrame): The DataFrame.
        column_pairs (list[tuple[str, str]]): Pairs of column labels to compare.
        alpha (float): The desired alpha level.

    Returns:
        pd.DataFrame: A DataFrame with indices matching the labels in `target_cols`.
            Columns include:
            - 'test_statistic': The T statistic.
            - 'p_value': The calculated p value.
            - 'stat_sig': A boolean flag indicating statistical significance.
            - 'count': The number of valid non-nan observations.
    """
    
    index_tuples = []
    counts = []
    test_statistics = []
    p_values = []

    for col0, col1 in column_pairs:
        count = (df[col0].notna() & df[col1].notna()).sum()
        result = ttest_rel(df[col0], df[col1], axis = 0, nan_policy = 'omit')

        # group_0, group_1
        index_tuples.append((col0, col1))
        counts.append(count)
        test_statistics.append(result.statistic) # type: ignore
        p_values.append(result.pvalue) # type: ignore

        # group_1, group_0 (so multi-index can be accessed both ways)
        index_tuples.append((col1, col0))      
        counts.append(count)
        test_statistics.append(np.nan if pd.isna(result.statistic) else -result.statistic) # type: ignore
        p_values.append(result.pvalue) # type: ignore

    return _create_test_frame(
        index_tuples,
        np.array(test_statistics),
        np.array(p_values),
        np.array(counts), # type: ignore
        alpha,
    )

def _dependent_wilcoxon(
   df: pd.DataFrame,
   column_pairs: list[list[str]],
   alpha: float,
) -> pd.DataFrame:
    """Run a dependent-samples Wilcoxon signed-rank test.

    Args:
        df (pd.DataFrame): The DataFrame.
        column_pairs (list[tuple[str, str]]): Pairs of column labels to compare.
        alpha (float): The desired alpha level.

    Returns:
        pd.DataFrame: A DataFrame with indices matching the labels in `target_cols`.
            Columns include:
            - 'test_statistic': The T statistic.
            - 'p_value': The calculated p value.
            - 'stat_sig': A boolean flag indicating statistical significance.
            - 'count': The number of valid non-nan observations.
    """
    
    index_tuples = []
    counts = []
    test_statistics = []
    p_values = []

    for col0, col1 in column_pairs:
        count = (df[col0].notna() & df[col1].notna()).sum()
        result = wilcoxon(
            df[col0],
            df[col1],
            zero_method = 'wilcox',
            method = 'auto',
            nan_policy = 'omit', # type: ignore
        )

        # group_0, group_1
        index_tuples.append((col0, col1))
        counts.append(count)
        test_statistics.append(result.statistic) # type: ignore
        p_values.append(result.pvalue) # type: ignore

        # group_1, group_0 (so multi-index can be accessed both ways)
        index_tuples.append((col1, col0))      
        counts.append(count)
        test_statistics.append(result.statistic) # type: ignore
        p_values.append(result.pvalue) # type: ignore

    return _create_test_frame(
        index_tuples,
        np.array(test_statistics),
        np.array(p_values),
        np.array(counts), # type: ignore
        alpha,
    )

def _independent_kruskal_wallis_h(
   df: pd.DataFrame,
   group_col: str,
   target_cols: list[str],
   alpha: float 
) -> pd.DataFrame:
    """Run a Kruskal-Wallis H test.

    Args:
        df (pd.DataFrame): The DataFrame.
        group_col (str): The grouping column label.
        target_cols (list[str]): The labels of columns to test independence with `group_col`.
        alpha (float): The desired alpha level.

    Returns:
        pd.DataFrame: A DataFrame with indices matching the labels in `target_cols`.
            Columns include:
            - 'test_statistic': The H statistic.
            - 'p_value': The calculated p value.
            - 'stat_sig': A boolean flag indicating statistical significance.
            - 'count': The number of valid non-nan observations.
    """
    
    counts = df.loc[df[group_col].notna(), target_cols].agg('count', axis = 0).values
    group_data = []

    for group in df[group_col].unique():
        if group == np.nan or pd.isna(group): continue
        group_filter = df.loc[df[group_col] == group, target_cols]
        group_data.append(group_filter.values)

    result = kruskal(*group_data, nan_policy = 'omit') # type: ignore

    return _create_test_frame(
        target_cols,
        np.array(result.statistic),
        np.array(result.pvalue),
        np.array(counts), # type: ignore
        alpha,
    )

def _independent_mann_whitney_u(
   df: pd.DataFrame,
   group_col: str,
   target_cols: list[str],
   alpha: float 
) -> pd.DataFrame:
    """Run a Mann-Whitney U test.

    Args:
        df (pd.DataFrame): The DataFrame.
        group_col (str): The grouping column label.
        target_cols (list[str]): The labels of columns to test independence with `group_col`.
        alpha (float): The desired alpha level.

    Returns:
        pd.DataFrame: A DataFrame with multi-index indices, (target_col, group_0, group_1).
            Columns include:
            - 'test_statistic': The U statistic.
            - 'p_value': The calculated p value.
            - 'stat_sig': A boolean flag indicating statistical significance.
            - 'count': The number of valid non-nan observations.
    """
    
    unique_groups = [group for group in df[group_col].unique() if pd.notna(group)]
    pairs = list(combinations(unique_groups, 2))

    index_tuples = []
    test_statistics = []
    p_values = []
    counts = []

    for target_col in target_cols:
        for group0, group1 in pairs:
            group0_filter = df.loc[df[group_col] == group0, target_col].dropna() # type: ignore
            group1_filter = df.loc[df[group_col] == group1, target_col].dropna() # type: ignore
            count = len(group0_filter) + len(group1_filter)

            result = mannwhitneyu(
                group0_filter, 
                group1_filter, 
                method = 'auto', 
                nan_policy = 'omit' # type: ignore
            ) 
            
            # group_0, group_1
            index_tuples.append((target_col, group0, group1))
            counts.append(count)
            test_statistics.append(result.statistic) # type: ignore
            p_values.append(result.pvalue) # type: ignore

            # group_1, group_0 (so multi-index can be accessed both ways)
            index_tuples.append((target_col, group1, group0))
            counts.append(count)  
            u_reversed = (len(group0_filter) * len(group1_filter)) - result.statistic          
            test_statistics.append(u_reversed) # type: ignore
            p_values.append(result.pvalue) # type: ignore

    return _create_test_frame(
        index_tuples,
        np.array(test_statistics),
        np.array(p_values),
        np.array(counts),
        alpha,
    )

def _independent_one_way_anova(
   df: pd.DataFrame,
   group_col: str,
   target_cols: list[str],
   alpha: float 
) -> pd.DataFrame:
    """Run a one-way ANOVA.

    Args:
        df (pd.DataFrame): The DataFrame.
        group_col (str): The grouping column label.
        target_cols (list[str]): The labels of columns to test independence with `group_col`.
        alpha (float): The desired alpha level.

    Returns:
        pd.DataFrame: A DataFrame with indices matching the labels in `target_cols`.
            Columns include:
            - 'test_statistic': The F statistic.
            - 'p_value': The calculated p value.
            - 'stat_sig': A boolean flag indicating statistical significance.
            - 'count': The number of valid non-nan observations.
    """
    
    counts = df.loc[df[group_col].notna(), target_cols].agg('count', axis = 0).values
    group_data = []

    for group in df[group_col].unique():
        if group == np.nan or pd.isna(group): continue
        group_filter = df.loc[df[group_col] == group, target_cols]
        group_data.append(group_filter.values)

    result = f_oneway(*group_data, nan_policy = 'omit') # type: ignore

    return _create_test_frame(
        target_cols,
        np.array(result.statistic),
        np.array(result.pvalue),
        np.array(counts),
        alpha,
    )

def _independent_t(
   df: pd.DataFrame,
   group_col: str,
   target_cols: list[str],
   alpha: float 
) -> pd.DataFrame:
    """Run an independent-samples t test.

    Args:
        df (pd.DataFrame): The DataFrame.
        group_col (str): The grouping column label.
        target_cols (list[str]): The labels of columns to test independence with `group_col`.
        alpha (float): The desired alpha level.

    Returns:
        pd.DataFrame: A DataFrame with multi-index indices, (target_col, group_0, group_1).
            Columns include:
            - 'test_statistic': The T statistic.
            - 'p_value': The calculated p value.
            - 'stat_sig': A boolean flag indicating statistical significance.
            - 'count': The number of valid non-nan observations.
    """

    unique_groups = [group for group in df[group_col].unique() if pd.notna(group)]
    pairs = list(combinations(unique_groups, 2))

    index_tuples = []
    test_statistics = []
    p_values = []
    counts = []

    for target_col in target_cols:
        for group0, group1 in pairs:
            group0_filter = df.loc[df[group_col] == group0, target_col].dropna() # type: ignore
            group1_filter = df.loc[df[group_col] == group1, target_col].dropna() # type: ignore
            count = len(group0_filter) + len(group1_filter)

            result = ttest_ind(group0_filter, group1_filter, nan_policy = 'omit')

            # group_0, group_1
            index_tuples.append((target_col, group0, group1))
            counts.append(count)
            test_statistics.append(result.statistic) # type: ignore
            p_values.append(result.pvalue) # type: ignore

            # group_1, group_0 (so multi-index can be accessed both ways)
            index_tuples.append((target_col, group1, group0))
            counts.append(count)            
            test_statistics.append(np.nan if pd.isna(result.statistic) else -result.statistic) # type: ignore
            p_values.append(result.pvalue) # type: ignore

    return _create_test_frame(
        index_tuples,
        np.array(test_statistics),
        np.array(p_values),
        np.array(counts),
        alpha,
    )

def _independent_chi_sq(
    df: pd.DataFrame,
    group_col: str,
    target_cols: list[str],
    alpha: float
) -> pd.DataFrame:
    """Run a Chi-squared test of independence.

    Runs a separate test between each pair of `group_col` and one of the `target_cols`.

    Args:
        df (pd.DataFrame): The DataFrame.
        group_col (str): The grouping column label.
        target_cols (list[str]): The labels of columns to test independence with `group_col`.
        alpha (float): The desired alpha level.

    Returns:
        pd.DataFrame: A DataFrame with indices matching the labels in `target_cols`.
            Columns include:
            - 'test_statistic': The Chi-squared test statistic.
            - 'p_value': The calculated p value.
            - 'stat_sig': A boolean flag indicating statistical significance.
            - 'count': The number of valid non-nan observations.
    """

    test_statistics = []
    p_values = []
    counts = []

    for target_col in target_cols:
        count = (df[group_col].notna() & df[target_col].notna()).sum()
        contingency = pd.crosstab(df[group_col].values, df[target_col].values)
        result = chi2_contingency(contingency.values)

        test_statistics.append(result.statistic) # type: ignore
        p_values.append(result.pvalue) # type: ignore
        counts.append(count)
    
    return _create_test_frame(
        target_cols,
        np.array(test_statistics),
        np.array(p_values),
        np.array(counts),
        alpha,
    )

def _independent_fisher_exact(
    df: pd.DataFrame,
    group_col: str,
    target_cols: list[str],
    alpha: float
) -> pd.DataFrame:
    """Run Fisher's Exact Test of independence.

    Runs a separate test between each pair of `group_col` and one of the `target_cols`. Requires that each column have 2 unique values (ignoring NaN).

    Args:
        df (pd.DataFrame): The DataFrame.
        group_col (str): The grouping column label.
        target_cols (list[str]): The labels of columns to test independence with `group_col`.
        alpha (float): The desired alpha level.

    Returns:
        pd.DataFrame: A DataFrame with indices matching the labels in `target_cols`.
            Columns include:
            - 'test_statistic': The prior odds ratio.
            - 'p_value': The calculated p value.
            - 'stat_sig': A boolean flag indicating statistical significance.
            - 'count': The number of valid non-nan observations.
    """

    test_statistics = []
    p_values = []
    counts = []

    for target_col in target_cols:
        count = (df[group_col].notna() & df[target_col].notna()).sum()
        contingency = pd.crosstab(df[group_col].values, df[target_col].values)

        if contingency.shape != (2, 2):
                raise ValueError(
                    f'Fisher\'s Exact Test requires a (2, 2) table. '
                    f'{group_col} vs {target_col} produced a {contingency.shape} table.'
                )

        result = fisher_exact(contingency.values)

        test_statistics.append(result.statistic) # type: ignore
        p_values.append(result.pvalue) # type: ignore
        counts.append(count)
    
    return _create_test_frame(
        target_cols,
        np.array(test_statistics),
        np.array(p_values),
        np.array(counts),
        alpha,
    )

def _one_sample_t(
    df: pd.DataFrame,
    cols: list[str],
    null: float,
    alpha: float,
) -> pd.DataFrame:
    """Run a one-sample t test.

    Args:
        df (pd.DataFrame): The DataFrame.
        cols (list[str]): The columns on which to operate.
        null (float, optional): The population mean under the null hypothesis. Defaults to 0.
        alpha (float): The desired alpha level.

    Returns:
        pd.DataFrame: A DataFrame with indices matching the labels in `cols`.
            Columns include:
            - 'test_statistic': The t statistic.
            - 'p_value': The calculated p value.
            - 'stat_sig': A boolean flag indicating statistical significance.
            - 'count': The number of valid non-nan observations.
    """

    counts = df[cols].agg('count', axis = 0)

    result = ttest_1samp(
        df[cols].to_numpy(),
        popmean = null,
        alternative = 'two-sided',
        nan_policy = 'omit'
    )
    
    return _create_test_frame(
        cols,
        np.array(result.statistic), # type: ignore
        np.array(result.pvalue), # type: ignore
        np.array(counts),
        alpha,
    )

def _one_sample_sign(
    df: pd.DataFrame,
    cols: list[str],
    null: float,
    alpha: float,
    proportion: bool = False,
) -> pd.DataFrame:
    """Run a one-sample sign test.

    Args:
        df (pd.DataFrame): The DataFrame.
        cols (list[str]): The columns on which to operate.
        null (float, optional): The population median under the null hypothesis. Defaults to 0.
        alpha (float): The desired alpha level.
        proportion (bool): Whether the test is on a proportion or continuous. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame with indices matching the labels in `cols`.
            Columns include:
            - 'test_statistic': The estimate of the proportion of successes.
            - 'p_value': The calculated p value.
            - 'stat_sig': A boolean flag indicating statistical significance.
            - 'count': The number of valid non-nan observations.
    """

    test_statistics = []
    p_values = []
    counts = []

    for col in cols:
        data = df[col].dropna()
        diffs = data - null
        positives = np.sum(diffs > 0)
        total_trials = len(data) if proportion else np.sum(diffs != 0)

        if total_trials == 0:
            test_statistics.append(np.nan)
            p_values.append(np.nan)
            counts.append(len(data))
        
        else:
            result = binomtest(
                positives,
                total_trials,
                p = null if proportion else 0.5
            )

            test_statistics.append(result.statistic)
            p_values.append(result.pvalue)
            counts.append(len(data))
    
    return _create_test_frame(
        cols,
        np.array(test_statistics),
        np.array(p_values),
        np.array(counts),
        alpha,
    )

def _one_sample_wilcoxon(
    df: pd.DataFrame,
    cols: list[str],
    null: float,
    alpha: float,
) -> pd.DataFrame:
    """Run a one-sample Wilcoxon signed-rank test.

    Args:
        df (pd.DataFrame): The DataFrame.
        cols (list[str]): The columns on which to operate.
        null (float, optional): The population median under the null hypothesis. Defaults to 0.
        alpha (float): The desired alpha level.

    Returns:
        pd.DataFrame: A DataFrame with indices matching the labels in `cols`.
            Columns include:
            - 'test_statistic': The sum of the ranks of the differences above or below zero, whichever is smaller.
            - 'p_value': The calculated p value.
            - 'stat_sig': A boolean flag indicating statistical significance.
            - 'count': The number of valid non-nan observations.
    """

    counts = df[cols].agg('count', axis = 0)

    result = wilcoxon(
        df[cols].to_numpy() - null,
        alternative = 'two-sided',
        zero_method = 'wilcox',
        method = 'auto',
        nan_policy = 'omit', # type: ignore
    )
    
    return _create_test_frame(
        cols,
        np.array(result.statistic),
        np.array(result.pvalue),
        np.array(counts),
        alpha,
    )

def _create_test_frame(
    indices: list[str] | list[tuple],
    test_statistics: np.ndarray,
    p_values: np.ndarray,
    counts: np.ndarray,
    alpha: float,
) -> pd.DataFrame:
    """Package test results into a DataFrame.

    Args:
        indices (list[str] | tuple): The labels associated with each column. If a list of tuples, will create a multi-index frame (target_col, group_0, group_1).
        test_statistics (np.ndarray): The array of test statistics.
        p_values (np.ndarray): The array of p values.
        counts (np.ndarray): The array of column non-nan counts.
        alpha (float): The desired alpha level.
        statistic_name (str): The name of the kind of values in `test_statistics`.

    Returns:
        pd.DataFrame: A DataFrame with indices matching the labels in `cols` and columns `statistic_name`, 'p_value', 'stat_sig', 'count'.
    """
    
    data_dict = {
        'test_statistic': test_statistics.astype(float),
        'p_value': p_values.astype(float),
        'stat_sig': (p_values < alpha).astype(bool),
        'count': counts.astype(int),
    }

    if isinstance(indices[0], tuple):
        names = ['group_0', 'group_1']

        if len(indices[0]) > 2:
            names = ['target_col', 'group_0', 'group_1']
            
        multi_index = pd.MultiIndex.from_tuples(
            indices, # type: ignore
            names = names,
        )

        frame_index = multi_index

    else:
        frame_index = indices

    result = pd.DataFrame(
        data_dict,
        index = frame_index
    )

    return result  

# TODO: Differentiate a pair match vs pair split Selector / method??

# TODO: Add other test methods...
# test_dependent_proportion(): mcnemar asymptotic, mcnemar exact binomial, cochran's Q
# test_regression(): linear, logistic
# Add 'bootstrap' method to tests

# TODO: Update "Notes:" in docstrings for each test to add reminders of when/why to use each (e.g., exact mcnemars for < 25, etc.)

# TODO: Update column selection resolution to ensure the default (when target_cols = None) doesn't include the group_col

# TODO: Add p-value correction methods...bonferroni, holm-bonferroni, benjamini-hochberg

# TODO: Consider adding test of normality (and maybe leverage alongside sample size when method is unspecified in higher-level funcs?)