import numpy as np
import pandas as pd
import scipy.stats
import re
from . import prep
from .selector import Selector

def test_one_sample(
    df: pd.DataFrame,
    method: str,
    *,
    null: float = 0.0,
    alpha: float = 0.05,
    cols: list[str] | set[str] | str | Selector | None = None,
) -> pd.DataFrame:
    """Run a one-sample test.

    Args:
        df (pd.DataFrame): The DataFrame.
        method (str): The test method. Supported choices: 't', 'wilcoxon', 'sign', 'bootstrap'.
        null (float, optional): The value representing the central tendency of the null hypothesis. Defaults to 0.
        alpha (float, optional): The desired alpha. Defaults to 0.05.
        cols (list[str] | set[str] | str | Selector | None, optional): Column(s) to include. If None, includes all columns. Defaults to None.

    Raises:
        ValueError: If string argument for `method` isn't recognized.

    Returns:
        pd.DataFrame: A DataFrame with indices matching the columns specified in the column-selection parameters.
            Columns include:
            - A descriptive difference column, dynamically named based on the test.
                * 'mean_diff' when `method = 't'`.
                * 'median_diff' when `method = 'wilcoxon' or `method = 'sign' and proportion = False`
                * 'prop_diff' when `method = 'sign' and proportion = True`
            - 'p_value': The calculated p value.
            - 'stat_sig': A boolean flag indicating statistical significance.
            - 'count': The number of valid non-nan observations.
    """

    cols = Selector.resolve_selection(df, cols)

    if method == 't':
        result = _one_sample_t(df, cols, null, alpha)

    elif method == 'wilcoxon':
        result = _one_sample_wilcoxon(df, cols, null, alpha)
    
    elif method == 'sign':
        result = _one_sample_sign(df, cols, null, alpha)
    
    # elif method == 'bootstrap':
    #     raise NotImplementedError(f'Method \'{method}\' is not yet implemented.')

    else:
        raise ValueError(f'One-sample test method \'{method}\' is not recognized.')

    return result

def test_one_sample_proportion(
    df: pd.DataFrame,
    method: str,
    *,
    null: float = 0.5,
    alpha: float = 0.05,
    cols: list[str] | set[str] | str | Selector | None = None,
) -> pd.DataFrame:
    """Run a one-sample test.

    Args:
        df (pd.DataFrame): The DataFrame.
        method (str): The test method. Supported choices: 't', 'sign'.
        null (float, optional): The value representing the central tendency of the null hypothesis. Defaults to 0.5.
        alpha (float, optional): The desired alpha. Defaults to 0.05.
        cols (list[str] | set[str] | str | Selector | None, optional): Column(s) to include. If None, includes all columns. Defaults to None.

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

    cols = Selector.resolve_selection(df, cols)

    if method == 't':
        result = _one_sample_t(df, cols, null, alpha)
    
    elif method == 'sign':
        result = _one_sample_sign(df, cols, null, alpha, proportion = True)
    
    # elif method == 'bootstrap':
    #     raise NotImplementedError(f'Method \'{method}\' is not yet implemented.')

    else:
        raise ValueError(f'One-sample test method \'{method}\' is not recognized.')

    return result    

def test_independent_proportion(
    df: pd.DataFrame,
    method: str,
    *,
    group_col: list[str] | set[str] | str | Selector,
    target_cols: list[str] | set[str] | str | Selector | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run an independent-samples test.

    Args:
        df (pd.DataFrame): The DataFrame.
        method (str): The test method. Supported choices: 'chi_squared', 'fisher_exact'.
        group_col (list[str] | set[str] | str | Selector): Column(s) to use as the grouping variable. If one-hot encoded, will be converted to mutually exclusive categories.
        target_cols (list[str] | set[str] | str | Selector | None, optional): Column(s) to evaluate for differences on the basis of `group_col`. If None, includes all columns. Defaults to None.
        alpha (float, optional): The desired alpha. Defaults to 0.05.

    Raises:
        ValueError: If string argument for `method` isn't recognized.

    Returns:
        pd.DataFrame: A DataFrame with indices matching the labels in `target_cols`.
            Columns include:
            - A test-statistic column, dynamically named based on the test.
                * 'test_statistic': The Chi-squared test statistic when `method = 'chi_squared'`.
                * 'odds_ratio': The prior odds ratio when `method = 'fisher_exact'`.
            - 'p_value': The calculated p value.
            - 'stat_sig': A boolean flag indicating statistical significance.
            - 'count': The number of valid non-nan observations.
    """

    df, group_col = prep.dummy_to_categorical(df, cols = group_col)
    target_cols = Selector.resolve_selection(df, target_cols)

    if method == 'chi_squared':
        result = _chi_sq_independence(df, group_col, target_cols, alpha)
    
    elif method == 'fisher_exact':
        result = _fisher_exact_independence(df, group_col, target_cols, alpha)

    # elif method == 'bootstrap':
    #     raise NotImplementedError(f'Method \'{method}\' is not yet implemented.')

    else:
        raise ValueError(f'Independent test method \'{method}\' is not recognized.')

    return result

def test_independent(
    df: pd.DataFrame,
    method: str,
    *,
    group_col: list[str] | set[str] | str | Selector,
    target_cols: list[str] | set[str] | str | Selector | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run an independent-samples test.

    Args:
        df (pd.DataFrame): The DataFrame.
        method (str): The test method. Supported choices: 't', 'mann_whitney', 'anova', 'kruskal_wallis'
        group_col (list[str] | set[str] | str | Selector): Column(s) to use as the grouping variable. If one-hot encoded, will be converted to mutually exclusive categories.
        target_cols (list[str] | set[str] | str | Selector | None, optional): Column(s) to evaluate for differences on the basis of `group_col`. If None, includes all columns. Defaults to None.
        alpha (float, optional): The desired alpha. Defaults to 0.05.

    Raises:
        ValueError: If string argument for `method` isn't recognized.

    Returns:
        pd.DataFrame: A DataFrame with indices matching the labels in `target_cols`.
            Columns include:
            - A test-statistic column, dynamically named based on the test.
                * '___': ___ when `method = 't'`.
                * '___': ___ when `method = 'mann_whitney'`.
                * 'test_statistic': The F statistic when `method = 'anova'`.
                * '___': ___ when `method = 'kruskal_wallis'`.
            - 'p_value': The calculated p value.
            - 'stat_sig': A boolean flag indicating statistical significance.
            - 'count': The number of valid non-nan observations.
    """

    df, group_col = prep.dummy_to_categorical(df, cols = group_col)
    target_cols = Selector.resolve_selection(df, target_cols)

    if method == 't':
        raise NotImplementedError(f'Method \'{method}\' is not yet implemented.')
        result = _t_independent(df, group_col, target_cols, alpha)
    
    elif method == 'mann_whitney':
        raise NotImplementedError(f'Method \'{method}\' is not yet implemented.')
        result = _mann_whitney_u_independent(df, group_col, target_cols, alpha)

    elif method == 'anova':
        result = _one_way_anova_independent(df, group_col, target_cols, alpha)

    elif method == 'kruskal_wallis':
        raise NotImplementedError(f'Method \'{method}\' is not yet implemented.')
        result = _kruskal_wallis_independent(df, group_col, target_cols, alpha)

    # elif method == 'bootstrap':
    #     raise NotImplementedError(f'Method \'{method}\' is not yet implemented.')

    else:
        raise ValueError(f'Independent test method \'{method}\' is not recognized.')

    return result

def _one_way_anova_independent(
   df: pd.DataFrame,
   group_col: str,
   target_cols: list[str],
   alpha: float 
) -> pd.DataFrame:
    
    counts = df.loc[df[group_col].notna(), target_cols].agg('count', axis = 0).values
    group_data = []

    for group in df[group_col].unique():
        if group == np.nan or pd.isna(group): continue
        group_filter = df.loc[df[group_col] == group, target_cols]
        group_data.append(group_filter.values)

    result = scipy.stats.f_oneway(*group_data, nan_policy = 'omit') # type: ignore

    return _create_test_frame(
        target_cols,
        result.statistic,
        result.pvalue,
        np.array(counts),
        alpha,
        'test_statistic'
    )

def _chi_sq_independence(
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
        result = scipy.stats.chi2_contingency(contingency.values)

        test_statistics.append(result.statistic) # type: ignore
        p_values.append(result.pvalue) # type: ignore
        counts.append(count)
    
    return _create_test_frame(
        target_cols,
        np.array(test_statistics),
        np.array(p_values),
        np.array(counts),
        alpha,
        'test_statistic'
    )

def _fisher_exact_independence(
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
            - 'odds_ratio': The prior odds ratio.
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

        result = scipy.stats.fisher_exact(contingency.values)

        test_statistics.append(result.statistic) # type: ignore
        p_values.append(result.pvalue) # type: ignore
        counts.append(count)
    
    return _create_test_frame(
        target_cols,
        np.array(test_statistics),
        np.array(p_values),
        np.array(counts),
        alpha,
        'odds_ratio'
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
            - 'mean_diff': The difference between observed and null means.
            - 'p_value': The calculated p value.
            - 'stat_sig': A boolean flag indicating statistical significance.
            - 'count': The number of valid non-nan observations.
    """

    desc = df[cols].agg(['count', 'mean'], axis = 0)

    result = scipy.stats.ttest_1samp(
        df[cols].to_numpy(),
        popmean = null,
        alternative = 'two-sided',
        nan_policy = 'omit'
    )
    
    return _create_test_frame(
        cols,
        desc.loc['mean'].to_numpy() - null,
        result.pvalue, # type: ignore
        desc.loc['count'].to_numpy(),
        alpha,
        'mean_diff',
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
        proportion (bool): Whether the test is on a proportion or continuous 

    Returns:
        pd.DataFrame: A DataFrame with indices matching the labels in `cols`.
            Columns include:
            - A descriptive difference column, dynamically named based on the test.
                * 'median_diff' when `proportion = False`
                * 'prop_diff' when `proportion = True`
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
            result = scipy.stats.binomtest(
                positives,
                total_trials,
                p = null if proportion else 0.5
            )

            test_statistic = np.mean(df[col]) - null if proportion else np.median(diffs)
            test_statistics.append(test_statistic)
            p_values.append(result.pvalue)
            counts.append(len(data))
    
    return _create_test_frame(
        cols,
        np.array(test_statistics),
        np.array(p_values),
        np.array(counts),
        alpha,
        'prop_diff' if proportion else 'median_diff',
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
            - 'median_diff': The difference between observed and null medians.
            - 'p_value': The calculated p value.
            - 'stat_sig': A boolean flag indicating statistical significance.
            - 'count': The number of valid non-nan observations.
    """

    desc = df[cols].agg(['count', 'median'], axis = 0)

    result = scipy.stats.wilcoxon(
        df[cols].to_numpy() - null,
        alternative = 'two-sided',
        zero_method = 'wilcox',
        nan_policy = 'omit', # type: ignore
    )
    
    return _create_test_frame(
        cols,
        desc.loc['median'].to_numpy() - null,
        result.pvalue, # type: ignore
        desc.loc['count'].to_numpy(),
        alpha,
        'median_diff'
    )

def _create_test_frame(
    cols: list[str],
    test_statistics: np.ndarray,
    p_values: np.ndarray,
    counts: np.ndarray,
    alpha: float,
    statistic_name: str,
) -> pd.DataFrame:
    """Package test results into a DataFrame.

    Args:
        cols (list[str]): The labels associated with each column.
        test_statistics (np.ndarray): The array of test statistics.
        p_values (np.ndarray): The array of p values.
        counts (np.ndarray): The array of column non-nan counts.
        alpha (float): The desired alpha level.
        statistic_name (str): The name of the kind of values in `test_statistics`.

    Returns:
        pd.DataFrame: A DataFrame with indices matching the labels in `cols` and columns `statistic_name`, 'p_value', 'stat_sig', 'count'.
    """
    
    data_dict = {
        f'{statistic_name}': test_statistics.astype(float),
        'p_value': p_values.astype(float),
        'stat_sig': (p_values < alpha).astype(bool),
        'count': counts.astype(int),
    }

    result = pd.DataFrame(
        data_dict,
        index = cols
    )

    return result  
  
# TODO: Add other test methods...
# Add 'bootstrap' method to tests
# test_independent(): independent t, mann-whitney u, one-way anova, kruskal-wallis
# test_dependent(): paired t, wilcoxon signed-rank
# test_dependent_proportion(): mcnemar asymptotic, mcnemar exact binomial, cochran's Q
# test_regression(): linear, logistic

# TODO: Add p-value correction methods...bonferroni, holm-bonferroni, benjamini-hochberg

# TODO: Consider adding test of normality (and maybe leverage alongside sample size when method is unspecified in higher-level funcs?)

# TODO: Consider adding power functions
# If i want to add a power func, desc is df.agg_rows(.., ['count', 'std'])
# rng = np.random.default_rng(0)

# for col in desc.columns:
#     rvs = lambda size: rng.normal(
#         loc = desc[col].loc['alt_mean'],
#         scale = desc[col].loc['std'],
#         size = size,
#     )

#     power = scipy.stats.power(
#         scipy.stats.ttest_1samp,
#         rvs,
#         desc[col].loc['count'],
#         significance = alpha,
#         n_resamples = 2000,
#         kwargs = {'popmean': 0}
#     )

#     desc.loc['power', col] = power.power