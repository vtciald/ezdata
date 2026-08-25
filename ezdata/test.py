import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon, kruskal, mannwhitneyu, f_oneway, ttest_ind, chi2_contingency, fisher_exact, ttest_1samp, binomtest
from . import prep
from .selector import Selector, ColumnSelector, PairSelector, GroupSelector
from collections.abc import Sequence
from itertools import combinations
from statsmodels.stats.contingency_tables import mcnemar, cochrans_q
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.multitest import multipletests
from scikit_posthocs import posthoc_dunn, posthoc_tukey
import statsmodels.api as sm

def test_one_sample(
    df: pd.DataFrame,
    method: str,
    *,
    null: float = 0.0,
    alpha: float = 0.05,
    dv: Sequence[str] | str | ColumnSelector | None = None,
) -> pd.DataFrame:
    """Run a one-sample test.

    Args:
        df (pd.DataFrame): The DataFrame.
        method (str): The test method. Supported choices: 't', 'wilcoxon', 'sign'.
        null (float, optional): The value representing the central tendency of the null hypothesis. Defaults to 0.
        alpha (float, optional): The desired alpha. Defaults to 0.05.
        dv (Sequence[str] | str | ColumnSelector | None, optional): Column(s) to include. If None, includes all columns. Defaults to None.

    Notes:
        * 't': One-sample t-test (parametric). Difference between column and null.
        * 'wilcoxon': One-sample Wilcoxon signed-rank test (non-parametric). Difference between column and null.
        * 'sign': One-sample sign test (non-parametric). Difference (ignoring magnitude) between column and null.
    
    Raises:
        ValueError: If string argument for `method` isn't recognized.

    Returns:
        pd.DataFrame: A DataFrame with indices matching the labels in `dv`.
            Columns include:
            - 'test_statistic': A statistic based on the `method` used.
                * T statistic when `method = 't'`.
                * The estimate of the proportion of successes when `method = 'sign'`.
                * The sum of the ranks of the differences above or below zero, whichever is smaller when `method = 'wilcoxon'.
            - 'p_value': The calculated p value.
            - 'stat_sig': A boolean flag indicating statistical significance.
            - 'count': The number of valid non-nan observations.
    """

    dv = Selector.resolve(df, dv)

    if method == 't':
        result = _one_sample_t(df, dv, null, alpha)

    elif method == 'wilcoxon':
        result = _one_sample_wilcoxon(df, dv, null, alpha)
    
    elif method == 'sign':
        result = _one_sample_sign(df, dv, null, alpha)

    else:
        raise ValueError(f'One-sample test method \'{method}\' is not recognized.')

    return result

def test_one_sample_proportion(
    df: pd.DataFrame,
    method: str,
    *,
    null: float = 0.5,
    alpha: float = 0.05,
    dv: Sequence[str] | str | ColumnSelector | None = None,
) -> pd.DataFrame:
    """Run a one-sample test.

    Args:
        df (pd.DataFrame): The DataFrame.
        method (str): The test method. Supported choices: 'exact'.
        null (float, optional): The value representing the central tendency of the null hypothesis. Defaults to 0.5.
        alpha (float, optional): The desired alpha. Defaults to 0.05.
        dv (Sequence[str] | str | ColumnSelector | None, optional): Column(s) to include. If None, includes all columns. Defaults to None.

    Notes:
        * 'exact': One-sample exact binomial test (non-parametric). Difference between column and null.
    
    Raises:
        ValueError: If string argument for `method` isn't recognized.

    Returns:
        pd.DataFrame: A DataFrame with indices matching the labels in `dv`.
            Columns include:
            - 'test_statistic': A statistic based on the `method` used.
                * The estimate of the proportion of successes when `method = 'exact'`.
            - 'p_value': The calculated p value.
            - 'stat_sig': A boolean flag indicating statistical significance.
            - 'count': The number of valid non-nan observations.
    """

    dv = Selector.resolve(df, dv)

    if method == 'exact':
        result = _one_sample_binomial(df, dv, null, alpha)

    else:
        raise ValueError(f'One-sample test method \'{method}\' is not recognized.')

    return result    

def test_independent_proportion(
    df: pd.DataFrame,
    method: str,
    *,
    iv: Sequence[str] | str | ColumnSelector,
    dv: Sequence[str] | str | ColumnSelector | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run an independent-samples test.

    Args:
        df (pd.DataFrame): The DataFrame.
        method (str): The test method. Supported choices: 'fisher_exact', 'chi_square'.
        iv (Sequence[str] | str | ColumnSelector): Column(s) to use as the grouping variable. If one-hot encoded, will be converted to mutually exclusive categories.
        dv (Sequence[str] | str | ColumnSelector | None, optional): Column(s) to evaluate for differences on the basis of `iv`. If None, includes all columns. Defaults to None.
        alpha (float, optional): The desired alpha. Defaults to 0.05.

    Notes:
        * 'fisher_exact': Fisher's exact test (non-parametric). Difference between 2 groups (recommended when sample size < 20 and/or any expected cell count < 5).
        * 'chi_square': Chi-square test (non-parametric). Difference among 2+ groups.
    
    Raises:
        ValueError: If string argument for `method` isn't recognized.

    Returns:
        pd.DataFrame: A DataFrame with indices matching the labels in `dv`.
            Columns include:
            - 'test_statistic': A statistic based on the `method` used.
                * The Chi-squared test statistic when `method = 'chi_squared'`.
                * The prior odds ratio when `method = 'fisher_exact'`.
            - 'p_value': The calculated p value.
            - 'stat_sig': A boolean flag indicating statistical significance.
            - 'count': The number of valid non-nan observations.
    """

    df, iv = prep.dummy_to_categorical(df, cols = iv)
    dv = Selector.resolve(df, dv)
    dv = [col for col in dv if col != iv]

    if method == 'chi_square':
        result = _independent_chi_sq(df, iv, dv, alpha)
    
    elif method == 'fisher_exact':
        result = _independent_fisher_exact(df, iv, dv, alpha)

    else:
        raise ValueError(f'Independent test method \'{method}\' is not recognized.')

    return result

def test_independent(
    df: pd.DataFrame,
    method: str,
    *,
    iv: Sequence[str] | str | ColumnSelector,
    dv: Sequence[str] | str | ColumnSelector | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run an independent-samples test.

    Args:
        df (pd.DataFrame): The DataFrame.
        method (str): The test method. Supported choices: 't', 'mann_whitney', 'anova', 'kruskal_wallis', 'tukey', 'dunn'.
        iv (Sequence[str] | str | ColumnSelector): Column(s) to use as the grouping variable. If one-hot encoded, will be converted to mutually exclusive categories.
        dv (Sequence[str] | str | ColumnSelector | None, optional): Column(s) to evaluate for differences on the basis of `iv`. If None, includes all columns. Defaults to None.
        alpha (float, optional): The desired alpha. Defaults to 0.05.

    Notes:
        * 't': Independent-samples t-test (parametric). Difference between 2 groups.
        * 'mann_whitney': Mann-Whitney U test (non-parametric). Difference between 2 groups.
        * 'anova': One-way ANOVA (parametric). Difference among 2+ groups.
        * 'kruskal_wallis': Kruskal-Wallis H test (non-parametric). Difference among 2+ groups.
        * 'tukey': Tukey's HSD (parametric). Pairwise follow-up to ANOVA.
        * 'dunn': Dunn's test (non-parametric). Pairwise follow-up to Kruskal-Wallis.
    
    Raises:
        ValueError: If string argument for `method` isn't recognized.

    Returns:
        pd.DataFrame: A DataFrame. The index structure varies based on the `method`.
            Columns include:
            - 'test_statistic': A statistic based on the `method` used.
                * T statistic when `method = 't'`.
                * U statistic when `method = 'mann_whitney'`.
                * F statistic when `method = 'anova'`.
                * H statistic when `method = 'kruskal_wallis'`.
            - 'p_value': The calculated p value.
            - 'stat_sig': A boolean flag indicating statistical significance.
            - 'count': The number of valid non-nan observations.
            For follow-up tests, only the 'p_value' and 'stat_sig' column will be non-NaN. No p-value correction method will be applied.
    """

    df, iv = prep.dummy_to_categorical(df, cols = iv)
    dv = Selector.resolve(df, dv)
    dv = [col for col in dv if col != iv]

    if method == 't':
        result = _independent_t(df, iv, dv, alpha)
    
    elif method == 'mann_whitney':
        result = _independent_mann_whitney_u(df, iv, dv, alpha)

    elif method == 'anova':
        result = _independent_one_way_anova(df, iv, dv, alpha)

    elif method == 'kruskal_wallis':
        result = _independent_kruskal_wallis_h(df, iv, dv, alpha)

    elif method == 'tukey':
        result = _independent_tukey(df, iv, dv, alpha)

    elif method == 'dunn':
        result = _independent_dunn(df, iv, dv, alpha)

    else:
        raise ValueError(f'Independent test method \'{method}\' is not recognized.')

    return result

def test_dependent(
    df: pd.DataFrame,
    method: str,
    *,
    dv: Sequence[str] | Sequence[Sequence[str]] | ColumnSelector | PairSelector | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run an dependent-samples test.

    Args:
        df (pd.DataFrame): The DataFrame.
        method (str): The test method. Supported choices: 't', 'wilcoxon'
        dv (Sequence[str] | Sequence[Sequence[str]] | ColumnSelector | PairSelector | None, optional): Column(s) to evaluate for differences on the basis of `iv`. If None, includes all columns. Defaults to None.
        alpha (float, optional): The desired alpha. Defaults to 0.05.

    Notes:
        * 't': Paired-samples t-test (parametric). Difference between 2 columns.
        * 'wilcoxon': Wilcoxon signed-rank test (non-parametric). Difference between 2 columns.
        * If `dv` is a sequence of strings (or ColumnSelector), all combinations of columns will be tested.

    Raises:
        ValueError: If string argument for `method` isn't recognized.

    Returns:
        pd.DataFrame: A DataFrame with multi-index indices, ('group_0', 'group_1')
            Columns include:
            - 'test_statistic': A statistic based on the `method` used.
                * T statistic when `method = 't'`.
                * The sum of the ranks of the differences above or below zero, whichever is smaller when `method = 'wilcoxon'`.
            - 'p_value': The calculated p value.
            - 'stat_sig': A boolean flag indicating statistical significance.
            - 'count': The number of valid non-nan observations.
    """

    dv = Selector.resolve_pair(df, dv)

    if method == 't':
        result = _dependent_t(df, dv, alpha)
    
    elif method == 'wilcoxon':
        result = _dependent_wilcoxon(df, dv, alpha)

    else:
        raise ValueError(f'Dependent test method \'{method}\' is not recognized.')

    return result

def test_dependent_proportion(
    df: pd.DataFrame,
    method: str,
    *,
    dv: Sequence[str] | Sequence[Sequence[str]] | ColumnSelector | PairSelector | GroupSelector | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run an dependent-samples test.

    Args:
        df (pd.DataFrame): The DataFrame.
        method (str): The test method. Supported choices: 'mcnemar_exact', 'mcnemar_asymptotic', 'cochran'.
        dv (Sequence[str] | Sequence[Sequence[str]] | ColumnSelector | PairSelector | None, optional): Column(s) to evaluate for differences on the basis of `iv`. If None, includes all columns. Defaults to None.
        alpha (float, optional): The desired alpha. Defaults to 0.05.

    Notes:
        * 'mcnemar_exact': McNemar's exact test (non-parametric). Difference between 2 columns (recommended when sample size < 25).
        * 'mcnemar_asymptotic': McNemar's asymptotic test (non-parametric). Difference between 2 columns with continuity correction (recommended when sample size >= 25).
        * 'cochran': Cochran's Q test (non-parametric). Difference among 2+ columns.
        * If `dv` is a sequence of strings (or ColumnSelector), all combinations of columns will be tested.

    Raises:
        ValueError: If string argument for `method` isn't recognized.

    Returns:
        pd.DataFrame: A DataFrame with multi-index indices. The structure of these indices varies based on the `method`.
            Columns include:
            - 'test_statistic': A statistic based on the `method` used.
                * T statistic when `method = 't'`.
                * The sum of the ranks of the differences above or below zero, whichever is smaller when `method = 'wilcoxon'`.
            - 'p_value': The calculated p value.
            - 'stat_sig': A boolean flag indicating statistical significance.
            - 'count': The number of valid non-nan observations.
    """

    if method in {'mcnemar_exact', 'mcnemar_asymptotic'}:
        exact = method == 'mcnemar_exact'

        if isinstance(dv, GroupSelector):
            raise TypeError(f'Mcnemar\'s test requires pairs of columns but a GroupSelector was given.')

        else:
            dv = Selector.resolve_pair(df, dv)
            result = _dependent_mcnemar(df, dv, alpha, exact = exact)
    
    elif method == 'cochran':
        dv = Selector.resolve_group(df, dv)
        result = _dependent_cochran(df, dv, alpha)

    else:
        raise ValueError(f'Dependent test method \'{method}\' is not recognized.')

    return result

def test_regression(
    df: pd.DataFrame,
    method: str,
    *,
    iv: Sequence[str] | str | ColumnSelector,
    dv: Sequence[str] | str | ColumnSelector | None = None,
    alpha: float = 0.05,
    interactions: Sequence[str] | Sequence[Sequence[str]] | ColumnSelector | PairSelector | None = None,
    print_summary: bool = False,
    
) -> pd.DataFrame:
    """Run a regression.

    If multiple columns are included for `dv`, will run multiple separate models.

    Args:
        df (pd.DataFrame): The DataFrame.
        method (str): The test method. Supported choices: 'linear', 'logistic', 'ordered_logistic'.
        iv (Sequence[str] | str | ColumnSelector): Column(s) to use as the independent variable(s). It is assumed that a constant is not yet added.
        dv (Sequence[str] | str | ColumnSelector | None, optional): Column(s) to use as the dependent variable(s). If None, includes all columns. Defaults to None.
        alpha (float, optional): The desired alpha. Defaults to 0.05.
        interactions (Sequence[str] | Sequence[Sequence[str]] | ColumnSelector | PairSelector | None): Interaction terms to compute. If given, mean-centers `iv` columns. Defaults to None.
        print_summary (bool, optional): If true, prints the model summary after fit. Defaults to False.

    Notes:
        * 'linear': Ordinary Least Squares (OLS) regression. Predict an interval- or ratio-scale column.
        * 'logistic': Logistic regression. Predict a binary column.
        * 'ordered_logistic': Ordered logistic regression. Predict an ordinal column.
    
    Raises:
        ValueError: If string argument for `method` isn't recognized.

    Returns:
        pd.DataFrame: A DataFrame with multi-index indices, ('dv', 'iv').
            Columns include:
            - 'test_statistic': A statistic based on the `method` used.
                * Beta for predictors and F statistic for overall model when `method = 'linear'`.
                * Log odds ratio for predictors and likelihood ratio for overall model when when `method = 'logistic'` or `method = 'ordered_logistic'`.
            - 'p_value': The calculated p value.
            - 'stat_sig': A boolean flag indicating statistical significance.
            - 'count': The number of valid non-nan observations.
    """

    iv = Selector.resolve(df, iv)
    dv = Selector.resolve(df, dv)

    if interactions is not None:
        interactions = Selector.resolve_pair(df, interactions)

    iv_set = set(iv)
    dv = [col for col in dv if col not in iv_set]

    if method in {'linear', 'logistic', 'ordered_logistic'}:
        result = _regression(df, method, iv, dv, alpha, interactions, print_summary)

    else:
        raise ValueError(f'Regression method \'{method}\' is not recognized.')

    return result

def p_correct(
    df: pd.DataFrame,
    method: str,
    *,
    alpha: float = 0.05, 
) -> pd.DataFrame:
    """Correct p values.

    Args:
        df (pd.DataFrame): A DataFrame containing a column 'p_value'.
        method (str): The p-value correction method. Can be one of 'bonferroni' (or 'bf'), 'holm_bonferroni' (or 'hb'), 'benjamini_hochberg' (or 'bh'), 'benjamini_yekutieli' (or 'by').
        alpha (float, optional): The desired alpha. Defaults to 0.05.

    Notes:
        * 'bonferroni': One-step Bonferroni correction for family-wise error rate (FWER).
        * 'holm_bonferroni': Step-down method with Bonferroni adjustments for FWER. More powerful than 'bonferroni'.
        * 'benjamini_hochberg': Correction for false-discovery rate (FDR). For tests that are independent or positively correlated.
        * 'benjamini_yekutieli': Correction for false-discovery rate (FDR). For tests that are negatively correlated.

    Raises:
        ValueError: If string argument for `method` isn't recognized.
        ValueError: If 'p_value' column is not found in `df`.

    Returns:
        pd.DataFrame: A copy of `df` with columns added:
            - 'p_value_bf' and 'stat_sig_bf' when `method = 'bonferroni'`.
            - 'p_value_hb' and 'stat_sig_hb' when `method = 'holm_bonferroni'`.
            - 'p_value_bh' and 'stat_sig_bh' when `method = 'benjamini_hochberg'`.
            - 'p_value_by' and 'stat_sig_by' when `method = 'benjamini_yekutieli'`.
    """

    method_map = {
        'bonferroni': ('bonferroni', '_bf'),
        'bf': ('bonferroni', '_bf'),
        'holm_bonferroni' : ('holm', '_hb'),
        'hb' : ('holm', '_hb'),
        'benjamini_hochberg' : ('fdr_bh', '_bh'),
        'bh' : ('fdr_bh', '_bh'),
        'benjamini_yekutieli' : ('fdr_by', '_by'),
        'by' : ('fdr_by', '_by'),
    }
    if method not in method_map:
        raise ValueError(f'P-value correction method \'{method}\' is not recognized.')

    elif 'p_value' not in df.columns:
        raise ValueError(f'Column \'p_value\' not found in DataFrame.')
    
    method, method_suffix = method_map[method.lower()]

    df = df.copy()

    stat_sig_corrected, p_value_corrected, _, _ = multipletests(
        df['p_value'],
        alpha = alpha,
        method = method,
        is_sorted = False,
        returnsorted = False
    )

    df['p_value' + method_suffix] = p_value_corrected
    df['stat_sig' + method_suffix] = stat_sig_corrected

    return df

def _regression(
    df: pd.DataFrame,
    method: str,
    iv: list[str],
    dv: list[str],
    alpha: float,
    interactions: list[list[str]] | None,
    print_summary: bool,
) -> pd.DataFrame:
    """Run a regression.

    If multiple columns are included for `dv`, will run multiple separate models.

    Args:
        df (pd.DataFrame): The DataFrame.
        method (str): The test method. Supported choices: 'linear', 'logistic', 'ordered_logistic'.
        iv (Sequence[str]): Column(s) to use as the independent variable(s). It is assumed that a constant is not yet added.
        dv (Sequence[str]): Column(s) to use as the dependent variable(s). If None, includes all columns.
        alpha (float): The desired alpha.
        interactions (Sequence[Sequence[str]] | None): Interaction terms to compute. If given, mean-centers `iv` columns.
        print_summary (bool): If true, prints the model summary after fit.

    Notes:
        * 'linear': Ordinary Least Squares (OLS) regression. Predict an interval- or ratio-scale column.
        * 'logistic': Logistic regression. Predict a binary column.
        * 'ordered_logistic': Ordered logistic regression. Predict an ordinal column.

    Returns:
        pd.DataFrame: A DataFrame with multi-index indices, ('dv', 'iv').
            Columns include:
            - 'test_statistic': A statistic based on the `method` used.
                * Beta for predictors and F statistic for overall model when `method = 'linear'`.
                * Log odds ratio for predictors and likelihood ratio for overall model when when `method = 'logistic'` or `method = 'ordered_logistic'`.
            - 'p_value': The calculated p value.
            - 'stat_sig': A boolean flag indicating statistical significance.
            - 'count': The number of valid non-nan observations.
    """

    counts = []
    index_tuples = []
    test_statistics = []
    p_values = []

    if interactions:
        df = df.copy()
        iv = iv.copy()

        for col in iv:
            if df[col].dropna().nunique() > 2:
                df[col] = df[col] - df[col].mean()

        for col0, col1 in interactions:
            interaction_col = f'{col0}:{col1}'
            df[interaction_col] = df[col0] * df[col1]

            iv.append(interaction_col)

    iv_set = set(iv)

    for dv_col in dv:
        y = df[dv_col].astype(float)

        if method == 'linear':
            X = sm.add_constant(df[iv].astype(float))
            model = sm.OLS(y, X, missing = 'drop')

        elif method == 'logistic':
            X = sm.add_constant(df[iv].astype(float))
            model = sm.Logit(y, X, missing = 'drop')

        elif method == 'ordered_logistic':
            X = df[iv].astype(float)
            model = OrderedModel(y, X, missing = 'drop', method = 'bfgs', distr = 'logit')

        result = model.fit(disp = 0)

        if method == 'linear':
            overall_p = result.f_pvalue
            overall_stat = result.fvalue

        else:
            overall_p = result.llr_pvalue
            overall_stat = result.llr

        index_tuples.append((dv_col, 'OVERALL'))
        test_statistics.append(overall_stat)
        p_values.append(overall_p)
        counts.append(result.nobs)

        if print_summary:
            print(result.summary())

        for iv_name in result.params.index:
            if iv_name in iv_set:
                index_tuples.append((dv_col, iv_name))
                test_statistics.append(result.params[iv_name])
                p_values.append(result.pvalues[iv_name])
                counts.append(result.nobs)

    return _create_test_frame(
        index_tuples,
        np.array(test_statistics),
        np.array(p_values),
        np.array(counts),
        alpha,
        ['dv', 'iv'],
    )

def _dependent_cochran(
   df: pd.DataFrame,
   column_groups: list[list[str]],
   alpha: float,
) -> pd.DataFrame:
    """Run a Cochran's Q test.

    Args:
        df (pd.DataFrame): The DataFrame.
        column_groups (list[tuple[str, str]]): Groups of column labels to test.
        alpha (float): The desired alpha level.

    Returns:
        pd.DataFrame: A DataFrame with indices that are strings of the column list (e.g., `"['Col1_pre', 'Col1_post']"`, `"['Col2_pre', 'Col2_post']"`).
            Columns include:
            - 'test_statistic': Cochran's Q test statistic.
            - 'p_value': The calculated p value.
            - 'stat_sig': A boolean flag indicating statistical significance.
            - 'count': The number of valid non-nan observations.
    """
    
    index_strings = []
    counts = []
    test_statistics = []
    p_values = []

    for col_group in column_groups:
        non_na = df[col_group].notna().all(axis = 1)
        non_tie_row = df[col_group].nunique(axis = 1) != 1
        count = (non_na & non_tie_row).sum()
        result = cochrans_q(df.loc[non_na, col_group].astype(float).values)

        index_strings.append(str(col_group))
        counts.append(count)
        test_statistics.append(result.statistic) # type: ignore
        p_values.append(result.pvalue) # type: ignore

    return _create_test_frame(
        index_strings,
        np.array(test_statistics),
        np.array(p_values),
        np.array(counts), 
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
        pd.DataFrame: A DataFrame with multi-index indices, ('group_0', 'group_1').
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
        count = (df[col0].notna() & df[col1].notna() & (df[col0] != df[col1])).sum()
        result = mcnemar(table, exact)

        index_tuples.append((col0, col1))
        counts.append(count)
        test_statistics.append(result.statistic) # type: ignore
        p_values.append(result.pvalue) # type: ignore

    return _create_test_frame(
        index_tuples,
        np.array(test_statistics),
        np.array(p_values),
        np.array(counts),
        alpha,
        ['group_0', 'group_1'],
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
        pd.DataFrame: A DataFrame with multi-index indices, ('group_0', 'group_1').
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
        result = ttest_rel(
            df[col0].astype(float),
            df[col1].astype(float),
            axis = 0,
            nan_policy = 'omit'
        )

        index_tuples.append((col0, col1))
        counts.append(count)
        test_statistics.append(result.statistic) # type: ignore
        p_values.append(result.pvalue) # type: ignore

    return _create_test_frame(
        index_tuples,
        np.array(test_statistics),
        np.array(p_values),
        np.array(counts), # type: ignore
        alpha,
        ['group_0', 'group_1'],
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
        pd.DataFrame: A DataFrame with multi-index indices, ('group_0', 'group_1').
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
        count = (df[col0].notna() & df[col1].notna() & (df[col0] != df[col1])).sum()
        result = wilcoxon(
            df[col0].astype(float),
            df[col1].astype(float),
            zero_method = 'wilcox',
            method = 'auto',
            nan_policy = 'omit', # type: ignore
        )

        index_tuples.append((col0, col1))
        counts.append(count)
        test_statistics.append(result.statistic) # type: ignore
        p_values.append(result.pvalue) # type: ignore

    return _create_test_frame(
        index_tuples,
        np.array(test_statistics),
        np.array(p_values),
        np.array(counts), # type: ignore
        alpha,
        ['group_0', 'group_1'],
    )

def _independent_kruskal_wallis_h(
   df: pd.DataFrame,
   iv: str,
   dv: list[str],
   alpha: float 
) -> pd.DataFrame:
    """Run a Kruskal-Wallis H test.

    Args:
        df (pd.DataFrame): The DataFrame.
        iv (str): The grouping column label.
        dv (list[str]): The labels of columns to test independence with `iv`.
        alpha (float): The desired alpha level.

    Returns:
        pd.DataFrame: A DataFrame with indices matching the labels in `dv`.
            Columns include:
            - 'test_statistic': The H statistic.
            - 'p_value': The calculated p value.
            - 'stat_sig': A boolean flag indicating statistical significance.
            - 'count': The number of valid non-nan observations.
    """
    
    counts = df.loc[df[iv].notna(), dv].agg('count', axis = 0).values
    group_data = []

    for group in df[iv].unique():
        if group == np.nan or pd.isna(group): continue
        group_filter = df.loc[df[iv] == group, dv].astype(float)
        group_data.append(group_filter.values)

    result = kruskal(*group_data, nan_policy = 'omit') # type: ignore

    return _create_test_frame(
        dv,
        np.array(result.statistic),
        np.array(result.pvalue),
        np.array(counts), # type: ignore
        alpha,
    )

def _independent_dunn(
   df: pd.DataFrame,
   iv: str,
   dv: list[str],
   alpha: float 
) -> pd.DataFrame:
    """Run a Dunn's post hoc-test.

    Args:
        df (pd.DataFrame): The DataFrame.
        iv (str): The grouping column label.
        dv (list[str]): The labels of columns to test independence with `iv`.
        alpha (float): The desired alpha level.

    Returns:
        pd.DataFrame: A DataFrame with indices matching the labels in `dv`.
            Columns include:
            - 'test_statistic': All NaN.
            - 'p_value': The calculated p value.
            - 'stat_sig': A boolean flag indicating statistical significance.
            - 'count': All NaN.
    """

    index_tuples = []
    test_statistics = []
    p_values = []
    counts = []

    unique_groups = [group for group in df[iv].unique() if pd.notna(group)]
    pairs = list(combinations(unique_groups, 2))

    for dv_col in dv:
        filter_df = df.loc[(df[iv].notna()) & (df[dv_col].notna())]
        result = posthoc_dunn(filter_df, val_col = dv_col, group_col = iv)

        for group0, group1 in pairs:
            index_tuples.append((dv_col, group0, group1))
            counts.append(np.nan)
            test_statistics.append(np.nan)
            p_values.append(result.loc[group0, group1])

    return _create_test_frame(
        index_tuples,
        np.array(test_statistics),
        np.array(p_values),
        np.array(counts),
        alpha,
        ['dv', 'group_0', 'group_1'],
        count_int = False,
    )

def _independent_tukey(
   df: pd.DataFrame,
   iv: str,
   dv: list[str],
   alpha: float 
) -> pd.DataFrame:
    """Run a Tukey's post-hoc test.

    Args:
        df (pd.DataFrame): The DataFrame.
        iv (str): The grouping column label.
        dv (list[str]): The labels of columns to test independence with `iv`.
        alpha (float): The desired alpha level.

    Returns:
        pd.DataFrame: A DataFrame with indices matching the labels in `dv`.
            Columns include:
            - 'test_statistic': All NaN.
            - 'p_value': The calculated p value.
            - 'stat_sig': A boolean flag indicating statistical significance.
            - 'count': All NaN.
    """

    index_tuples = []
    test_statistics = []
    p_values = []
    counts = []

    unique_groups = [group for group in df[iv].unique() if pd.notna(group)]
    pairs = list(combinations(unique_groups, 2))

    for dv_col in dv:
        filter_df = df.loc[(df[iv].notna()) & (df[dv_col].notna())]
        result = posthoc_tukey(filter_df, val_col = dv_col, group_col = iv)

        for group0, group1 in pairs:
            index_tuples.append((dv_col, group0, group1))
            counts.append(np.nan)
            test_statistics.append(np.nan)
            p_values.append(result.loc[group0, group1])

    return _create_test_frame(
        index_tuples,
        np.array(test_statistics),
        np.array(p_values),
        np.array(counts),
        alpha,
        ['dv', 'group_0', 'group_1'],
        count_int = False,
    )

def _independent_mann_whitney_u(
   df: pd.DataFrame,
   iv: str,
   dv: list[str],
   alpha: float 
) -> pd.DataFrame:
    """Run a Mann-Whitney U test.

    Args:
        df (pd.DataFrame): The DataFrame.
        iv (str): The grouping column label.
        dv (list[str]): The labels of columns to test independence with `iv`.
        alpha (float): The desired alpha level.

    Returns:
        pd.DataFrame: A DataFrame with multi-index indices, (dv, group_0, group_1).
            Columns include:
            - 'test_statistic': The U statistic.
            - 'p_value': The calculated p value.
            - 'stat_sig': A boolean flag indicating statistical significance.
            - 'count': The number of valid non-nan observations.
    """
    
    unique_groups = [group for group in df[iv].unique() if pd.notna(group)]
    pairs = list(combinations(unique_groups, 2))

    index_tuples = []
    test_statistics = []
    p_values = []
    counts = []

    for dv_col in dv:
        for group0, group1 in pairs:
            group0_filter = df.loc[df[iv] == group0, dv_col].astype(float).dropna() # type: ignore
            group1_filter = df.loc[df[iv] == group1, dv_col].astype(float).dropna() # type: ignore
            count = len(group0_filter) + len(group1_filter)

            result = mannwhitneyu(
                group0_filter, 
                group1_filter, 
                method = 'auto', 
                nan_policy = 'omit' # type: ignore
            ) 
            
            index_tuples.append((dv_col, group0, group1))
            counts.append(count)
            test_statistics.append(result.statistic) # type: ignore
            p_values.append(result.pvalue) # type: ignore

    return _create_test_frame(
        index_tuples,
        np.array(test_statistics),
        np.array(p_values),
        np.array(counts),
        alpha,
        ['dv', 'group_0', 'group_1'],
    )

def _independent_one_way_anova(
   df: pd.DataFrame,
   iv: str,
   dv: list[str],
   alpha: float 
) -> pd.DataFrame:
    """Run a one-way ANOVA.

    Args:
        df (pd.DataFrame): The DataFrame.
        iv (str): The grouping column label.
        dv (list[str]): The labels of columns to test independence with `iv`.
        alpha (float): The desired alpha level.

    Returns:
        pd.DataFrame: A DataFrame with indices matching the labels in `dv`.
            Columns include:
            - 'test_statistic': The F statistic.
            - 'p_value': The calculated p value.
            - 'stat_sig': A boolean flag indicating statistical significance.
            - 'count': The number of valid non-nan observations.
    """
    
    counts = df.loc[df[iv].notna(), dv].agg('count', axis = 0).values
    group_data = []

    for group in df[iv].unique():
        if group == np.nan or pd.isna(group): continue
        group_filter = df.loc[df[iv] == group, dv].astype(float)
        group_data.append(group_filter.values)

    result = f_oneway(*group_data, nan_policy = 'omit') # type: ignore

    return _create_test_frame(
        dv,
        np.array(result.statistic),
        np.array(result.pvalue),
        np.array(counts),
        alpha,
    )

def _independent_t(
   df: pd.DataFrame,
   iv: str,
   dv: list[str],
   alpha: float 
) -> pd.DataFrame:
    """Run an independent-samples t test.

    Args:
        df (pd.DataFrame): The DataFrame.
        iv (str): The grouping column label.
        dv (list[str]): The labels of columns to test independence with `iv`.
        alpha (float): The desired alpha level.

    Returns:
        pd.DataFrame: A DataFrame with multi-index indices, (dv, group_0, group_1).
            Columns include:
            - 'test_statistic': The T statistic.
            - 'p_value': The calculated p value.
            - 'stat_sig': A boolean flag indicating statistical significance.
            - 'count': The number of valid non-nan observations.
    """

    unique_groups = [group for group in df[iv].unique() if pd.notna(group)]
    pairs = list(combinations(unique_groups, 2))

    index_tuples = []
    test_statistics = []
    p_values = []
    counts = []

    for dv_col in dv:
        for group0, group1 in pairs:
            group0_filter = df.loc[df[iv] == group0, dv_col].astype(float).dropna() # type: ignore
            group1_filter = df.loc[df[iv] == group1, dv_col].astype(float).dropna() # type: ignore
            count = len(group0_filter) + len(group1_filter)

            result = ttest_ind(group0_filter, group1_filter, nan_policy = 'omit')

            index_tuples.append((dv_col, group0, group1))
            counts.append(count)
            test_statistics.append(result.statistic) # type: ignore
            p_values.append(result.pvalue) # type: ignore

    return _create_test_frame(
        index_tuples,
        np.array(test_statistics),
        np.array(p_values),
        np.array(counts),
        alpha,
        ['dv', 'group_0', 'group_1'],
    )

def _independent_chi_sq(
    df: pd.DataFrame,
    iv: str,
    dv: list[str],
    alpha: float
) -> pd.DataFrame:
    """Run a Chi-squared test.

    Runs a separate test between each pair of `iv` and one of the `dv`.

    Args:
        df (pd.DataFrame): The DataFrame.
        iv (str): The grouping column label.
        dv (list[str]): The labels of columns to test independence with `iv`.
        alpha (float): The desired alpha level.

    Returns:
        pd.DataFrame: A DataFrame with indices matching the labels in `dv`.
            Columns include:
            - 'test_statistic': The Chi-squared test statistic.
            - 'p_value': The calculated p value.
            - 'stat_sig': A boolean flag indicating statistical significance.
            - 'count': The number of valid non-nan observations.
    """

    test_statistics = []
    p_values = []
    counts = []

    for dv_col in dv:
        count = (df[iv].notna() & df[dv_col].notna()).sum()
        contingency = pd.crosstab(df[iv].values, df[dv_col].values)
        result = chi2_contingency(contingency.values)

        test_statistics.append(result.statistic) # type: ignore
        p_values.append(result.pvalue) # type: ignore
        counts.append(count)
    
    return _create_test_frame(
        dv,
        np.array(test_statistics),
        np.array(p_values),
        np.array(counts),
        alpha,
    )

def _independent_fisher_exact(
    df: pd.DataFrame,
    iv: str,
    dv: list[str],
    alpha: float
) -> pd.DataFrame:
    """Run Fisher's Exact Test of independence.

    Runs a separate test between each pair of `iv` and one of the `dv`. Requires that each column have 2 unique values (ignoring NaN).

    Args:
        df (pd.DataFrame): The DataFrame.
        iv (str): The grouping column label.
        dv (list[str]): The labels of columns to test independence with `iv`.
        alpha (float): The desired alpha level.

    Returns:
        pd.DataFrame: A DataFrame with indices matching the labels in `dv`.
            Columns include:
            - 'test_statistic': The prior odds ratio.
            - 'p_value': The calculated p value.
            - 'stat_sig': A boolean flag indicating statistical significance.
            - 'count': The number of valid non-nan observations.
    """

    test_statistics = []
    p_values = []
    counts = []

    for dv_col in dv:
        count = (df[iv].notna() & df[dv_col].notna()).sum()
        contingency = pd.crosstab(df[iv].values, df[dv_col].values)

        if contingency.shape != (2, 2):
                raise ValueError(
                    f'Fisher\'s Exact Test requires a (2, 2) table. '
                    f'{iv} vs {dv_col} produced a {contingency.shape} table.'
                )

        result = fisher_exact(contingency.values)

        test_statistics.append(result.statistic) # type: ignore
        p_values.append(result.pvalue) # type: ignore
        counts.append(count)
    
    return _create_test_frame(
        dv,
        np.array(test_statistics),
        np.array(p_values),
        np.array(counts),
        alpha,
    )

def _one_sample_t(
    df: pd.DataFrame,
    dv: list[str],
    null: float,
    alpha: float,
) -> pd.DataFrame:
    """Run a one-sample t test.

    Args:
        df (pd.DataFrame): The DataFrame.
        dv (list[str]): The columns on which to operate.
        null (float, optional): The population mean under the null hypothesis. Defaults to 0.
        alpha (float): The desired alpha level.

    Returns:
        pd.DataFrame: A DataFrame with indices matching the labels in `dv`.
            Columns include:
            - 'test_statistic': The t statistic.
            - 'p_value': The calculated p value.
            - 'stat_sig': A boolean flag indicating statistical significance.
            - 'count': The number of valid non-nan observations.
    """

    counts = df[dv].agg('count', axis = 0)

    result = ttest_1samp(
        df[dv].astype(float).to_numpy(),
        popmean = null,
        alternative = 'two-sided',
        nan_policy = 'omit'
    )
    
    return _create_test_frame(
        dv,
        np.array(result.statistic), # type: ignore
        np.array(result.pvalue), # type: ignore
        np.array(counts),
        alpha,
    )

def _one_sample_sign(
    df: pd.DataFrame,
    dv: list[str],
    null: float,
    alpha: float,
) -> pd.DataFrame:
    """Run a one-sample sign test.

    Args:
        df (pd.DataFrame): The DataFrame.
        dv (list[str]): The columns on which to operate.
        null (float): The population median under the null hypothesis.
        alpha (float): The desired alpha level.

    Returns:
        pd.DataFrame: A DataFrame with indices matching the labels in `dv`.
            Columns include:
            - 'test_statistic': The estimate of the proportion positive differences.
            - 'p_value': The calculated p value.
            - 'stat_sig': A boolean flag indicating statistical significance.
            - 'count': The number of valid non-nan observations.
    """

    test_statistics = []
    p_values = []
    counts = []

    for col in dv:
        data = df[col].dropna().astype(int)
        diffs = data - null
        positives = np.sum(diffs > 0)
        total_trials = np.sum(diffs != 0)
        counts.append(total_trials)

        if total_trials == 0:
            test_statistics.append(np.nan)
            p_values.append(np.nan)
        
        else:
            result = binomtest(
                positives,
                total_trials,
                p = 0.5
            )

            test_statistics.append(result.statistic)
            p_values.append(result.pvalue)
    
    return _create_test_frame(
        dv,
        np.array(test_statistics),
        np.array(p_values),
        np.array(counts),
        alpha,
    )

def _one_sample_binomial(
    df: pd.DataFrame,
    dv: list[str],
    null: float,
    alpha: float,
) -> pd.DataFrame:
    """Run a one-sample binomial test.

    Args:
        df (pd.DataFrame): The DataFrame.
        dv (list[str]): The columns on which to operate.
        null (float, optional): The population median under the null hypothesis. Defaults to 0.
        alpha (float): The desired alpha level.

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

    for col in dv:
        data = df[col].dropna().astype(int)
        successes = np.sum(data)
        total_trials = len(data)
        counts.append(len(data))

        if total_trials == 0:
            test_statistics.append(np.nan)
            p_values.append(np.nan)
        
        else:
            result = binomtest(
                successes,
                total_trials,
                p = null,
            )

            test_statistics.append(result.statistic)
            p_values.append(result.pvalue)
    
    return _create_test_frame(
        dv,
        np.array(test_statistics),
        np.array(p_values),
        np.array(counts),
        alpha,
    )

def _one_sample_wilcoxon(
    df: pd.DataFrame,
    dv: list[str],
    null: float,
    alpha: float,
) -> pd.DataFrame:
    """Run a one-sample Wilcoxon signed-rank test.

    Args:
        df (pd.DataFrame): The DataFrame.
        dv (list[str]): The columns on which to operate.
        null (float, optional): The population median under the null hypothesis. Defaults to 0.
        alpha (float): The desired alpha level.

    Returns:
        pd.DataFrame: A DataFrame with indices matching the labels in `dv`.
            Columns include:
            - 'test_statistic': The sum of the ranks of the differences above or below zero, whichever is smaller.
            - 'p_value': The calculated p value.
            - 'stat_sig': A boolean flag indicating statistical significance.
            - 'count': The number of valid non-nan observations.
    """

    counts = ((df[dv] != null) & (df[dv].notna())).sum()

    result = wilcoxon(
        df[dv].astype(float).to_numpy() - null,
        alternative = 'two-sided',
        zero_method = 'wilcox',
        method = 'auto',
        nan_policy = 'omit', # type: ignore
    )
    
    return _create_test_frame(
        dv,
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
    multi_labels: list[str] | None = None,
    count_int: bool = True,
) -> pd.DataFrame:
    """Package test results into a DataFrame.

    Args:
        indices (list[str] | tuple): The labels associated with each column. If a list of tuples, will create a multi-index frame.
        test_statistics (np.ndarray): The array of test statistics.
        p_values (np.ndarray): The array of p values.
        counts (np.ndarray): The array of column non-nan counts.
        alpha (float): The desired alpha level.
        multi_labels (list[str] | None, optional): The labels for the index levels (for a multi-index DataFrame). Must match the length of interior elements of indices. Defaults to None.
        count_int (bool, optional): If true, ensures count is an int dtype. Otherwise, does not modify the dtype. Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame with indices matching `indices` and columns `statistic_name`, 'p_value', 'stat_sig', 'count'.
    """
    
    data_dict = {
        'test_statistic': test_statistics.astype(float),
        'p_value': p_values.astype(float),
        'stat_sig': (p_values < alpha).astype(bool),
        'count': counts.astype(int) if count_int else counts,
    }

    if isinstance(indices[0], tuple):          
        multi_index = pd.MultiIndex.from_tuples(
            indices, # type: ignore
            names = multi_labels,
        )

        frame_index = multi_index

    else:
        frame_index = indices

    result = pd.DataFrame(
        data_dict,
        index = frame_index
    )

    return result