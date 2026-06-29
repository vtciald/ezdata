import pandas as pd
import numpy as np
from ezdata.processor import DataProcessor
from ezdata import test
import pytest

def test_one_sample_t():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [1, 1, 1, 1, 0, 0, 0, 0],
        'Col2': [-1, 0, 1, 0, 1, 0, -2, 0],
        'Col3': [10, -1, 2, 3, 4, 1, 2, 1]
    })

    expected = pd.DataFrame(
        {
            'mean_diff': [0.5, -0.125, 2.75],
            'p_value': [0.0331, 0.7318, 0.0497],
            'stat_sig': [True, False, True],
            'count': [8, 8, 8],
        },
        index = ['Col1', 'Col2', 'Col3'],
    )

    result = dp.test_one_sample(test_df, 't')

    result['mean_diff'] = result['mean_diff'].round(4)
    result['p_value'] = result['p_value'].round(4)

    pd.testing.assert_frame_equal(result, expected)

def test_one_sample_wilcoxon():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [1, 1, 1, 1, 0, 0, 0, 0],
        'Col2': [-1, 0, 1, 0, 1, 0, -2, 0],
        'Col3': [10, -1, 2, 3, 4, 1, 2, 1]
    })

    expected = pd.DataFrame(
        {
            'median_diff': [0.5, 0.0, 2.0],
            'p_value': [0.125, 1.0, 0.0312],
            'stat_sig': [False, False, True],
            'count': [8, 8, 8],
        },
        index = ['Col1', 'Col2', 'Col3'],
    )

    result = dp.test_one_sample(test_df, 'wilcoxon')

    result['median_diff'] = result['median_diff'].round(4)
    result['p_value'] = result['p_value'].round(4)

    pd.testing.assert_frame_equal(result, expected)

def test_one_sample_sign():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [1, 1, 1, 1, 0, 0, 0, 0],
        'Col2': [-1, 0, 1, 0, 1, 0, -2, 0],
        'Col3': [10, -1, 2, 3, 4, 1, 2, 1]
    })

    expected = pd.DataFrame(
        {
            'median_diff': [0.5, 0.0, 2.0],
            'p_value': [0.125, 1.0, 0.0703],
            'stat_sig': [False, False, False],
            'count': [8, 8, 8],
        },
        index = ['Col1', 'Col2', 'Col3'],
    )

    result = dp.test_one_sample(test_df, 'sign')

    result['median_diff'] = result['median_diff'].round(4)
    result['p_value'] = result['p_value'].round(4)

    pd.testing.assert_frame_equal(result, expected)

def test_one_sample_sign_proportion():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        'Col2': [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        'Col3': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    })

    expected = pd.DataFrame(
        {
            'prop_diff': [0.0, 0.25, -0.4167],
            'p_value': [1.0, 0.146, 0.0063],
            'stat_sig': [False, False, True],
            'count': [12, 12, 12],
        },
        index = ['Col1', 'Col2', 'Col3'],
    )

    result = dp.test_one_sample_proportion(test_df, 'sign')

    result['prop_diff'] = result['prop_diff'].round(4)
    result['p_value'] = result['p_value'].round(4)

    pd.testing.assert_frame_equal(result, expected)

def test_independent_proportion_chi_sq():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        'Col2': [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        'Col3': ['yes', 'yes', 'yes', 'yes', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'maybe'],
    })

    expected = pd.DataFrame(
        {
            'test_statistic': [1.7778, 6.2857],
            'p_value': [0.1824, 0.0432],
            'stat_sig': [False, True],
            'count': [12, 12],
        },
        index = ['Col2', 'Col3'],
    )

    result = dp.test_independent_proportion(test_df, 'chi_squared', group_col = 'Col1', target_cols = ['Col2', 'Col3'])

    result['test_statistic'] = result['test_statistic'].round(4)
    result['p_value'] = result['p_value'].round(4)

    pd.testing.assert_frame_equal(result, expected)

def test_independent_proportion_chi_sq_dummy_to_categorical():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        'Col2': [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        'Col3': ['yes', 'yes', 'yes', 'yes', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'maybe'],
    })

    expected = pd.DataFrame(
        {
            'test_statistic': [8.5714],
            'p_value': [0.0728],
            'stat_sig': [False],
            'count': [12],
        },
        index = ['Col3'],
    )

    result = dp.test_independent_proportion(test_df, 'chi_squared', group_col = ['Col1', 'Col2'], target_cols = 'Col3')

    result['test_statistic'] = result['test_statistic'].round(4)
    result['p_value'] = result['p_value'].round(4)

    pd.testing.assert_frame_equal(result, expected)

def test_independent_proportion_fisher_exact():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
        'Col2': [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    })

    expected = pd.DataFrame(
        {
            'odds_ratio': [2.6667],
            'p_value': [0.6145],
            'stat_sig': [False],
            'count': [16],
        },
        index = ['Col2'],
    )

    result = dp.test_independent_proportion(test_df, 'fisher_exact', group_col = 'Col1', target_cols = 'Col2')

    result['odds_ratio'] = result['odds_ratio'].round(4)
    result['p_value'] = result['p_value'].round(4)
    
    pd.testing.assert_frame_equal(result, expected)

def test_independent_proportion_fisher_exact_error():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        'Col2': [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        'Col3': ['yes', 'yes', 'yes', 'yes', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'maybe'],
    })

    expected = pd.DataFrame(
        {
            'test_statistic': [1.7778, 6.2857],
            'p_value': [0.1824, 0.0432],
            'stat_sig': [False, True],
            'count': [12, 12],
        },
        index = ['Col2', 'Col3'],
    )

    with pytest.raises(ValueError):
        result = dp.test_independent_proportion(test_df, 'fisher_exact', group_col = 'Col1', target_cols = 'Col3')

def test_independent_anova():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [10, 42, 64, 75, 2, 635, 78, 8, 53, 74, np.nan, 86, 86, 43, 31, 75, 86, 63, 42, 4, 57, 698, 34],
        'Col2': [43, 64, 85, 243, 745, 9, 97, 46, 53, 42, 765, 86, 96, 680, 53, 75, 500, 43, 75, 85, 45, 34, 65],
        'Group': [np.nan, 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A']
    })

    expected = pd.DataFrame(
        {
            'test_statistic': [0.9805, 5.2185],
            'p_value': [0.3943, 0.0156],
            'stat_sig': [False, True],
            'count': [21, 22],
        },
        index = ['Col1', 'Col2'],
    )

    result = dp.test_independent(test_df, 'anova', target_cols = ['Col1', 'Col2'], group_col = 'Group')

    result['test_statistic'] = result['test_statistic'].round(4)
    result['p_value'] = result['p_value'].round(4)
    
    pd.testing.assert_frame_equal(result, expected)


# Test one sample methods
test_one_sample_t()
test_one_sample_wilcoxon()
test_one_sample_sign()
test_one_sample_sign_proportion()

# Test independent methods
test_independent_proportion_chi_sq()
test_independent_proportion_chi_sq_dummy_to_categorical()
test_independent_proportion_fisher_exact()
test_independent_proportion_fisher_exact_error()
test_independent_anova()

