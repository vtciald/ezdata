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
            'test_statistic': [2.6458, -0.3568, 2.3684],
            'p_value': [0.0331, 0.7318, 0.0497],
            'stat_sig': [True, False, True],
            'count': [8, 8, 8],
        },
        index = ['Col1', 'Col2', 'Col3'],
    )

    result = dp.test_one_sample(test_df, 't')

    result['test_statistic'] = result['test_statistic'].round(4)
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
            'test_statistic': [0.0, 4.0, 2.0],
            'p_value': [0.125, 1.0, 0.0312],
            'stat_sig': [False, False, True],
            'count': [8, 8, 8],
        },
        index = ['Col1', 'Col2', 'Col3'],
    )

    result = dp.test_one_sample(test_df, 'wilcoxon')

    result['test_statistic'] = result['test_statistic'].round(4)
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
            'test_statistic': [1.00, 0.500, 0.875],
            'p_value': [0.125, 1.0, 0.0703],
            'stat_sig': [False, False, False],
            'count': [8, 8, 8],
        },
        index = ['Col1', 'Col2', 'Col3'],
    )

    result = dp.test_one_sample(test_df, 'sign')

    result['test_statistic'] = result['test_statistic'].round(4)
    result['p_value'] = result['p_value'].round(4)

    pd.testing.assert_frame_equal(result, expected)

def test_one_sample_exact_proportion():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        'Col2': [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        'Col3': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    })

    expected = pd.DataFrame(
        {
            'test_statistic': [0.5, 0.75, 0.0833],
            'p_value': [1.0, 0.146, 0.0063],
            'stat_sig': [False, False, True],
            'count': [12, 12, 12],
        },
        index = ['Col1', 'Col2', 'Col3'],
    )

    result = dp.test_one_sample_proportion(test_df, 'exact')

    result['test_statistic'] = result['test_statistic'].round(4)
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

    result = dp.test_independent_proportion(test_df, 'chi_square', iv = 'Col1', dv = ['Col2', 'Col3'])

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

    result = dp.test_independent_proportion(test_df, 'chi_square', iv = ['Col1', 'Col2'], dv = 'Col3')

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
            'test_statistic': [2.6667],
            'p_value': [0.6145],
            'stat_sig': [False],
            'count': [16],
        },
        index = ['Col2'],
    )

    result = dp.test_independent_proportion(test_df, 'fisher_exact', iv = 'Col1', dv = 'Col2')

    result['test_statistic'] = result['test_statistic'].round(4)
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
        result = dp.test_independent_proportion(test_df, 'fisher_exact', iv = 'Col1', dv = 'Col3')

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

    result = dp.test_independent(test_df, 'anova', dv = ['Col1', 'Col2'], iv = 'Group')

    result['test_statistic'] = result['test_statistic'].round(4)
    result['p_value'] = result['p_value'].round(4)
    
    pd.testing.assert_frame_equal(result, expected)

def test_independent_t():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [10, 42, 64, 75, 2, 635, 78, 8, 53, 74, np.nan, 86, 86, 43, 31, 75, 86, 63, 42, 4, 57, 698, 34],
        'Col2': [43, 64, 85, 243, 745, 9, 97, 46, 53, 42, 765, 86, 96, 680, 53, 75, 500, 43, 75, 85, 45, 34, 65],
        'Group': [np.nan, 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A']
    })

    multi_index = pd.MultiIndex.from_tuples(
        [
            ('Col1', 'A', 'B'),
            ('Col1', 'A', 'C'),
            ('Col1', 'B', 'C'),
            ('Col2', 'A', 'B'),
            ('Col2', 'A', 'C'),
            ('Col2', 'B', 'C'),
        ],
        names = ['dv', 'group_0', 'group_1'],
    )

    expected = pd.DataFrame(
        {
            'test_statistic': [-1.3207, -1.4374, -0.1630, 2.4767, 2.1200, -1.4579],
            'p_value': [0.2112, 0.1762, 0.8732, 0.0278, 0.0538, 0.1705],
            'stat_sig': [False, False, False, True, False, False],
            'count': [14, 14, 14, 15, 15, 14],
        },
        index = multi_index,
    )

    result = dp.test_independent(test_df, 't', dv = ['Col1', 'Col2'], iv = 'Group')
    
    result['test_statistic'] = result['test_statistic'].round(4)
    result['p_value'] = result['p_value'].round(4)
    
    pd.testing.assert_frame_equal(result, expected)

def test_independent_mann_whitney():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [10, 42, 64, 75, 2, 635, 78, 8, 53, 74, np.nan, 86, 86, 43, 31, 75, 86, 63, 42, 4, 57, 698, 34],
        'Col2': [43, 64, 85, 243, 745, 9, 97, 46, 53, 42, 765, 86, 96, 680, 53, 75, 500, 43, 75, 85, 45, 34, 65],
        'Group': [np.nan, 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A']
    })

    multi_index = pd.MultiIndex.from_tuples(
        [
            ('Col1', 'A', 'B'),
            ('Col1', 'A', 'C'),
            ('Col1', 'B', 'C'),
            ('Col2', 'A', 'B'),
            ('Col2', 'A', 'C'),
            ('Col2', 'B', 'C'),
        ],
        names = ['dv', 'group_0', 'group_1'],
    )

    expected = pd.DataFrame(
        {
            'test_statistic': [8.5, 7.0, 15.5, 46.5, 38.0, 16.0],
            'p_value': [0.0474, 0.0293, 0.2764, 0.0369, 0.2712, 0.3056],
            'stat_sig': [True, True, False, True, False, False],
            'count': [14, 14, 14, 15, 15, 14],
        },
        index = multi_index,
    )

    result = dp.test_independent(test_df, 'mann_whitney', dv = ['Col1', 'Col2'], iv = 'Group')
    
    result['test_statistic'] = result['test_statistic'].round(4)
    result['p_value'] = result['p_value'].round(4)
    
    pd.testing.assert_frame_equal(result, expected)

def test_independent_kruskal_wallis():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [10, 42, 64, 75, 2, 635, 78, 8, 53, 74, np.nan, 86, 86, 43, 31, 75, 86, 63, 42, 4, 57, 698, 34],
        'Col2': [43, 64, 85, 243, 745, 9, 97, 46, 53, 42, 765, 86, 96, 680, 53, 75, 500, 43, 75, 85, 45, 34, 65],
        'Group': [np.nan, 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A']
    })

    expected = pd.DataFrame(
        {
            'test_statistic': [6.9790, 4.8936],
            'p_value': [0.0305, 0.0866],
            'stat_sig': [True, False],
            'count': [21, 22],
        },
        index = ['Col1', 'Col2'],
    )

    result = dp.test_independent(test_df, 'kruskal_wallis', dv = ['Col1', 'Col2'], iv = 'Group')

    result['test_statistic'] = result['test_statistic'].round(4)
    result['p_value'] = result['p_value'].round(4)
    
    pd.testing.assert_frame_equal(result, expected)

def test_dependent_t():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col_pre': [10, 42, 64, 75, 2, 635, 78, 8, 53, 74, np.nan, 86, 86, 43, 31, 75, 86, 63, 42, 4, 57, 698, 34],
        'Col_post': [43, 64, 85, 243, 745, 9, 97, 46, 53, 42, 765, 86, 96, 680, 53, 75, 500, 43, 75, 85, 45, 34, 65],
        'Col3': [33, 0, -10, 44, 0, -40, 97, 46, 0, -5, 765, 0, 96, 32, 0, 75, 25, 43, 0, -10, 0, 220, 65],
    })

    multi_index = pd.MultiIndex.from_tuples(
        [
            ('Col_pre', 'Col_post'),
            ('Col_pre', 'Col3'),
            ('Col_post', 'Col3'),
        ],
        names = ['group_0', 'group_1'],
    )

    expected = pd.DataFrame(
        {
            'test_statistic': [-0.6396, 2.0604, 2.4497],
            'p_value': [0.5294, 0.0520, 0.0227],
            'stat_sig': [False, False, True],
            'count': [22, 22, 23],
        },
        index = multi_index,
    )

    result = dp.test_dependent(test_df, 't', dv = ['Col_pre', 'Col_post', 'Col3'])

    result['test_statistic'] = result['test_statistic'].round(4)
    result['p_value'] = result['p_value'].round(4)
    
    pd.testing.assert_frame_equal(result, expected)

def test_dependent_t_pair():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1_pre': [10, 42, 64, 75, 2, 635, 78, 8, 53, 74, np.nan, 86, 86, 43, 31, 75, 86, 63, 42, 4, 57, 698, 34],
        'Col1_post': [43, 64, 85, 243, 745, 9, 97, 46, 53, 42, 765, 86, 96, 680, 53, 75, 500, 43, 75, 85, 45, 34, 65],
        'Col2_pre': [10, 42, 64, 75, 2, 635, 78, 8, 53, 74, np.nan, 86, 86, 43, 31, 75, 86, 63, 42, 4, 57, 698, 34],
        'Col2_post': [43, 64, 85, 243, 745, 9, 97, 46, 53, 42, 765, 86, 96, 680, 53, 75, 500, 43, 75, 85, 45, 34, 65],
        'Test_pre': [33, 0, -10, 44, 0, -40, 97, 46, 0, -5, 765, 0, 96, 32, 0, 75, 25, 43, 0, -10, 0, 220, 65],
        'Test_post': [33, 0, -10, 44, 0, -40, 97, 46, 0, -5, 765, 0, 96, 32, 0, 75, 25, 43, 0, -10, 0, 220, 65],
    })

    multi_index = pd.MultiIndex.from_tuples(
        [
            ('Col1_pre', 'Col1_post'),
            ('Col2_pre', 'Col2_post'),
        ],
        names = ['group_0', 'group_1'],
    )

    expected = pd.DataFrame(
        {
            'test_statistic': [-0.6396, -0.6396],
            'p_value': [0.5294, 0.5294],
            'stat_sig': [False, False],
            'count': [22, 22],
        },
        index = multi_index,
    )

    result = dp.test_dependent(test_df, 't', dv = dp.select_pair_by_root(r'(pre|post)', prefix = 'Col'))

    result['test_statistic'] = result['test_statistic'].round(4)
    result['p_value'] = result['p_value'].round(4)
    
    pd.testing.assert_frame_equal(result, expected)

def test_dependent_wilcoxon():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1_pre': [10, 42, 64, 75, 2, 635, 78, 8, 53, 74, np.nan, 86, 86, 43, 31, 75, 86, 63, 42, 4, 57, 698, 34],
        'Col1_post': [43, 64, 85, 243, 745, 9, 97, 46, 53, 42, 765, 86, 96, 680, 53, 75, 500, 43, 75, 85, 45, 34, 65],
        'Col_pre': [10, 42, 64, 1, 2, 635, 78, 8, 53, 13, 13, 86, 86, 43, 31, 75, 4, 14, 42, 4, 57, 698, 34],
        'Col_post': [43, 64, 85, 243, 745, 9, 97, 46, 53, 42, 765, 86, 96, 680, 53, 75, 500, 43, 75, 85, 45, 34, 65],
    })

    multi_index = pd.MultiIndex.from_tuples(
        [
            ('Col1_pre', 'Col1_post'),
            ('Col_pre', 'Col_post'),
        ],
        names = ['group_0', 'group_1'],
    )

    expected = pd.DataFrame(
        {
            'test_statistic': [49.0, 36.0],
            'p_value': [0.0641, 0.0100],
            'stat_sig': [False, True],
            'count': [22, 23],
        },
        index = multi_index,
    )

    result = dp.test_dependent(test_df, 'wilcoxon', dv = dp.select_pair_by_root(r'(pre|post)'))

    result['test_statistic'] = result['test_statistic'].round(4)
    result['p_value'] = result['p_value'].round(4)
    
    pd.testing.assert_frame_equal(result, expected)

def test_dependent_proportion_mcnemar_exact():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1_pre': [np.nan, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
        'Col1_post': [1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        'Col2_pre': [1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
        'Col2_post': [0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    })

    multi_index = pd.MultiIndex.from_tuples(
        [
            ('Col1_pre', 'Col1_post'),
            ('Col2_pre', 'Col2_post'),
        ],
        names = ['group_0', 'group_1'],
    )

    expected = pd.DataFrame(
        {
            'test_statistic': [3.0, 2.0],
            'p_value': [0.0574, 0.0074],
            'stat_sig': [False, True],
            'count': [31, 32],
        },
        index = multi_index,
    )

    result = dp.test_dependent_proportion(test_df, 'mcnemar_exact', dv = dp.select_pair_by_root(r'(pre|post)'))

    result['test_statistic'] = result['test_statistic'].round(4)
    result['p_value'] = result['p_value'].round(4)
    
    pd.testing.assert_frame_equal(result, expected)

def test_dependent_proportion_mcnemar_asymptotic():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1_pre': [np.nan, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
        'Col1_post': [1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        'Col2_pre': [1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
        'Col2_post': [0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    })

    multi_index = pd.MultiIndex.from_tuples(
        [
            ('Col1_pre', 'Col1_post'),
            ('Col2_pre', 'Col2_post'),
        ],
        names = ['group_0', 'group_1'],
    )

    expected = pd.DataFrame(
        {
            'test_statistic': [3.5, 6.6667],
            'p_value': [0.0614, 0.0098],
            'stat_sig': [False, True],
            'count': [31, 32],
        },
        index = multi_index,
    )

    result = dp.test_dependent_proportion(test_df, 'mcnemar_asymptotic', dv = dp.select_pair_by_root(r'(pre|post)'))

    result['test_statistic'] = result['test_statistic'].round(4)
    result['p_value'] = result['p_value'].round(4)
    
    pd.testing.assert_frame_equal(result, expected)

def test_dependent_proportion_cochran():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1_pre': [np.nan, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
        'Col1_post': [1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        'Col2_pre': [1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
        'Col2_post': [0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    })

    expected = pd.DataFrame(
        {
            'test_statistic': [4.5714, 8.0667],
            'p_value': [0.0325, 0.0045],
            'stat_sig': [True, True],
            'count': [31, 32],
        },
        index = ["['Col1_pre', 'Col1_post']", "['Col2_pre', 'Col2_post']"]
    )

    result = dp.test_dependent_proportion(test_df, 'cochran', dv = dp.select_pair_by_root(r'(pre|post)'))

    result['test_statistic'] = result['test_statistic'].round(4)
    result['p_value'] = result['p_value'].round(4)
    
    pd.testing.assert_frame_equal(result, expected)

def test_regression_linear():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'iv1': [np.nan, 5.2, 10, 7.8, 32, 2, 3, 13.1, 15.4, 54, 17.0, 2, 13, 1.4, 3, 16, 23.1, 57.4, 32.0, 3.1, 7.5, 4.2, 8.9, 6.4, 9.1, 1.8],
        'iv2': [0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0],
        'dv1': [14.2, 18.5, 12.1, 24.3, 21.0, 15.4, 19.8, 29.1, 33.5, 22.0, 31.4, 24.9, 52.1, 9.4, 44.5, 41.2, 36.8, 74.2, 48.0, 11.1, 16.5, 15.2, 22.9, 14.4, 21.1, 8.8],
        'dv2': [1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, np.nan, 1, 0, 1], 
    })

    multi_index = pd.MultiIndex.from_tuples(
        [
            ('dv1', 'iv1'),
            ('dv1', 'iv2'),
            ('dv2', 'iv1'),
            ('dv2', 'iv2'),
        ],
        names = ['dv', 'iv'],
    )

    expected = pd.DataFrame(
        {
            'test_statistic': [0.6068, 2.4375, 0.0000, -1.0000],
            'p_value': [0.0023, 0.6470, 0.0020, 0.0000],
            'stat_sig': [True, False, True, True],
            'count': [25, 25, 24, 24],
        },
        index = multi_index,
    )

    result = dp.test_regression(test_df, 'linear', dv = ['dv1', 'dv2'], iv = ['iv1', 'iv2'])

    result['test_statistic'] = result['test_statistic'].round(4)
    result['p_value'] = result['p_value'].round(4)
    
    pd.testing.assert_frame_equal(result, expected)

def test_regression_logistic():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'iv1': [np.nan, 5.2, 10, 7.8, 32, 2, 3, 13.1, 15.4, 54, 17.0, 2, 13, 1.4, 3, 16, 23.1, 57.4, 32.0, 3.1, 7.5, 4.2, 8.9, 6.4, 9.1, 1.8],
        'iv2': [0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0],
        'dv1': [0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, np.nan, 0, 0, 0],
        'dv2': [0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0],
    })

    multi_index = pd.MultiIndex.from_tuples(
        [
            ('dv1', 'iv1'),
            ('dv1', 'iv2'),
            ('dv2', 'iv1'),
            ('dv2', 'iv2'),
        ],
        names = ['dv', 'iv'],
    )

    expected = pd.DataFrame(
        {
            'test_statistic': [0.2854, -1.6272, 0.0753, -0.4724],
            'p_value': [0.0384, 0.2377, 0.1549, 0.6127],
            'stat_sig': [True, False, False, False],
            'count': [24, 24, 25, 25],
        },
        index = multi_index,
    )

    result = dp.test_regression(test_df, 'logistic', dv = ['dv1', 'dv2'], iv = ['iv1', 'iv2'])

    result['test_statistic'] = result['test_statistic'].round(4)
    result['p_value'] = result['p_value'].round(4)
    
    pd.testing.assert_frame_equal(result, expected)

def test_regression_logit():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'iv1': [np.nan, 5.2, 10, 7.8, 32, 2, 3, 13.1, 15.4, 54, 17.0, 2, 13, 1.4, 3, 16, 23.1, 57.4, 32.0, 3.1, 7.5, 4.2, 8.9, 6.4, 9.1, 1.8],
        'iv2': [0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0],
        'dv1': [1, 2, 1, 0, 2, 0, 0, 1, 0, 2, 2, 0, 1, 0, 1, 2, 2, 2, 2, 1, 1, 0, np.nan, 0, 0, 0],
        'dv2': [0, 0, 1, 0, 2, 1, 0, 2, 0, 2, 2, 1, 2, 0, 1, 2, 2, 2, 0, 0, 1, 1, 2, 1, 0, 0],
    })

    multi_index = pd.MultiIndex.from_tuples(
        [
            ('dv1', 'iv1'),
            ('dv1', 'iv2'),
            ('dv2', 'iv1'),
            ('dv2', 'iv2'),
        ],
        names = ['dv', 'iv'],
    )

    expected = pd.DataFrame(
        {
            'test_statistic': [0.3304, -1.5496, 0.1284, -0.5436],
            'p_value': [0.0064, 0.2099, 0.0515, 0.5423],
            'stat_sig': [True, False, False, False],
            'count': [24, 24, 25, 25],
        },
        index = multi_index,
    )

    result = dp.test_regression(test_df, 'logit', dv = ['dv1', 'dv2'], iv = ['iv1', 'iv2'])

    result['test_statistic'] = result['test_statistic'].round(4)
    result['p_value'] = result['p_value'].round(4)
    
    pd.testing.assert_frame_equal(result, expected)

# Test one sample methods
test_one_sample_t()
test_one_sample_wilcoxon()
test_one_sample_sign()
test_one_sample_exact_proportion()

# Test independent methods
test_independent_proportion_chi_sq()
test_independent_proportion_chi_sq_dummy_to_categorical()
test_independent_proportion_fisher_exact()
test_independent_proportion_fisher_exact_error()
test_independent_anova()
test_independent_t()
test_independent_mann_whitney()
test_independent_kruskal_wallis()

# Test dependent methods
test_dependent_t()
test_dependent_t_pair()
test_dependent_wilcoxon()
test_dependent_proportion_mcnemar_exact()
test_dependent_proportion_mcnemar_asymptotic()
test_dependent_proportion_cochran()

# Test regression
test_regression_linear()
test_regression_logistic()
test_regression_logit()
