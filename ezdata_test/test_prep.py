import pandas as pd
import numpy as np
import pytest
from ezdata.processor import DataProcessor
from ezdata.selector import Selector
from ezdata import prep
import re

def test_clean_arg():
    
    dp = DataProcessor()

    test_string = 'Hello\xa0World–'
    test_list = [['“Derp’', 'This\u00A0is\u202fbad'], 'Hello\xa0World–']
    test_dict = {'“Derp’': 25, 'Hello\xa0World–': 'Bad\u2014String\u00A0Value'}

    str_result = dp.clean_arg(test_string)
    list_result = dp.clean_arg(test_list)
    dict_result = dp.clean_arg(test_dict)
    
    assert str_result == 'Hello World-'
    assert list_result == [['"Derp\'', 'This is bad'], 'Hello World-']
    assert dict_result == {'"Derp\'': 25, 'Hello World-': 'Bad-String Value'}

def test_clean_df():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        '“Derp’': ['Bad\u2014String\u00A0Value', 'Hello\xa0World–'],
        'Col2': [25, 50]
    })

    expected = pd.DataFrame({
        '"Derp\'': ['Bad-String Value', 'Hello World-'],
        'Col2': [25, 50]
    })

    result = dp.clean_df(test_df)

    pd.testing.assert_frame_equal(result, expected)

def test_clean_df_cols():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        '“Derp’': ['Bad\u2014String\u00A0Value', 'Hello\xa0World–'],
        '“Test’': ['Bad\u2014String\u00A0Value', 'Hello\xa0World–'],
        'Col2': [25, 50]
    })

    expected = pd.DataFrame({
        '"Derp\'': ['Bad-String Value', 'Hello World-'],
        '“Test’': ['Bad\u2014String\u00A0Value', 'Hello\xa0World–'],
        'Col2': [25, 50]
    })

    result = dp.clean_df(test_df, cols = ['“Derp’', 'Col2'])

    pd.testing.assert_frame_equal(result, expected)

def test_clean_df_cols_regex():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        '“Derp’': ['Bad\u2014String\u00A0Value', 'Hello\xa0World–'],
        '“Test’': ['Bad\u2014String\u00A0Value', 'Hello\xa0World–'],
        'Col2': [25, 50]
    })

    expected = pd.DataFrame({
        '"Derp\'': ['Bad-String Value', 'Hello World-'],
        '“Test’': ['Bad\u2014String\u00A0Value', 'Hello\xa0World–'],
        'Col2': [25, 50]
    })

    result = dp.clean_df(test_df, cols = dp.select(pattern = r'er'))

    pd.testing.assert_frame_equal(result, expected)

def test_drop_cols():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [10, 20],
        'Col2': [25, 50],
        'test_Other_0': ['a string would', 'likely appear here']
    })

    expected = pd.DataFrame({
        'Col1': [10, 20],
        'Col2': [25, 50]
    })

    result = dp.drop_cols(test_df, dp.select(pattern = dp.PATTERN_ALIDA_OTHER_OE))

    pd.testing.assert_frame_equal(result, expected)

def test_remove_verbal_anchors():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': ['1 - Not at all', 2, 3, 4, '5 - Totally'],
        'Col2': ['1: Something low', '2', '3', '4', '10: Something high'],
        'Col3': ['1 (Something else low)', 2, 3, 4, '5 (Something else high'],
    })

    expected = pd.DataFrame({
        'Col1': [1, 2, 3, 4, 5],
        'Col2': [1, 2, 3, 4, 10],
        'Col3': [1, 2, 3, 4, 5],
    }).astype({
        'Col1': 'Int64',
        'Col2': 'Int64',
        'Col3': 'Int64'
    })

    result = dp.remove_verbal_anchors(test_df)

    pd.testing.assert_frame_equal(result, expected)

def test_remove_verbal_anchors_new_prefix():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': ['1 - Not at all', 2, 3, 4, '5 - Totally'],
        'Col2': ['1: Something low', '2', '3', '4', '10: Something high'],
        'Col3': ['1 (Something else low)', 2, 3, 4, '5 (Something else high'],
    }).astype({
        'Col1': 'object',
        'Col2': 'object',
        'Col3': 'object',
    })

    expected = pd.DataFrame({
        'Col1': ['1 - Not at all', 2, 3, 4, '5 - Totally'],
        'Col2': ['1: Something low', '2', '3', '4', '10: Something high'],
        'Col3': ['1 (Something else low)', 2, 3, 4, '5 (Something else high'],
        'new_Col1': [1, 2, 3, 4, 5],
        'new_Col2': [1, 2, 3, 4, 10],
        'new_Col3': [1, 2, 3, 4, 5],
    }).astype({
        'Col1': 'object',
        'Col2': 'object',
        'Col3': 'object',
        'new_Col1': 'Int64',
        'new_Col2': 'Int64',
        'new_Col3': 'Int64',
    })

    result = dp.remove_verbal_anchors(test_df, new_col_prefix = 'new_')

    pd.testing.assert_frame_equal(result, expected)

def test_remove_verbal_anchors_cols():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': ['1 - Not at all', 2, 3, 4, '5 - Totally'],
        'Col2': ['1: Something low', '2', '3', '4', '10: Something high'],
        'Col3': ['1 (Something else low)', 2, 3, 4, '5 (Something else high'],
    })

    expected = pd.DataFrame({
        'Col1': [1, 2, 3, 4, 5],
        'Col2': [1, 2, 3, 4, 10],
        'Col3': ['1 (Something else low)', 2, 3, 4, '5 (Something else high'],
    }).astype({
        'Col1': 'Int64',
        'Col2': 'Int64'
    })

    result = dp.remove_verbal_anchors(test_df, cols = ['Col1', 'Col2'])

    pd.testing.assert_frame_equal(result, expected)

def test_recode_values():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': ['Not at all', 'Somewhat', 'Totally'],
        'Col2': ['Not at all', 'Somewhat', 'Totally'],
    })

    expected = pd.DataFrame({
        'Col1': [1, 2, 'Three'],
        'Col2': ['Not at all', 'Somewhat', 'Totally'],
    })

    mapper = {
        'Not at.*': 1,
        'Somewhat': 2,
        'Totally': 'Three',
    }
    result = dp.recode_vals(test_df, mapper, regex_keys = True, cols = dp.select(suffix = '1'))

    pd.testing.assert_frame_equal(result, expected)

def test_recode_values_new_col():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': ['Not at all', 'Somewhat', 'Totally'],
        'Col2': ['Not at all', 'Somewhat', 'Totally'],
    })

    expected = pd.DataFrame({
        'Col1': ['Not at all', 'Somewhat', 'Totally'],
        'Col2': ['Not at all', 'Somewhat', 'Totally'],
        'rc_Col1': [1, 2, 'Three'],
    })

    mapper = {
        'Not at.*': 1,
        'Somewhat': 2,
        'Totally': 'Three',
    }
    result = dp.recode_vals(test_df, mapper, new_col_prefix = 'rc_', regex_keys = True, cols = dp.select(suffix = '1'))

    pd.testing.assert_frame_equal(result, expected)

def test_recode_values_extract():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': ['1 - Not at all', 2, 3, 4, '5 - Totally'],
        'Col2': ['1: Something low', '2', '3', '4', '10: Something high'],
        'Col3': ['1 (Something else low)', 2, 3, 4, '5 (Something else high'],
    })

    expected = pd.DataFrame({
        'Col1': [1, 2, 3, 4, 5],
        'Col2': [1, 2, 3, 4, 10],
        'Col3': ['1 (Something else low)', 2, 3, 4, '5 (Something else high'],
    }).astype({
        'Col1': 'Int64',
        'Col2': 'Int64'
    })

    result = dp.remove_verbal_anchors(test_df, cols = ['Col1', 'Col2'])

    pd.testing.assert_frame_equal(result, expected)

def test_filter_straightliners():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [1, 2, 3, 4, 5],
        'Col2': [1, 2, 3, 7, 7],
        'Col3': [1, 2, 6, 6, 6]
    })

    expected = pd.DataFrame({
        'Col1': [np.nan, np.nan, 3, 4, 5],
        'Col2': [np.nan, np.nan, 3, 7, 7],
        'Col3': [np.nan, np.nan, 6, 6, 6]
    })

    result = dp.filter_straightliners(test_df)

    pd.testing.assert_frame_equal(result, expected)

def test_filter_straightliners_nan():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [np.nan, 2, 3, 4, 5],
        'Col2': [1, 2, 3, 7, 7],
        'Col3': [1, 2, 6, 6, 6]
    })

    expected = pd.DataFrame({
        'Col1': [np.nan, np.nan, 3, 4, 5],
        'Col2': [np.nan, np.nan, 3, 7, 7],
        'Col3': [np.nan, np.nan, 6, 6, 6]
    })

    result = dp.filter_straightliners(test_df)

    pd.testing.assert_frame_equal(result, expected)

def test_filter_straightliners_min_unique():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [1, 2, 3, 4, 5],
        'Col2': [1, 2, 3, 7, 7],
        'Col3': [1, 2, 6, 6, 6]
    })

    expected = pd.DataFrame({
        'Col1': [np.nan, np.nan, np.nan, 4, 5],
        'Col2': [np.nan, np.nan, np.nan, 7, 7],
        'Col3': [np.nan, np.nan, np.nan, 6, 6]
    })

    result = dp.filter_straightliners(test_df, min_unique = 3)

    pd.testing.assert_frame_equal(result, expected)

def test_filter_straightliners_cols():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [1, 2, 3, 4, 5],
        'Col2': [1, 2, 3, 7, 7],
        'Col3': [1, 2, 6, 6, 6],
        'Col4': [1, 1, 1, 1, 1]
    })

    expected = pd.DataFrame({
        'Col1': [np.nan, np.nan, 3, 4, 5],
        'Col2': [np.nan, np.nan, 3, 7, 7],
        'Col3': [np.nan, np.nan, 6, 6, 6],
        'Col4': [1, 1, 1, 1, 1]
    })

    result = dp.filter_straightliners(test_df, cols = ['Col1', 'Col2', 'Col3'])

    pd.testing.assert_frame_equal(result, expected)

def test_bin_i():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [1, 3, 3, 6, 6, 6, 9, 9],
        'Col2': [1, 16, 25, 33, 41, 50, 51, 52]
    })

    expected = pd.DataFrame({
        'Col1': ['(0.992, 3.0]', '(0.992, 3.0]', '(0.992, 3.0]', '(5.0, 7.0]', '(5.0, 7.0]', '(5.0, 7.0]', '(7.0, 9.0]', '(7.0, 9.0]'],
        'Col2': ['(0.949, 13.75]', '(13.75, 26.5]', '(13.75, 26.5]', '(26.5, 39.25]', '(39.25, 52.0]', '(39.25, 52.0]', '(39.25, 52.0]', '(39.25, 52.0]']
    })

    result = dp.bin(test_df, 'i4')

    pd.testing.assert_frame_equal(result, expected)

def test_bin_q():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [1, 3, 3, 6, 6, 6, 9, 9],
        'Col2': [1, 16, 25, 33, 41, 50, 51, 52]
    })

    expected = pd.DataFrame({
        'Col1': ['(0.999, 3.0]', '(0.999, 3.0]', '(0.999, 3.0]', '(3.0, 6.0]', '(3.0, 6.0]', '(3.0, 6.0]', '(6.75, 9.0]', '(6.75, 9.0]'],
        'Col2': ['(0.999, 22.75]', '(0.999, 22.75]', '(22.75, 37.0]', '(22.75, 37.0]', '(37.0, 50.25]', '(37.0, 50.25]', '(50.25, 52.0]', '(50.25, 52.0]']
    })

    result = dp.bin(test_df, 'q4')

    pd.testing.assert_frame_equal(result, expected)

def test_bin_list():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [1, 3, 3, 6, 6, 6, 9, 9],
        'Col2': [1, 16, 25, 33, 41, 50, 51, 52]
    })

    expected = pd.DataFrame({
        'Col1': ['(0, 30]', '(0, 30]', '(0, 30]', '(0, 30]', '(0, 30]', '(0, 30]', '(0, 30]', '(0, 30]'],
        'Col2': ['(0, 30]', '(0, 30]', '(0, 30]', '(30, 60]', '(30, 60]', '(30, 60]', '(30, 60]', '(30, 60]']
    })

    result = dp.bin(test_df, [0, 30, 60])
    
    pd.testing.assert_frame_equal(result, expected)

def test_bin_list_nan():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [np.nan, 3, 3, 6, 6, 6, 9, 9],
        'Col2': [1, 16, 25, 33, 41, 50, 51, 52]
    })

    expected = pd.DataFrame({
        'Col1': [np.nan, '(0, 30]', '(0, 30]', '(0, 30]', '(0, 30]', '(0, 30]', '(0, 30]', '(0, 30]'],
        'Col2': ['(0, 30]', '(0, 30]', '(0, 30]', '(30, 60]', '(30, 60]', '(30, 60]', '(30, 60]', '(30, 60]']
    })

    result = dp.bin(test_df, [0, 30, 60])
    
    pd.testing.assert_frame_equal(result, expected)

def test_filter_by_bounds():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [1, 3, 3, 6, 6, 6, 9, 9],
        'Col2': [1, 16, 25, 33, 41, 50, 51, 52]
    })

    expected = pd.DataFrame({
        'Col1': [np.nan, np.nan, np.nan, 6, 6, 6, 9, 9],
        'Col2': [np.nan, 16, 25, 33, 41, 50, np.nan, np.nan]
    })

    result = dp.filter_by_bounds(test_df, min_val = 5, max_val = 50)
    
    pd.testing.assert_frame_equal(result, expected)

def test_filter_by_iqr():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [-6, 1, 3, 3, 6, 6, 6, 9, 9, 9],
        'Col2': [1, 1, 16, 25, 33, 41, 50, 51, 52, 100]
    })

    expected = pd.DataFrame({
        'Col1': [np.nan, 1, 3, 3, 6, 6, 6, 9, 9, 9],
        'Col2': [1, 1, 16, 25, 33, 41, 50, 51, 52, np.nan]
    })

    result = dp.filter_by_iqr(test_df)
    
    pd.testing.assert_frame_equal(result, expected)

def test_filter_by_iqr_nan():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [-6, np.nan, 3, 3, 6, 6, 6, 9, 9, 9],
        'Col2': [1, 1, 16, 25, 33, 41, 50, 51, 52, 100]
    })

    expected = pd.DataFrame({
        'Col1': [-6, np.nan, 3, 3, 6, 6, 6, 9, 9, 9],
        'Col2': [1, 1, 16, 25, 33, 41, 50, 51, 52, np.nan]
    })

    result = dp.filter_by_iqr(test_df)
    
    pd.testing.assert_frame_equal(result, expected)

def test_filter_by_iqr_factor():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [-6, 1, 3, 3, 6, 6, 6, 9, 9, 9],
        'Col2': [1, 1, 16, 25, 33, 41, 50, 51, 52, 100]
    })

    expected = pd.DataFrame({
        'Col1': [np.nan, 1, 3, 3, 6, 6, 6, 9, 9, 9],
        'Col2': [1, 1, 16, 25, 33, 41, 50, 51, 52, 100]
    })

    result = dp.filter_by_iqr(test_df, factor = 1.6)
    
    pd.testing.assert_frame_equal(result, expected)

def test_filter_by_stdev():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [-6, 1, 3, 3, 6, 6, 6, 9, 9, 9],
        'Col2': [1, 1, 16, 25, 33, 41, 50, 51, 52, 100]
    })

    expected = pd.DataFrame({
        'Col1': [np.nan, 1, 3, 3, 6, 6, 6, 9, 9, 9],
        'Col2': [np.nan, np.nan, 16, 25, 33, 41, 50, 51, 52, np.nan]
    })

    result = dp.filter_by_stdev(test_df, n_stdevs = 1)
    
    pd.testing.assert_frame_equal(result, expected)

def test_resolve_selection():
    
    dp = DataProcessor()

    select_params = [
        {'labels': ['Age', 'Gender', 'Another_1_Column', 'Another_2_Column', 'Another_Test', 'InviteSegment_Group1']},
        {'prefix': 'Another'},
        {'suffix': 'Column'},
        {'pattern': r'\d'},
        {'pattern': re.compile(r'group', re.IGNORECASE)},
        {
            'labels': ['Age', 'Gender', 'Another_1_Column', 'Another_2_Column', 'Another_Test', 'InviteSegment_Group1'],
            'prefix': 'Another',
            'suffix': 'Column',
            'pattern': r'\d',
        },
    ]

    select_results = [
        ['Age', 'Gender', 'Another_1_Column', 'Another_2_Column', 'Another_Test', 'InviteSegment_Group1'],
        ['Another_1_Column', 'Another_2_Column', 'Another_3_Column', 'Another_4_Column', 'Another_5_Column', 'Another_Test_Column', 'Another_Test'],
        ['Group_Column', 'Another_1_Column', 'Another_2_Column', 'Another_3_Column', 'Another_4_Column', 'Another_5_Column', 'Another_Test_Column'],
        ['InviteSegment_Group1', 'InviteSegment_Group2', 'Another_1_Column', 'Another_2_Column', 'Another_3_Column', 'Another_4_Column', 'Another_5_Column'],
        ['InviteSegment_Group1', 'InviteSegment_Group2', 'Group_Column'],
        ['Another_1_Column', 'Another_2_Column'],
    ]

    test_df = pd.DataFrame({
        'Age': [1, 2, 3],
        'Gender': [4, 5, 6],
        'InviteSegment_Group1': [7, 8, 9],
        'InviteSegment_Group2': [10, 11, 12],
        'Group_Column': [13, 14, 15],
        'Another_1_Column': [16, 17, 18],
        'Another_2_Column': [19, 20, 21],
        'Another_3_Column': [22, 23, 24],
        'Another_4_Column': [25, 26, 27],
        'Another_5_Column': [28, 29, 30],
        'Another_Test_Column': [31, 32, 33],
        'Another_Test': [34, 35, 36],
    })

    for params, expected_cols in zip(select_params, select_results):
        result_cols = Selector.resolve(test_df, dp.select(**params))

        assert sorted(expected_cols) == sorted(result_cols) 

def test_resolve_selection_no_params():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Test_col_one': [1, 2, 3],
        'Test_col2': [4, 5, 6],
        'test_col3': [7, 8, 9],
        'Col4_test': [10, 11, 12],
        'Col5_test': [13, 14, 15],
        'Col6': [16, 17, 18],
    })

    expected_cols = test_df.columns.tolist()
    result_cols = Selector.resolve(test_df, dp.select())

    assert expected_cols == result_cols

def test_resolve_selection_no_match():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Test_col_one': [1, 2, 3],
        'Test_col2': [4, 5, 6],
        'test_col3': [7, 8, 9],
        'Col4_test': [10, 11, 12],
        'Col5_test': [13, 14, 15],
        'Col6': [16, 17, 18],
    })

    expected_cols = []
    result_cols = Selector.resolve(test_df, dp.select(labels = ' '))

    assert expected_cols == result_cols

def test_resolve_selection_empty_string():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Test_col_one': [1, 2, 3],
        'Test_col2': [4, 5, 6],
        'test_col3': [7, 8, 9],
        'Col4_test': [10, 11, 12],
        'Col5_test': [13, 14, 15],
        'Col6': [16, 17, 18],
    })

    expected_cols = []
    result_cols = Selector.resolve(test_df, dp.select(labels = ''))

    assert expected_cols == result_cols

def test_resolve_selection_str():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Test_col_one': [1, 2, 3],
        'Test_col2': [4, 5, 6],
        'test_col3': [7, 8, 9],
        'Col4_test': [10, 11, 12],
        'Col5_test': [13, 14, 15],
        'Col6': [16, 17, 18],
    })

    expected_cols = ['Test_col_one']
    result_cols = Selector.resolve(test_df, 'Test_col_one')

    assert sorted(expected_cols) == sorted(result_cols)

def test_resolve_selection_list():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Test_col_one': [1, 2, 3],
        'Test_col2': [4, 5, 6],
        'test_col3': [7, 8, 9],
        'Col4_test': [10, 11, 12],
        'Col5_test': [13, 14, 15],
        'Col6': [16, 17, 18],
    })

    expected_cols = ['Test_col_one', 'test_col3']
    result_cols = Selector.resolve(test_df, ['Test_col_one', 'test_col3'])

    assert sorted(expected_cols) == sorted(result_cols)

def test_resolve_selection_tuple():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Test_col_one': [1, 2, 3],
        'Test_col2': [4, 5, 6],
        'test_col3': [7, 8, 9],
        'Col4_test': [10, 11, 12],
        'Col5_test': [13, 14, 15],
        'Col6': [16, 17, 18],
    })

    expected_cols = ['Test_col_one', 'test_col3']
    result_cols = Selector.resolve(test_df, ('Test_col_one', 'test_col3'))

    assert sorted(expected_cols) == sorted(result_cols)

def test_resolve_selection_exclude_labels():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Test_col_one': [1, 2, 3],
        'Test_col2': [4, 5, 6],
        'test_col3': [7, 8, 9],
        'Col4_test': [10, 11, 12],
        'Col5_test': [13, 14, 15],
        'Col6': [16, 17, 18],
    })

    expected_cols = ['Test_col_one', 'test_col3']
    result_cols = Selector.resolve(test_df, dp.select(labels = ['Test_col_one', 'test_col3'], exclude_labels = 'test_col3'))

    assert sorted(expected_cols) == sorted(result_cols)

def test_resolve_selection_exclude_prefix():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Test_col_one': [1, 2, 3],
        'Test_col2': [4, 5, 6],
        'test_col3': [7, 8, 9],
        'Col4_test': [10, 11, 12],
        'Col5_test': [13, 14, 15],
        'Col6': [16, 17, 18],
    })

    expected_cols = ['Test_col_one', 'Test_col2', 'test_col3']
    result_cols = Selector.resolve(test_df, dp.select(exclude_prefix = 'Col'))

    assert sorted(expected_cols) == sorted(result_cols)

def test_resolve_selection_exclude_suffix():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Test_col_one': [1, 2, 3],
        'Test_col2': [4, 5, 6],
        'test_col3': [7, 8, 9],
        'Col4_test': [10, 11, 12],
        'Col5_test': [13, 14, 15],
        'Col6': [16, 17, 18],
    })

    expected_cols = ['Test_col_one', 'Test_col2', 'test_col3', 'Col6']
    result_cols = Selector.resolve(test_df, dp.select(exclude_suffix = 'test'))

    assert sorted(expected_cols) == sorted(result_cols)

def test_resolve_selection_exclude_pattern():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Test_col_one': [1, 2, 3],
        'Test_col2': [4, 5, 6],
        'test_col3': [7, 8, 9],
        'Col4_test': [10, 11, 12],
        'Col5_test': [13, 14, 15],
        'Col6': [16, 17, 18],
    })

    expected_cols = ['Test_col_one']
    result_cols = Selector.resolve(test_df, dp.select(exclude_pattern = r'ol\d'))

    assert sorted(expected_cols) == sorted(result_cols)

def test_rename_cols():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Test_col_one': [1, 2, 3],
        'Test_col2': [4, 5, 6],
        'test_col3': [7, 8, 9],
        'Col4_test': [10, 11, 12],
        'Col5_test': [13, 14, 15],
        'Col6': [16, 17, 18],
    })

    mapper = {
        'Test_col_one': 'New label 1',
        '4_t': 'New label 2'
    }

    expected = pd.DataFrame({
        'New label 1': [1, 2, 3],
        'Test_col2': [4, 5, 6],
        'test_col3': [7, 8, 9],
        'Col4_test': [10, 11, 12],
        'Col5_test': [13, 14, 15],
        'Col6': [16, 17, 18],
    })

    result = dp.rename_cols(test_df, mapper)

    pd.testing.assert_frame_equal(result, expected)

def test_rename_cols_regex_arg():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Test_col_one': [1, 2, 3],
        'Test_col2': [4, 5, 6],
        'test_col3': [7, 8, 9],
        'Col4_test': [10, 11, 12],
        'Col5_test': [13, 14, 15],
        'Col6': [16, 17, 18],
    })

    mapper = {
        'Test_col_one': 'New label 1',
        '4_t': 'New label 2'
    }

    expected = pd.DataFrame({
        'New label 1': [1, 2, 3],
        'Test_col2': [4, 5, 6],
        'test_col3': [7, 8, 9],
        'New label 2': [10, 11, 12],
        'Col5_test': [13, 14, 15],
        'Col6': [16, 17, 18],
    })

    result = dp.rename_cols(test_df, mapper, regex_keys = True)

    pd.testing.assert_frame_equal(result, expected)

def test_rename_cols_regex_pattern():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Test_col_one': [1, 2, 3],
        'Test_col2': [4, 5, 6],
        'test_col3': [7, 8, 9],
        'Col4_test': [10, 11, 12],
        'Col5_test': [13, 14, 15],
        'Col6': [16, 17, 18],
    })

    mapper = {
        'Test_col_one': 'New label 1',
        re.compile('4_t'): 'New label 2'
    }

    expected = pd.DataFrame({
        'New label 1': [1, 2, 3],
        'Test_col2': [4, 5, 6],
        'test_col3': [7, 8, 9],
        'New label 2': [10, 11, 12],
        'Col5_test': [13, 14, 15],
        'Col6': [16, 17, 18],
    })

    result = dp.rename_cols(test_df, mapper)

    pd.testing.assert_frame_equal(result, expected)

def test_rename_cols_duplicate():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Test_col_one': [1, 2, 3],
        'Test_col2': [4, 5, 6],
        'test_col3': [7, 8, 9],
        'Col4_test': [10, 11, 12],
        'Col5_test': [13, 14, 15],
        'Col6': [16, 17, 18],
    })

    mapper = {
        'Test_col_one': 'New label 1',
        re.compile(r'col\d', re.IGNORECASE): 'New label 2'
    }

    data_rows = [
        [1, 4, 7, 10, 13, 16],
        [2, 5, 8, 11, 14, 17],
        [3, 6, 9, 12, 15, 18],
    ]
    columns = [
        'New label 1', 'New label 2', 'New label 2', 'New label 2', 'New label 2', 'New label 2',
    ]

    expected = pd.DataFrame(
        data_rows,
        columns = columns
    )

    result = dp.rename_cols(test_df, mapper)

    pd.testing.assert_frame_equal(result, expected)

def test_rename_cols_extract():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1: Extra stuff here': ['Not at all', 'Somewhat', 'Totally'],
        'Col2: Something else': ['Not at all', 'Somewhat', 'Totally'],
    })

    expected = pd.DataFrame({
        'Col1': ['Not at all', 'Somewhat', 'Totally'],
        'Col2': ['Not at all', 'Somewhat', 'Totally'],
    })

    mapper_regex = r'(^.+):.+'
    result = dp.rename_cols(test_df, mapper_regex)

    pd.testing.assert_frame_equal(result, expected)

def test_dummy_to_categorical():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [1, 1, 0, 0],
        'Col2': [1, 0, 1, 0],
    })

    expected = pd.DataFrame({
        'Col1': [1, 1, 0, 0],
        'Col2': [1, 0, 1, 0],
        'NewCol': ['Col1 & Col2', 'Col1', 'Col2', 'Category_None'],
    })

    result, _ = dp.dummy_to_categorical(test_df, cols = ['Col1', 'Col2'], new_col_name = 'NewCol')

    pd.testing.assert_frame_equal(result, expected)

def test_dummy_to_categorical_nan():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [1, 1, 0, 0, np.nan],
        'Col2': [1, 0, 1, 0, np.nan],
    })

    expected = pd.DataFrame({
        'Col1': [1, 1, 0, 0, np.nan],
        'Col2': [1, 0, 1, 0, np.nan],
        'NewCol': ['Col1 & Col2', 'Col1', 'Col2', 'Category_None', np.nan],
    })

    result, _ = dp.dummy_to_categorical(test_df, cols = ['Col1', 'Col2'], new_col_name = 'NewCol')

    pd.testing.assert_frame_equal(result, expected)

def test_dummy_to_categorical_nan_part():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [1, 1, 0, 0, 1],
        'Col2': [1, 0, 1, 0, np.nan],
    })

    expected = pd.DataFrame({
        'Col1': [1, 1, 0, 0, 1],
        'Col2': [1, 0, 1, 0, np.nan],
        'NewCol': ['Col1 & Col2', 'Col1', 'Col2', 'Category_None', np.nan],
    })

    result, _ = dp.dummy_to_categorical(test_df, cols = ['Col1', 'Col2'], new_col_name = 'NewCol')

    pd.testing.assert_frame_equal(result, expected)

def test_dummy_to_categorical_error():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Col1': [1, 1, 5, 0, 1],
        'Col2': [1, 0, 1, 0, np.nan],
    })

    expected = pd.DataFrame({
        'Col1': [1, 1, 5, 0, 1],
        'Col2': [1, 0, 1, 0, np.nan],
        'NewCol': ['Col1 & Col2', 'Col1', 'Col2', 'Category_None', np.nan],
    })

    with pytest.raises(ValueError):
        result, _ = dp.dummy_to_categorical(test_df, cols = ['Col1', 'Col2'], new_col_name = 'NewCol')

def test_resolve_selection_pair_root():
    
    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Age': [1, 2, 3],
        'Gender': [4, 5, 6],
        'InviteSegment_Group1': [7, 8, 9],
        'InviteSegment_Group2': [10, 11, 12],
        'Group_Column': [13, 14, 15],
        'Another_1_Column': [16, 17, 18],
        'Another_2_Column': [19, 20, 21],
        'Another_3_Column': [22, 23, 24],
        'Another_4_Column': [25, 26, 27],
        'Another_5_Column': [28, 29, 30],
        'Another_Test_Column': [31, 32, 33],
        'Another_Test': [34, 35, 36],
    })

    expected1 = [
        ['InviteSegment_Group1', 'InviteSegment_Group2'],
        ['Another_1_Column', 'Another_2_Column'],
        ['Another_1_Column', 'Another_3_Column'],
        ['Another_2_Column', 'Another_3_Column']
    ]

    expected2 = [
        ['Another_1_Column', 'Another_2_Column'],
        ['Another_1_Column', 'Another_3_Column'],
        ['Another_2_Column', 'Another_3_Column'],
    ]

    expected3 = [
        ['Another_1_Column', 'Another_Test'],
        ['Age', 'Gender'],
    ]

    result1 = Selector.resolve_pair(test_df, dp.select_pair_by_root(r'[123]'))

    assert expected1 == result1

    result2 = Selector.resolve_pair(test_df, dp.select_pair_by_root(r'[123]' , prefix = 'Another', suffix = 'Column'))

    assert expected2 == result2

    result3 = Selector.resolve_pair(test_df, [('Another_1_Column', 'Another_Test'), ('Age', 'Gender')])

    assert expected3 == result3

def test_resolve_selection_pair_match():
    
    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Age': [1, 2, 3],
        'Gender': [4, 5, 6],
        'InviteSegment_Group1': [7, 8, 9],
        'InviteSegment_Group2': [10, 11, 12],
        'Group_Column': [13, 14, 15],
        'Another_1_Column': [16, 17, 18],
        'Another_2_Column': [19, 20, 21],
        'Another_3_Column': [22, 23, 24],
        'Another_4_Column': [25, 26, 27],
        'Another_5_Column': [28, 29, 30],
        'Another_Test_Column': [31, 32, 33],
        'Another_Test': [34, 35, 36],
    })

    expected1 = [
        ['InviteSegment_Group1', 'Another_1_Column'], 
        ['InviteSegment_Group2', 'Another_2_Column'],
    ]

    expected2 = []

    result1 = Selector.resolve_pair(test_df, dp.select_pair_by_match(r'[123]'))

    assert expected1 == result1

    result2 = Selector.resolve_pair(test_df, dp.select_pair_by_match(r'[123]' , prefix = 'Another', suffix = 'Column'))

    assert expected2 == result2


def test_resolve_selection_group_root():
    
    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Age': [1, 2, 3],
        'Gender': [4, 5, 6],
        'InviteSegment_Group1': [7, 8, 9],
        'InviteSegment_Group2': [10, 11, 12],
        'Group_Column': [13, 14, 15],
        'Another_1_Column': [16, 17, 18],
        'Another_2_Column': [19, 20, 21],
        'Another_3_Column': [22, 23, 24],
        'Another_4_Column': [25, 26, 27],
        'Another_5_Column': [28, 29, 30],
        'Another_6_Column': [28, 29, 30],
        'Another_Test_Column': [31, 32, 33],
        'Another_Test': [34, 35, 36],
        'Some1_start': [34, 35, 36],
        'Some1_mid': [34, 35, 36],
        'Some1_end': [34, 35, 36],
        'Some2_start': [34, 35, 36],
        'Some2_mid': [34, 35, 36],
        'Some2_end': [34, 35, 36],
        'Some3_start': [34, 35, 36],
        'Some3_mid': [34, 35, 36],
        'Some3_end': [34, 35, 36],
    })

    expected1 = [
        ['InviteSegment_Group1', 'InviteSegment_Group2'],
        ['Another_1_Column', 'Another_2_Column', 'Another_3_Column'],
        ['Some1_start', 'Some2_start', 'Some3_start'],
        ['Some1_mid', 'Some2_mid', 'Some3_mid'],
        ['Some1_end', 'Some2_end', 'Some3_end']
    ]

    expected2 = [
        ['Some1_start', 'Some1_mid', 'Some1_end'],
        ['Some2_start', 'Some2_mid', 'Some2_end'],
        ['Some3_start', 'Some3_mid', 'Some3_end'],
    ]

    expected3 = [
        ['Some2_start', 'Some2_mid', 'Some2_end'],
        ['Some3_start', 'Some3_mid', 'Some3_end']
    ]

    result1 = Selector.resolve_group(test_df, dp.select_group_by_root(r'[123]'))

    assert expected1 == result1

    result2 = Selector.resolve_group(test_df, dp.select_group_by_root(r'_(start|mid|end)'))

    assert expected2 == result2

    result3 = Selector.resolve_group(test_df, [('Some2_start', 'Some2_mid', 'Some2_end'), ('Some3_start', 'Some3_mid', 'Some3_end')])
    
    assert expected3 == result3

def test_resolve_selection_group_error():
    
    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Age': [1, 2, 3],
        'Gender': [4, 5, 6],
        'InviteSegment_Group1': [7, 8, 9],
        'InviteSegment_Group2': [10, 11, 12],
        'Group_Column': [13, 14, 15],
        'Another_1_Column': [16, 17, 18],
        'Another_2_Column': [19, 20, 21],
        'Another_3_Column': [22, 23, 24],
        'Another_4_Column': [25, 26, 27],
        'Another_5_Column': [28, 29, 30],
        'Another_6_Column': [28, 29, 30],
        'Another_Test_Column': [31, 32, 33],
        'Another_Test': [34, 35, 36],
        'Some1_start': [34, 35, 36],
        'Some1_mid': [34, 35, 36],
        'Some1_end': [34, 35, 36],
        'Some2_start': [34, 35, 36],
        'Some2_mid': [34, 35, 36],
        'Some2_end': [34, 35, 36],
        'Some3_start': [34, 35, 36],
        'Some3_mid': [34, 35, 36],
        'Some3_end': [34, 35, 36],
    })

    with pytest.raises(TypeError):
        result1 = Selector.resolve_group(test_df, 0) # type: ignore

    with pytest.raises(TypeError):
        result2 = Selector.resolve_group(test_df, [['Test1', 3], ['Test1', 'Test3']]) # type: ignore

    with pytest.raises(ValueError):
        result4 = Selector.resolve_pair(test_df, [['Test1', 'Test2', 'Test3'], ['Test1', 'Test2', 'Test3']])


def test_resolve_selection_group_match():
    
    dp = DataProcessor()

    test_df = pd.DataFrame({
        'Age': [1, 2, 3],
        'Gender': [4, 5, 6],
        'InviteSegment_Group1': [7, 8, 9],
        'InviteSegment_Group2': [10, 11, 12],
        'Group_Column': [13, 14, 15],
        'Another_1_Column': [16, 17, 18],
        'Another_2_Column': [19, 20, 21],
        'Another_3_Column': [22, 23, 24],
        'Another_4_Column': [25, 26, 27],
        'Another_5_Column': [28, 29, 30],
        'Another_6_Column': [28, 29, 30],
        'Another_Test_Column': [31, 32, 33],
        'Another_Test': [34, 35, 36],
        'Some1_start': [34, 35, 36],
        'Some1_mid': [34, 35, 36],
        'Some1_end': [34, 35, 36],
        'Some2_start': [34, 35, 36],
        'Some2_mid': [34, 35, 36],
        'Some2_end': [34, 35, 36],
        'Some3_start': [34, 35, 36],
        'Some3_mid': [34, 35, 36],
        'Some3_end': [34, 35, 36],
    })

    expected1 = [
        ['InviteSegment_Group1', 'Another_1_Column', 'Some1_start', 'Some1_mid', 'Some1_end'],
        ['InviteSegment_Group2', 'Another_2_Column', 'Some2_start', 'Some2_mid', 'Some2_end'],
        ['Another_3_Column', 'Some3_start', 'Some3_mid', 'Some3_end'],
    ]

    expected2 = [
        ['Some1_start', 'Some2_start', 'Some3_start'],
        ['Some1_mid', 'Some2_mid', 'Some3_mid'],
        ['Some1_end', 'Some2_end', 'Some3_end'],
    ]

    result1 = Selector.resolve_group(test_df, dp.select_group_by_match(r'[123]'))
    
    assert expected1 == result1

    result2 = Selector.resolve_group(test_df, dp.select_group_by_match(r'_(start|mid|end)'))

    assert expected2 == result2


# Standardizing characters in arguments
test_clean_arg()

# Standardizing characters in dfs
test_clean_df()
test_clean_df_cols()
test_clean_df_cols_regex()

# Removing columns from dfs
test_drop_cols()

# Removing verbal anchors from values
test_remove_verbal_anchors()
test_remove_verbal_anchors_cols()
test_remove_verbal_anchors_new_prefix()

# Filtering straightliners
test_filter_straightliners()
test_filter_straightliners_min_unique()
test_filter_straightliners_cols()
test_filter_straightliners_nan()

# Binning values
test_bin_i()
test_bin_q()
test_bin_list()
test_bin_list_nan()

# Filtering by bounds
test_filter_by_bounds()

# Filtering by IQR
test_filter_by_iqr()
test_filter_by_iqr_factor()
test_filter_by_iqr_nan()

# Filtering by stdev
test_filter_by_stdev()

# Test getting column names
test_resolve_selection()
test_resolve_selection_no_params()
test_resolve_selection_no_match()
test_resolve_selection_empty_string()
test_resolve_selection_str()
test_resolve_selection_list()
test_resolve_selection_tuple()

# Test renaming columns
test_rename_cols()
test_rename_cols_regex_arg()
test_rename_cols_regex_pattern()
test_rename_cols_duplicate()
test_rename_cols_extract()

# Test recoding values
test_recode_values()
test_recode_values_new_col()

# Test converting dummy cols to categorical
test_dummy_to_categorical()
test_dummy_to_categorical_nan()
test_dummy_to_categorical_nan_part()
test_dummy_to_categorical_error()

# Test resolving pair/group selection
test_resolve_selection_pair_root()
test_resolve_selection_pair_match()
test_resolve_selection_group_root()
test_resolve_selection_group_match()
test_resolve_selection_group_error()

# Test exclusions in column resolution
test_resolve_selection_exclude_labels()
test_resolve_selection_exclude_prefix()
test_resolve_selection_exclude_suffix()
test_resolve_selection_exclude_pattern()