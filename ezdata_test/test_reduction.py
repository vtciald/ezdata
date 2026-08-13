import pandas as pd
import numpy as np
from ezdata.processor import DataProcessor
from ezdata import reduction
import pytest

def test_reduce_pca():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'col1': [5.2, 10.0, 7.8, 32.0, 2.0, 3.0, 13.1, 15.4],
        'col2': [0, 1, 0, 1, 1, 0, 0, 1],
        'col3': [2.1, 3.4, 1.2, 5.5, 0.8, 1.1, 4.2, 3.8],
        'ignore_col': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'],
    })

    expected = pd.DataFrame({
        'col1': [5.2, 10.0, 7.8, 32.0, 2.0, 3.0, 13.1, 15.4],
        'col2': [0, 1, 0, 1, 1, 0, 0, 1],
        'col3': [2.1, 3.4, 1.2, 5.5, 0.8, 1.1, 4.2, 3.8],
        'ignore_col': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'],
        'pca_component_0': [-1.1118, 0.6116, -1.2832, 2.9988, -0.9818, -1.6625, 0.2770, 1.1519],
        'pca_component_1': [-0.5794, 0.8000, -0.4728, -0.3216, 1.5903, -0.3027, -1.2625, 0.5487],
    })

    result = dp.reduce_pca(test_df, cols = dp.select(prefix = 'col'), n_components = 2, random_state = 0)

    result['pca_component_0'] = result['pca_component_0'].round(4)
    result['pca_component_1'] = result['pca_component_1'].round(4)
    
    pd.testing.assert_frame_equal(result, expected)

def test_reduce_pca_nan():

    dp = DataProcessor()

    test_df = pd.DataFrame({
        'col1': [np.nan, 10.0, 7.8, 32.0, 2.0, 3.0, 13.1, 15.4],
        'col2': [0, 1, np.nan, 1, 1, 0, 0, 1],
        'col3': [2.1, 3.4, 1.2, 5.5, 0.8, 1.1, 4.2, 3.8],
        'ignore_col': ['a', 'b', 'c', np.nan, 'e', 'f', 'g', 'h'],
    })

    expected = pd.DataFrame({
        'col1': [np.nan, 10.0, 7.8, 32.0, 2.0, 3.0, 13.1, 15.4],
        'col2': [0, 1, np.nan, 1, 1, 0, 0, 1],
        'col3': [2.1, 3.4, 1.2, 5.5, 0.8, 1.1, 4.2, 3.8],
        'ignore_col': ['a', 'b', 'c', np.nan, 'e', 'f', 'g', 'h'],
        'pca_component_0': [np.nan, 0.1659, np.nan, 2.4817, -1.3956, -1.9205, -0.0206, 0.6891],
        'pca_component_1': [np.nan, -0.6573, np.nan, 0.0953, -1.2665, 0.7932, 1.5290, -0.4937],
    })

    result = dp.reduce_pca(test_df, cols = dp.select(prefix = 'col'), n_components = 2, random_state = 0)    

    result['pca_component_0'] = result['pca_component_0'].round(4)
    result['pca_component_1'] = result['pca_component_1'].round(4)

    pd.testing.assert_frame_equal(result, expected)

# Test reduction
test_reduce_pca()
test_reduce_pca_nan()