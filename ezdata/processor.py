from . import cluster
from . import plot
from . import prep
from . import stats
from . import reduction
from . import test

class DataProcessor:
    """Create a DataProcessor object to process data.
    """

    # Constants from prep.py
    PATTERN_ALIDA_OTHER_OE = prep.PATTERN_ALIDA_OTHER_OE

    # Methods from prep.py
    clean_arg = staticmethod(prep.clean_arg)
    clean_df = staticmethod(prep.clean_df)
    remove_cols = staticmethod(prep.remove_cols)
    remove_verbal_anchors = staticmethod(prep.remove_verbal_anchors)
    filter_straightliners = staticmethod(prep.filter_straightliners)
    bin = staticmethod(prep.bin)
    filter_by_bounds = staticmethod(prep.filter_by_bounds)
    filter_by_iqr = staticmethod(prep.filter_by_iqr)
    rename_cols = staticmethod(prep.rename_cols)

    # Methods from stats.py
    agg_cols = staticmethod(stats.agg_cols)
    agg_rows = staticmethod(stats.agg_rows)
    calc_ci = staticmethod(stats.calc_ci)