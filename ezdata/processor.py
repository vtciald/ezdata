import pandas as pd
import re
from . import cluster
from . import plot
from . import prep
from . import stats
from . import reduction
from . import test
from .selector import Selector, ColumnSelector, PairSelector, GroupSelector
from collections.abc import Sequence
    
class DataProcessor:
    """Create a DataProcessor object to process data.
    """

    def select(
        self,
        labels: Sequence[str] | str | None = None,
        prefix: str | None = None,
        suffix: str | None = None,
        pattern: re.Pattern | str | None = None,
        exclude_labels: Sequence[str] | str | None = None,
        exclude_prefix: str | None = None,
        exclude_suffix: str | None = None,
        exclude_pattern: re.Pattern | str | None = None,
    ) -> ColumnSelector:
        """Create a ColumnSelector instance.

        Selection parameters (e.g., `labels`, `prefix`, etc.) are used in conjunction with one another, taking the intersection of matching columns. In other words, only columns matching all selection criteria will be selected.

        Args:
            labels (list[str] | set[str] | str | None, optional): Full column labels to select. Defaults to None.
            prefix (str | None, optional): The prefix of columns to select. Defaults to None.
            suffix (str | None, optional): The suffix of columns to select. Defaults to None.
            pattern (str | re.Pattern | None, optional): A regex pattern describing columns to select. Defaults to None.
            exclude_labels (Sequence[str] | str | None, optional): Full column labels to omit. Defaults to None.
            exclude_prefix (str | None, optional): The prefix of columns to omit. Defaults to None.
            exclude_suffix (str | None, optional): The suffix of columns to omit. Defaults to None.
            exclude_pattern (str | re.Pattern | None, optional): A regex pattern describing columns to omit. Defaults to None.

        Note:
            If all selection arguments are None, all columns will be selected.
        """

        return ColumnSelector(
            labels = labels,
            prefix = prefix,
            suffix = suffix,
            pattern = pattern,
            exclude_labels = exclude_labels,
            exclude_prefix = exclude_prefix,
            exclude_suffix = exclude_suffix,
            exclude_pattern = exclude_pattern,
        )
    
    def select_pair_by_root(
        self,
        pair_pattern: re.Pattern | str,
        labels: Sequence[str] | str | None = None,
        prefix: str | None = None,
        suffix: str | None = None,
        pattern: re.Pattern | str | None = None,
        exclude_labels: Sequence[str] | str | None = None,
        exclude_prefix: str | None = None,
        exclude_suffix: str | None = None,
        exclude_pattern: re.Pattern | str | None = None,
    ) -> PairSelector:
        """Create a PairSelector instance.

        Creates groups of columns that match if substrings matching `group_pattern` were to be removed.

        Selection parameters (e.g., `labels`, `prefix`, etc.) are used in conjunction with one another, taking the intersection of matching columns. In other words, only columns matching all selection criteria will be selected.

        Args:
            pair_pattern (re.Pattern | str): A regex pattern that describes the portion of the label that differentiates paired columns.
            labels (list[str] | set[str] | str | None, optional): Full column labels to select. Defaults to None.
            prefix (str | None, optional): The prefix of columns to select. Defaults to None.
            suffix (str | None, optional): The suffix of columns to select. Defaults to None.
            pattern (str | re.Pattern | None, optional): A regex pattern describing columns to select. Defaults to None.
            exclude_labels (Sequence[str] | str | None, optional): Full column labels to omit. Defaults to None.
            exclude_prefix (str | None, optional): The prefix of columns to omit. Defaults to None.
            exclude_suffix (str | None, optional): The suffix of columns to omit. Defaults to None.
            exclude_pattern (str | re.Pattern | None, optional): A regex pattern describing columns to omit. Defaults to None.
        """

        return PairSelector(
            pair_pattern,
            match = False,
            labels = labels,
            prefix = prefix,
            suffix = suffix,
            pattern = pattern,
            exclude_labels = exclude_labels,
            exclude_prefix = exclude_prefix,
            exclude_suffix = exclude_suffix,
            exclude_pattern = exclude_pattern,
        )
    
    def select_pair_by_match(
        self,
        pair_pattern: re.Pattern | str,
        labels: Sequence[str] | str | None = None,
        prefix: str | None = None,
        suffix: str | None = None,
        pattern: re.Pattern | str | None = None,
        exclude_labels: Sequence[str] | str | None = None,
        exclude_prefix: str | None = None,
        exclude_suffix: str | None = None,
        exclude_pattern: re.Pattern | str | None = None,
    ) -> PairSelector:
        """Create a PairSelector instance.

        Creates pairs of columns that match on their first matching `group_pattern`.

        Selection parameters (e.g., `labels`, `prefix`, etc.) are used in conjunction with one another, taking the intersection of matching columns. In other words, only columns matching all selection criteria will be selected.

        Args:
            pair_pattern (re.Pattern | str): A regex pattern that describes the portion of the label that differentiates paired columns.
            labels (list[str] | set[str] | str | None, optional): Full column labels to select. Defaults to None.
            prefix (str | None, optional): The prefix of columns to select. Defaults to None.
            suffix (str | None, optional): The suffix of columns to select. Defaults to None.
            pattern (str | re.Pattern | None, optional): A regex pattern describing columns to select. Defaults to None.
            exclude_labels (Sequence[str] | str | None, optional): Full column labels to omit. Defaults to None.
            exclude_prefix (str | None, optional): The prefix of columns to omit. Defaults to None.
            exclude_suffix (str | None, optional): The suffix of columns to omit. Defaults to None.
            exclude_pattern (str | re.Pattern | None, optional): A regex pattern describing columns to omit. Defaults to None.
        """

        return PairSelector(
            pair_pattern,
            match = True,
            labels = labels,
            prefix = prefix,
            suffix = suffix,
            pattern = pattern,
            exclude_labels = exclude_labels,
            exclude_prefix = exclude_prefix,
            exclude_suffix = exclude_suffix,
            exclude_pattern = exclude_pattern,
        )
    
    def select_group_by_root(
        self,
        group_pattern: re.Pattern | str,
        labels: Sequence[str] | str | None = None,
        prefix: str | None = None,
        suffix: str | None = None,
        pattern: re.Pattern | str | None = None,
        exclude_labels: Sequence[str] | str | None = None,
        exclude_prefix: str | None = None,
        exclude_suffix: str | None = None,
        exclude_pattern: re.Pattern | str | None = None,
    ) -> GroupSelector:
        """Create a GroupSelector instance.

        Creates groups of columns that match if substrings matching `group_pattern` were to be removed.

        Selection parameters (e.g., `labels`, `prefix`, etc.) are used in conjunction with one another, taking the intersection of matching columns. In other words, only columns matching all selection criteria will be selected.

        Args:
            group_pattern (re.Pattern | str): A regex pattern that describes the portion of the label that differentiates members of column groups.
            labels (list[str] | set[str] | str | None, optional): Full column labels to select. Defaults to None.
            prefix (str | None, optional): The prefix of columns to select. Defaults to None.
            suffix (str | None, optional): The suffix of columns to select. Defaults to None.
            pattern (str | re.Pattern | None, optional): A regex pattern describing columns to select. Defaults to None.
            exclude_labels (Sequence[str] | str | None, optional): Full column labels to omit. Defaults to None.
            exclude_prefix (str | None, optional): The prefix of columns to omit. Defaults to None.
            exclude_suffix (str | None, optional): The suffix of columns to omit. Defaults to None.
            exclude_pattern (str | re.Pattern | None, optional): A regex pattern describing columns to omit. Defaults to None.
        """

        return GroupSelector(
            group_pattern,
            match = False,
            labels = labels,
            prefix = prefix,
            suffix = suffix,
            pattern = pattern,
            exclude_labels = exclude_labels,
            exclude_prefix = exclude_prefix,
            exclude_suffix = exclude_suffix,
            exclude_pattern = exclude_pattern,
        )

    def select_group_by_match(
        self,
        group_pattern: re.Pattern | str,
        labels: Sequence[str] | str | None = None,
        prefix: str | None = None,
        suffix: str | None = None,
        pattern: re.Pattern | str | None = None,
        exclude_labels: Sequence[str] | str | None = None,
        exclude_prefix: str | None = None,
        exclude_suffix: str | None = None,
        exclude_pattern: re.Pattern | str | None = None,
    ) -> GroupSelector:
        """Initialize a GroupSelector instance.

        Creates groups of columns that match on their first matching `group_pattern`.

        Selection parameters (e.g., `labels`, `prefix`, etc.) are used in conjunction with one another, taking the intersection of matching columns. In other words, only columns matching all selection criteria will be selected.

        Args:
            group_pattern (re.Pattern | str): A regex pattern that describes the portion of the label that indicates members of column groups.
            labels (list[str] | set[str] | str | None, optional): Full column labels to select. Defaults to None.
            prefix (str | None, optional): The prefix of columns to select. Defaults to None.
            suffix (str | None, optional): The suffix of columns to select. Defaults to None.
            pattern (str | re.Pattern | None, optional): A regex pattern describing columns to select. Defaults to None.
            exclude_labels (Sequence[str] | str | None, optional): Full column labels to omit. Defaults to None.
            exclude_prefix (str | None, optional): The prefix of columns to omit. Defaults to None.
            exclude_suffix (str | None, optional): The suffix of columns to omit. Defaults to None.
            exclude_pattern (str | re.Pattern | None, optional): A regex pattern describing columns to omit. Defaults to None.
        """

        return GroupSelector(
            group_pattern,
            match = True,
            labels = labels,
            prefix = prefix,
            suffix = suffix,
            pattern = pattern,
            exclude_labels = exclude_labels,
            exclude_prefix = exclude_prefix,
            exclude_suffix = exclude_suffix,
            exclude_pattern = exclude_pattern,
        )

    # Constants from prep.py
    PATTERN_ALIDA_OTHER_OE = prep.PATTERN_ALIDA_OTHER_OE

    # Functions from prep.py
    clean_arg = staticmethod(prep.clean_arg)
    clean_df = staticmethod(prep.clean_df)
    rename_cols = staticmethod(prep.rename_cols)
    drop_cols = staticmethod(prep.drop_cols)
    recode_vals = staticmethod(prep.recode_vals)
    remove_verbal_anchors = staticmethod(prep.remove_verbal_anchors)
    bin = staticmethod(prep.bin)
    filter_by_bounds = staticmethod(prep.filter_by_bounds)
    filter_by_iqr = staticmethod(prep.filter_by_iqr)
    filter_by_stdev = staticmethod(prep.filter_by_stdev)
    filter_straightliners = staticmethod(prep.filter_straightliners)
    dummy_to_categorical = staticmethod(prep.dummy_to_categorical)
    
    # Functions from stats.py
    agg_cols = staticmethod(stats.agg_cols)
    agg_rows = staticmethod(stats.agg_rows)
    calc_ci = staticmethod(stats.calc_ci)

    # Functions from test.py
    test_one_sample = staticmethod(test.test_one_sample)
    test_one_sample_proportion = staticmethod(test.test_one_sample_proportion)
    test_independent = staticmethod(test.test_independent)
    test_independent_proportion = staticmethod(test.test_independent_proportion)
    test_dependent = staticmethod(test.test_dependent)
    test_dependent_proportion = staticmethod(test.test_dependent_proportion)
    test_regression = staticmethod(test.test_regression)
    p_correct = staticmethod(test.p_correct)

    # Functions from reduction.py
    reduce_pca = staticmethod(reduction.reduce_pca)