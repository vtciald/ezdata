from __future__ import annotations
from typing import cast
from abc import ABC, abstractmethod
import re
import pandas as pd
from itertools import combinations

class Selector(ABC):
    """Interface for all column-selection strategies."""

    @abstractmethod
    def __call__(
        self,
        df: pd.DataFrame,           
    ) -> list[str] | list[tuple[str, str]]:
        """Must evaluate against a DataFrame and return the appropriate format."""
        pass

    @staticmethod
    def resolve_pair(
        df: pd.DataFrame,
        selection: list[str] | set[str] | tuple[str, str] | list[tuple[str, str]] | set[tuple[str, str]] | 'ColumnSelector' | 'PairSelector' | None,
    ) -> list[tuple[str, str]]:
        """Resolve column-pair selection.

        Args:
            df (pd.DataFrame): The DataFrame.
            selection (list[str] | set[str] | | tuple[str, str] | list[tuple[str, str]] | set[tuple[str, str]] | ColumnSelector | PairSelector | None): String column label(s) or a Selector.

        Returns:
            list[str]: A list of string column labels.
        """

        cols = None

        if isinstance(selection, PairSelector):
            return selection(df)
        
        elif isinstance(selection, ColumnSelector):
            cols = selection(df)

        elif selection is None:
            cols = df.columns.tolist()
    
        elif isinstance(selection, tuple):
            return cast(list[tuple[str, str]], [selection])

        elif isinstance(selection, set):
            cols = list(selection)

        elif isinstance(selection, list):
            cols = selection
            
            if isinstance(cols[0], tuple):
                return cast(list[tuple[str, str]], cols)

        if cols is None:
            raise TypeError(
                f'Invalid argument for column-pair selection: \'{selection}\'.'
            )

        return cast(list[tuple[str, str]], list(combinations(cols, 2)))

    @staticmethod
    def resolve(
        df: pd.DataFrame,
        selection: list[str] | set[str] | str | 'ColumnSelector' | None,
    ) -> list[str]:
        """Resolve column selection.

        Args:
            df (pd.DataFrame): The DataFrame.
            selection (list[str] | set[str] | str | ColumnSelector | None): String column label(s) or a Selector.

        Returns:
            list[str]: A list of string column labels.
        """
            
        if isinstance(selection, Selector):
            return selection(df)
        
        if isinstance(selection, str):
            return [selection]
        
        if isinstance(selection, (list, set)):
            return list(selection)
        
        if selection is None:
            return df.columns.tolist()

class ColumnSelector(Selector):
    """Create a ColumnSelector object to select columns.
    """

    def __init__(
        self,
        *,
        labels: list[str] | set[str] | str | None = None,
        prefix: str | None = None,
        suffix: str | None = None,
        pattern: re.Pattern | str | None = None,
    ) -> None:
        """Initialize a ColumnSelector instance.

        Selection parameters (e.g., `labels`, `prefix`, etc.) are used in conjunction with one another, taking the intersection of matching columns. In other words, only columns matching all selection criteria will be selected.

        Args:
            labels (list[str] | set[str] | str | None, optional): Full column labels to select. Defaults to None.
            prefix (str | None, optional): The prefix of columns to select. Defaults to None.
            suffix (str | None, optional): The suffix of columns to select. Defaults to None.
            pattern (str | re.Pattern | None, optional): A regex pattern describing columns to select. Defaults to None.

        Note:
            If all selection arguments are None, all columns will be selected.
        """

        self.prefix = prefix
        self.suffix = suffix
        
        # Set self.labels
        if isinstance(labels, (list, set)):
            self.labels = set(labels)

        elif isinstance(labels, str):
            self.labels = set([labels])

        else:
            self.labels = labels

        # Set self.pattern
        if isinstance(pattern, str):
            self.pattern = re.compile(pattern)

        else:
            self.pattern = pattern
    
    def __call__(
        self,
        df: pd.DataFrame,
    ) -> list[str]:
        """Resolve column selection from Selector.

        Args:
            df (pd.DataFrame): The DataFrame.

        Returns:
            list[str]: The list of selected column labels.
        """
        
        all_cols = df.columns.tolist()
        
        if self.labels is None and self.prefix is None and self.suffix is None and self.pattern is None:
            return all_cols
        
        matched_cols = [
            col for col in all_cols
            if (self.labels is None or col in self.labels)
            and (self.prefix is None or col.startswith(self.prefix))
            and (self.suffix is None or col.endswith(self.suffix))
            and (self.pattern is None or re.search(self.pattern, col))
        ]

        return matched_cols
    
class PairSelector(Selector):
    """Create a PairSelector object to select pairs of columns.
    """

    def __init__(
        self,
        pair_pattern: re.Pattern | str,
        *,
        labels: list[str] | set[str] | str | None = None,
        prefix: str | None = None,
        suffix: str | None = None,
        pattern: re.Pattern | str | None = None,
    ) -> None:
        """Initialize a PairSelector instance.

        Selection parameters (e.g., `labels`, `prefix`, etc.) are used in conjunction with one another, taking the intersection of matching columns. In other words, only columns matching all selection criteria will be selected.

        Args:
            pair_pattern (re.Pattern | str): A regex pattern that describes the portion of the label that differentiates paired columns.
            labels (list[str] | set[str] | str | None, optional): Full column labels to select. Defaults to None.
            prefix (str | None, optional): The prefix of columns to select. Defaults to None.
            suffix (str | None, optional): The suffix of columns to select. Defaults to None.
            pattern (str | re.Pattern | None, optional): A regex pattern describing columns to select. Defaults to None.

        Note:
            If all selection arguments are None, all columns will be selected.
        """

        self.prefix = prefix
        self.suffix = suffix
        
        # Set self.labels
        if isinstance(labels, (list, set)):
            self.labels = set(labels)

        elif isinstance(labels, str):
            self.labels = set([labels])

        else:
            self.labels = labels

        # Set self.pattern
        if isinstance(pattern, str):
            self.pattern = re.compile(pattern)

        else:
            self.pattern = pattern

        # Set self.pair_pattern
        if isinstance(pair_pattern, str):
            self.pair_pattern = re.compile(pair_pattern)

        else:
            self.pair_pattern = pair_pattern
    
    def __call__(
        self,
        df: pd.DataFrame,
    ) -> list[tuple[str, str]]:
        """Resolve selection.

        Args:
            df (pd.DataFrame): The DataFrame.

        Returns:
            list[tuple[str, str]]: The list of column label pairs.
        """
        
        all_cols = df.columns.tolist()
        
        pair_dict = {}
        
        for col in all_cols:
            if ((self.labels is None or col in self.labels)
            and (self.prefix is None or col.startswith(self.prefix))
            and (self.suffix is None or col.endswith(self.suffix))
            and (self.pattern is None or re.search(self.pattern, col))
            and (re.search(self.pair_pattern, col))):
                key = re.sub(self.pair_pattern, '', col)

                if key in pair_dict:
                    pair_dict[key].append(col)

                else:
                    pair_dict[key] = [col]
            
        column_pairs = []
        for cols in pair_dict.values():
            if len(cols) > 1:
                pairs = list(combinations(cols, 2))
                column_pairs.extend(pairs)

        return column_pairs