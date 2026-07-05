from __future__ import annotations
from typing import cast
from abc import ABC, abstractmethod
from collections.abc import Sequence
import re
import pandas as pd
from itertools import combinations

class Selector(ABC):
    """Interface for all column-selection strategies."""

    @abstractmethod
    def __call__(
        self,
        df: pd.DataFrame,           
    ) -> list[str] | list[list[str]]:
        """Must evaluate against a DataFrame and return the appropriate format."""
        pass

    @staticmethod
    def resolve_pair(
        df: pd.DataFrame,
        selection: Sequence[str] | Sequence[Sequence[str]] | 'ColumnSelector' | 'PairSelector' | None,
    ) -> list[list[str]]:
        """Resolve column-pair selection.

        Args:
            df (pd.DataFrame): The DataFrame.
            selection (Sequence[str] | Sequence[Sequence[str]] | ColumnSelector | PairSelector | None): String column label(s) or a Selector.

        Returns:
            list[str]: A list of string column labels.
        """
       
        if isinstance(selection, PairSelector):
            return selection(df)
        
        elif isinstance(selection, ColumnSelector):
            cols = selection(df)
    
        elif isinstance(selection, Sequence) and not isinstance(selection, str):
            if isinstance(selection[0], str):
                cols = list(selection)

            elif isinstance(selection[0], Sequence):
                return [list(item) for item in selection]
            
        elif selection is None:
            cols = df.columns.tolist()

        else:
            raise TypeError(
                f'Invalid argument for column-pair selection: \'{selection}\'.'
            )          
        
        pairs = list(combinations(cols, 2))
        column_pairs = [[str(pair[0]), str(pair[1])] for pair in pairs]
    
        return column_pairs

    @staticmethod
    def resolve(
        df: pd.DataFrame,
        selection: Sequence[str] | str | 'ColumnSelector' | None,
    ) -> list[str]:
        """Resolve column selection.

        Args:
            df (pd.DataFrame): The DataFrame.
            selection (Sequence[str] | str | ColumnSelector | None): String column label(s) or a Selector.

        Returns:
            list[str]: A list of string column labels.
        """
            
        if isinstance(selection, ColumnSelector):
            return selection(df)
        
        elif isinstance(selection, str):
            return [selection]
        
        elif isinstance(selection, Sequence):
            return list(selection)
        
        elif selection is None:
            return df.columns.tolist()
        
        else:
            raise TypeError(
                f'Invalid argument for pair selection: \'{selection}\'.'
            )
        
    def _assign_labels(
        self,
        labels: Sequence[str] | str | None,
    ) -> None:
        """Assign labels arg to attribute.

        Args:
            labels (Sequence[str] | str | None): The labels argument.

        Raises:
            TypeError: If labels is an invalid type.
        """
        
        if isinstance(labels, Sequence):
            for label in labels:
                if not isinstance(label, str):
                    raise TypeError(
                        f'Invalid type for labels argument: \'{labels}\' (\'{label}\'). '
                        'Must be a string, sequence of strings, or None.'
                    )
                
            self.labels = set(labels)

        elif isinstance(labels, str):
            self.labels = set([labels])

        elif labels is None:
            self.labels = None

        else:
            raise TypeError(
                f'Invalid type for labels argument: \'{labels}\'. '
                'Must be a string, sequence of strings, or None.'
            )
        
    def _assign_prefix(
        self,
        prefix: str | None = None,
    ) -> None:
        """Assign prefix arg to attribute.

        Args:
            prefix (str | None): The prefix argument.

        Raises:
            TypeError: If prefix is an invalid type.
        """
        
        if isinstance(prefix, str):
            self.prefix = prefix

        elif prefix is None:
            self.prefix = None

        else:
            raise TypeError(
                f'Invalid type for prefix argument: \'{prefix}\'. '
                'Must be a string or None.'
            )
        
    def _assign_suffix(
        self,
        suffix: str | None = None,
    ) -> None:
        """Assign suffix arg to attribute.

        Args:
            suffix (str | None): The suffix argument.

        Raises:
            TypeError: If suffix is an invalid type.
        """
        
        if isinstance(suffix, str):
            self.suffix = suffix

        elif suffix is None:
            self.suffix = None

        else:
            raise TypeError(
                f'Invalid type for suffix argument: \'{suffix}\'. '
                'Must be a string or None.'
            )
        
    def _assign_pattern(
        self,
        pattern: re.Pattern | str | None = None,
    ) -> None:
        """Assign pattern arg to attribute.

        Args:
            pattern (str | re.Pattern | None): The pattern argument.

        Raises:
            TypeError: If pattern is an invalid type.
        """
        
        if isinstance(pattern, str):
            self.pattern = re.compile(pattern)

        elif isinstance(pattern, re.Pattern):
            self.pattern = pattern

        elif pattern is None:
            self.pattern = None

        else:
            raise TypeError(
                f'Invalid type for pattern argument: \'{pattern}\'. '
                'Must be a string, re.Pattern, or None.'
            )
        
    def _assign_group_pattern(
        self,
        group_pattern: re.Pattern | str | None = None,
    ) -> None:
        """Assign pattern arg to attribute.

        Args:
            pattern (str | re.Pattern | None): The pattern argument.

        Raises:
            TypeError: If pattern is an invalid type.
        """
        
        if isinstance(group_pattern, str):
            self.group_pattern = re.compile(group_pattern)

        elif isinstance(group_pattern, re.Pattern):
            self.group_pattern = group_pattern

        elif group_pattern is None:
            self.group_pattern = None

        else:
            raise TypeError(
                f'Invalid type for group_pattern argument: \'{group_pattern}\'. '
                'Must be a string, re.Pattern, or None.'
            )
        
    def _get_col_groups(
        self,
        df: pd.DataFrame,
    ) -> dict[str, list[str]]:
        
        all_cols = df.columns.tolist()
        
        group_dict = {}
        
        for col in all_cols:
            if ((self.labels is None or col in self.labels)
            and (self.prefix is None or col.startswith(self.prefix))
            and (self.suffix is None or col.endswith(self.suffix))
            and (self.pattern is None or re.search(self.pattern, col))
            and (re.search(self.group_pattern, col))): # type: ignore
                key = re.sub(self.group_pattern, '', col) # type: ignore

                if key in group_dict:
                    group_dict[key].append(col)

                else:
                    group_dict[key] = [col]

        return group_dict

class ColumnSelector(Selector):
    """Create a ColumnSelector object to select columns.
    """

    def __init__(
        self,
        *,
        labels: Sequence[str] | str | None = None,
        prefix: str | None = None,
        suffix: str | None = None,
        pattern: re.Pattern | str | None = None,
    ) -> None:
        """Initialize a ColumnSelector instance.

        Selection parameters (e.g., `labels`, `prefix`, etc.) are used in conjunction with one another, taking the intersection of matching columns. In other words, only columns matching all selection criteria will be selected.

        Args:
            labels (Sequence[str] | str | None, optional): Full column labels to select. Defaults to None.
            prefix (str | None, optional): The prefix of columns to select. Defaults to None.
            suffix (str | None, optional): The suffix of columns to select. Defaults to None.
            pattern (str | re.Pattern | None, optional): A regex pattern describing columns to select. Defaults to None.

        Note:
            If all selection arguments are None, all columns will be selected.
        """

        self._assign_labels(labels)
        self._assign_prefix(prefix)
        self._assign_suffix(suffix)
        self._assign_pattern(pattern)
    
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
        group_pattern: re.Pattern | str,
        *,
        labels: Sequence[str] | str | None = None,
        prefix: str | None = None,
        suffix: str | None = None,
        pattern: re.Pattern | str | None = None,
    ) -> None:
        """Initialize a PairSelector instance.

        Selection parameters (e.g., `labels`, `prefix`, etc.) are used in conjunction with one another, taking the intersection of matching columns. In other words, only columns matching all selection criteria will be selected.

        Args:
            group_pattern (re.Pattern | str): A regex pattern that describes the portion of the label that differentiates paired columns.
            labels (Sequence[str] | str | None, optional): Full column labels to select. Defaults to None.
            prefix (str | None, optional): The prefix of columns to select. Defaults to None.
            suffix (str | None, optional): The suffix of columns to select. Defaults to None.
            pattern (str | re.Pattern | None, optional): A regex pattern describing columns to select. Defaults to None.

        Note:
            If all selection arguments are None, all columns will be selected.
        """

        self._assign_labels(labels)
        self._assign_prefix(prefix)
        self._assign_suffix(suffix)
        self._assign_pattern(pattern)
        self._assign_group_pattern(group_pattern)
    
    def __call__(
        self,
        df: pd.DataFrame,
    ) -> list[list[str]]:
        """Resolve selection.

        Args:
            df (pd.DataFrame): The DataFrame.

        Returns:
            list[list[str]]: The list of column label pairs.
        """

        group_dict = self._get_col_groups(df)
            
        column_pairs = []
        for cols in group_dict.values():
            if len(cols) > 1:
                pairs = list(combinations(cols, 2))

                for pair in pairs:
                    column_pairs.append([pair[0], pair[1]])

        return column_pairs