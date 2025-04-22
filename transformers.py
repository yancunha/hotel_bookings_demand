from typing import Literal, Union

import pandas as pd
import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder


class CategoricalOrdinalEncoder(
    TransformerMixin, BaseEstimator
):
    handle_unknown: Literal["error", "use_encoded_value"]

    def __init__(
        self,
        handle_unknown: Literal["error", "use_encoded_value"] = "use_encoded_value",
        unknown_value: Union[float, int, None] = np.nan,
    ):
        """Ordinal encoder that supports missing values and maintains categorical information.

        Similar to sklearn's OrdinalEncoder but with better support for missing values.

        When configured with `.set_output(transform="pandas")`, it returns a DataFrame
        with columns of dtype "category" instead of int, preserving the original category names.
        In this mode, the encoder ensures that each category is always encoded with the same value
        or converted to NaN if not seen during fitting, avoiding issues with `.astype("category")`.

        Args:
            handle_unknown (Literal["error", "use_encoded_value"]): How to handle
                categories not seen during fitting. Default: "use_encoded_value".
            unknown_value (float | int | None): Value to use for unknown categories
                when handle_unknown="use_encoded_value". Default: np.nan.
        """
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value

    def _as_safe_array(self, X: npt.ArrayLike) -> npt.NDArray:
        """Converts input to a NumPy array, treating None values as np.nan.

        Args:
            X (npt.ArrayLike): Array-like input to convert.

        Returns:
            npt.NDArray: NumPy array with None values replaced by np.nan.
        """
        X = np.asarray(X).copy()

        # None is handled incorrectly by sklearn, so replace it with nan
        X[pd.isna(X)] = np.nan

        return X
    
    def _clean_strings(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in X.columns:
            X[col] = X[col].astype(str).str.lower().str.strip()
        return X

    def fit(self, X, y=None):
        """Fits the encoder to the input data.

        Args:
            X: Array-like of shape (n_samples, n_features) with categorical data.
            y: Ignored, exists only for compatibility with sklearn.

        Returns:
            self: Returns the fitted encoder.
        """
        # Save features
        self.n_features_in_ = np.shape(X)[1]
        if isinstance(X, pd.DataFrame):
            X = self._clean_strings(X)
            self.feature_names_in_ = np.asanyarray(X.columns)

        # Initialize and fit OrdinalEncoder
        self.encoder_ = OrdinalEncoder(
            handle_unknown=self.handle_unknown, unknown_value=self.unknown_value
        )
        X = self._as_safe_array(X)
        self.encoder_.fit(X)

        # Precompute the dtypes
        self.dtypes_ = []
        for i in range(self.n_features_in_):
            cats: npt.NDArray = self.encoder_.categories_[i]  # type: ignore
            if pd.isna(cats[-1]):
                cats = cats[:-1]
            assert pd.notna(cats).all()
            self.dtypes_.append(pd.CategoricalDtype(pd.Index(cats).astype("string")))  # type: ignore

        return self

    def transform(self, X):
        """Encodes the categorical data.

        Args:
            X: Array-like of shape (n_samples, n_features) with categorical data.

        Returns:
            array-like: Encoded data. If output="pandas", returns a DataFrame with
                columns of type "category". Otherwise, returns a NumPy array with integers.
        """
        # Compute pandas index/column information
        index = X.index if isinstance(X, pd.DataFrame) else np.arange(X.shape[0])
        columns = getattr(
            self, "feature_names_in_", [f"x{i}" for i in range(self.n_features_in_)]
        )

        # Clean strings if input is a DataFrame
        if isinstance(X, pd.DataFrame):
            X = self._clean_strings(X)

        # Transform data to int
        X = self._as_safe_array(X)
        X_encoded = self.encoder_.transform(X)

        # If numpy output, return the int-encoded data
        if getattr(self, "_sklearn_output_config", {}).get("transform") != "pandas":
            return X_encoded

        # If pandas, convert to categorical
        X_encoded = np.nan_to_num(X_encoded, nan=-1, copy=False).astype(int, copy=False)
        X_cat = pd.DataFrame(index=index)
        for i, col in enumerate(columns):
            X_cat[col] = pd.Categorical.from_codes(
                X_encoded[:, i],  # type: ignore
                dtype=self.dtypes_[i],
            )

        return X_cat

    def set_output(self, *, transform=None) -> "CategoricalOrdinalEncoder":
        """Configures the output type of the transformer.

        Args:
            transform (str | None): Desired output type:
                - "default": Returns np.ndarray.
                - "pandas": Returns pd.DataFrame.
                - None: Does not modify the output type.

        Returns:
            CategoricalOrdinalEncoder: self.
        """
        if transform is None:
            return self

        if not hasattr(self, "_sklearn_output_config"):
            self._sklearn_output_config = {}

        self._sklearn_output_config["transform"] = transform
        return self

    def get_feature_names_out(self, input_features=None):
        """Returns the names of the output features.

        Args:
            input_features (list[str] | None): Names of the input features.
                If None, uses the names saved during fitting.

        Returns:
            list[str]: Names of the output features.
        """
        if input_features is not None:
            return input_features
        return self.feature_names_in_


class CompareColumnsTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to compare two columns element-wise and add a new column
    indicating whether the values in the two columns are different.
    """
    def __init__(self, col1: str, col2: str, new_col: str):
        self.col1 = col1
        self.col2 = col2
        self.new_col = new_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.new_col] = (X[self.col1] != X[self.col2]).astype(int)
        return X


class ColumnOperationTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to perform an operation (sum or multiply) on two columns
    and add the result as a new column.
    """
    def __init__(self, col1: str, col2: str, new_col: str, operation: str = "sum"):
        """
        Parameters:
        - col1: The name of the first column.
        - col2: The name of the second column.
        - new_col: The name of the new column to be created.
        - operation: The operation to perform ("sum" or "multiply" or subtraction).
        """
        self.col1 = col1
        self.col2 = col2
        self.new_col = new_col
        self.operation = operation

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.operation == "sum":
            X[self.new_col] = X[self.col1] + X[self.col2]
        elif self.operation == "multiply":
            X[self.new_col] = X[self.col1] * X[self.col2]
        elif self.operation == "subtraction":
            X[self.new_col] = X[self.col1] - X[self.col2]
        else:
            raise ValueError(f"Unsupported operation: {self.operation}. Use 'sum' or 'multiply' or subtraction .")
        return X


class SetNegativeToZeroTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to set all negative values in a specified column to 0.
    """
    def __init__(self, col: str):
        self.col = col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.col] = X[self.col].apply(lambda x: max(x, 0))
        return X[[self.col]].to_numpy()
    

class LeadTimeBucketTransformer(BaseEstimator, TransformerMixin):
    """
    Categorize lead_time into buckets (e.g., short, medium, long).
    """
    def __init__(self, bins=None, labels=None):
        self.bins = bins or [0, 15, 45, 90, 120, np.inf]  # Default bins: 0-15, 15-45, 45-90, 90-120, 120+
        self.labels = labels or ['Short', 'Medium', 'Long', 'Very Long', 'Very Very Long']

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['lead_time_bucket'] = pd.cut(X['lead_time'], bins=self.bins, labels=self.labels, right=False)
        return X
    

# Transformer 2: Add is_vacation feature
class VacationFlagTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vacation_periods=None):
        self.vacation_periods = vacation_periods or [('06-01', '08-31'), ('12-15', '01-15')]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['arrival_date'] = pd.to_datetime(X['arrival_date'])
        X['is_vacation'] = X['arrival_date'].apply(self._is_vacation)
        return X

    def _is_vacation(self, date):
        for start, end in self.vacation_periods:
            start_date = pd.to_datetime(f"{date.year}-{start}")
            end_date = pd.to_datetime(f"{date.year}-{end}") if start < end else pd.to_datetime(f"{date.year + 1}-{end}")
            if start_date <= date <= end_date:
                return 1
        return 0


class HasValueAboveZeroTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to create a binary column indicating if a numerical feature's value is above zero.
    """
    def __init__(self, columns: list):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[f"has_{col}"] = (X[col] > 0).astype(int)
        return X


class HasSpecificValueTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to create a binary column indicating if an object feature matches a specific condition.
    """
    def __init__(self, columns: list, condition: str = "none"):
        self.columns = columns
        self.condition = condition

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[f"has_{col}"] = (X[col].str.lower() != self.condition).astype(int)
        return X


class CreateNewColumnsTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to create new columns and append them to the DataFrame.
    """
    def __init__(self, transformers):
        self.transformers = transformers

    def fit(self, X, y=None):
        for name, transformer in self.transformers:
            transformer.fit(X, y)
        return self

    def transform(self, X):
        X = X.copy()
        for name, transformer in self.transformers:
            transformed = transformer.transform(X)
            if isinstance(transformed, pd.DataFrame):
                X = pd.concat([X, transformed], axis=1)
            else:
                X[name] = transformed
        return X