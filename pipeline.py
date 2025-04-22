# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from transformers import (
    VacationFlagTransformer, 
    LeadTimeBucketTransformer,
    CompareColumnsTransformer,
    ColumnOperationTransformer,
    CategoricalOrdinalEncoder,
    HasValueAboveZeroTransformer,
    HasSpecificValueTransformer,
    CreateNewColumnsTransformer
)
from sklearn import set_config
# Enable pandas output globally
set_config(transform_output="pandas")


def set_negatives_to_zero_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sets all negative values in the DataFrame to zero.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing numerical values.

    Returns:
        pd.DataFrame: DataFrame with all negative values replaced by zero.
    """
    return df.clip(lower=0)

def remove_duplicates(X: pd.DataFrame) -> pd.DataFrame:
    """
    Removes duplicate rows from the DataFrame.

    Parameters:
        X (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with duplicate rows removed.
    """
    return X.drop_duplicates()

def extract_date_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts date-related features from the 'arrival_date' column.

    Parameters:
        X (pd.DataFrame): Input DataFrame containing an 'arrival_date' column.

    Returns:
        pd.DataFrame: DataFrame with new columns:
            - 'arrival_year': Year of the arrival date.
            - 'arrival_month': Month of the arrival date.
            - 'arrival_day': Day of the arrival date.
            - 'is_weekend': Boolean indicating if the arrival date falls on a weekend.
    """
    X = X.copy()
    X["arrival_year"] = pd.to_datetime(X["arrival_date"]).dt.year
    X["arrival_month"] = pd.to_datetime(X["arrival_date"]).dt.month
    X["arrival_day"] = pd.to_datetime(X["arrival_date"]).dt.day
    X["is_weekend"] = pd.to_datetime(X["arrival_date"]).dt.dayofweek >= 5
    return X

def create_pipeline() -> Pipeline:

    initial_preprocessing =  ColumnTransformer(
        transformers = [
            (
                # Apply the set_negative_to_zero function to the average_daily_rate column
                "set_negative_to_zero_average_daily_rate", FunctionTransformer(
                    set_negatives_to_zero_vectorized,
                    feature_names_out="one-to-one"
                ), ["average_daily_rate"]
            )
        ],
        remainder="passthrough", 
        force_int_remainder_cols=False,
        verbose_feature_names_out=False 
    )

    create_new_columns = CreateNewColumnsTransformer(
        transformers=[
            ("vacation_flag", VacationFlagTransformer()),
            ("lead_time_bins", LeadTimeBucketTransformer()),
            ("numerical_binary_transformer", HasValueAboveZeroTransformer(
                columns=[
                    "previous_cancellations", 
                    "booking_changes",
                    "total_of_special_requests",
                    "days_in_waiting_list",
                ]
            )),
            ("categorical_binary_transformer", HasSpecificValueTransformer(
                columns=["required_car_parking_spaces"], condition="none"
            )),
            ("compare_columns_room", CompareColumnsTransformer(
                col1="assigned_room_type", 
                col2="reserved_room_type", 
                new_col="has_room_reassignment"
            )),
            ("sum_columns_total_nights", ColumnOperationTransformer(
                col1="stays_in_week_nights", 
                col2="stays_in_weekend_nights", 
                new_col="total_nights",
                operation="sum"
            )),
            ("multiply_columns_total_money", ColumnOperationTransformer(
                col1="total_nights", 
                col2="average_daily_rate", 
                new_col="total_money",
                operation="multiply"
            )),
       ]
    )

    categorical_encoding = ColumnTransformer(
        transformers = [
            (
                #apply categorical ordinal encoder
                "CategoricalOrdinalEncoder",
                CategoricalOrdinalEncoder().set_output(transform="pandas"),
                make_column_selector(
                    dtype_include=["string", "category", "object"]  # type:ignore
                ),
            ),
        ],
        remainder="passthrough",
        force_int_remainder_cols=False,  # type: ignore
        verbose_feature_names_out=False,
        verbose=True,
    )

    # Combine preprocessing steps
    pipeline = Pipeline(steps=[
        ("extract_date_features", FunctionTransformer(extract_date_features)),

        # Preprocess numerical and categorical features
        ("preprocessor", initial_preprocessing),

        # Create new columns
        ("vacation_flag", VacationFlagTransformer()),
        ("lead_time_bins", LeadTimeBucketTransformer()),
        ("numerical_binary_transformer", HasValueAboveZeroTransformer(
            columns=[
                "previous_cancellations", 
                "booking_changes",
                "total_of_special_requests",
                "days_in_waiting_list",
            ]
        )),
        ("categorical_binary_transformer", HasSpecificValueTransformer(
            columns=["required_car_parking_spaces"], condition="none"
        )),

        ("compare_columns_room", CompareColumnsTransformer(
            col1="assigned_room_type", 
            col2="reserved_room_type", 
            new_col="has_room_reassignment"
        )),
        ("sum_columns_total_nights", ColumnOperationTransformer(
            col1="stays_in_week_nights", 
            col2="stays_in_weekend_nights", 
            new_col="total_nights",
            operation="sum"
        )),
        ("multiply_columns_total_money", ColumnOperationTransformer(
            col1="total_nights", 
            col2="average_daily_rate", 
            new_col="total_money",
            operation="multiply"
        )),

        # Apply categorical processing
        ("categorical_transfomer", categorical_encoding)
    ])

    return pipeline