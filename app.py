# DataTalk Streamlit App
# Rebuilt by integrating the backend logic from Finalproject(1).ipynb
# with the Streamlit interface from app(1).py.
#
# IMPORTANT:
# Keep GROQ_API_KEY in .streamlit/secrets.toml:
# GROQ_API_KEY = "your_key_here"
#
import os
import json
import html
import re
from io import BytesIO

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from groq import Groq

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from xgboost import XGBClassifier, XGBRegressor



# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DataTalk",
    page_icon="💬",
    layout="wide"
)


# ============================================================
# UI / DOWNLOAD / THEME HELPERS
# ============================================================

def figure_to_png_bytes(figure):
    buffer = BytesIO()
    figure.savefig(
        buffer,
        format="png",
        dpi=180,
        bbox_inches="tight"
    )
    buffer.seek(0)
    return buffer.getvalue()


def download_chart_button(
    figure,
    file_name,
    key
):
    st.download_button(
        label="⬇️ Download chart",
        data=figure_to_png_bytes(figure),
        file_name=file_name,
        mime="image/png",
        key=key
    )


def chart_file_name(plan):
    operation = plan.get("operation", "chart")
    x_column = plan.get("x_column")
    y_column = plan.get("y_column")

    parts = ["datatalk", str(operation)]

    if x_column:
        parts.append(str(x_column))

    if y_column:
        parts.append(str(y_column))

    safe_name = "_".join(parts)
    safe_name = "".join(
        character if character.isalnum() or character in {"_", "-"}
        else "_"
        for character in safe_name
    )

    return safe_name + ".png"


def apply_selected_theme():
    theme = st.session_state.get(
        "theme_choice",
        "Auto"
    )

    if theme == "Light":
        st.markdown(
            """
            <style>
            .stApp {
                background-color: #ffffff;
                color: #111827;
            }
            [data-testid="stSidebar"] {
                background-color: #f4f6f8;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

    elif theme == "Dark":
        st.markdown(
            """
            <style>
            .stApp {
                background-color: #111827;
                color: #f9fafb;
            }
            [data-testid="stSidebar"] {
                background-color: #0f172a;
            }
            </style>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# DATA CLEANING FUNCTIONS
# ============================================================

def detect_identifier_columns(data):
    identifier_columns = []

    for column in data.columns:
        column_name = str(column).lower()

        if (
            column_name == "id"
            or column_name.endswith("_id")
            or column_name.startswith("id_")
            or column_name.endswith("_identifier")
        ):
            identifier_columns.append(column)

    return identifier_columns


def standardize_column_names(data):
    cleaned_data = data.copy()

    cleaned_data.columns = (
        cleaned_data.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-zA-Z0-9_]+", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )

    if cleaned_data.columns.duplicated().any():
        raise ValueError(
            "Duplicate column names found after standardization."
        )

    return cleaned_data


def remove_duplicate_rows(data):
    cleaned_data = data.copy()
    duplicate_count = int(cleaned_data.duplicated().sum())

    cleaned_data = (
        cleaned_data
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return cleaned_data, duplicate_count


def check_missing_values(data):
    if len(data) == 0:
        return pd.DataFrame(
            columns=[
                "Column",
                "Missing Values",
                "Missing Percentage"
            ]
        )

    missing_count = data.isnull().sum()

    missing_percentage = (
        missing_count / len(data) * 100
    ).round(2)

    missing_report = pd.DataFrame(
        {
            "Column": data.columns,
            "Missing Values": missing_count.values,
            "Missing Percentage": missing_percentage.values
        }
    )

    return (
        missing_report
        .sort_values(
            by="Missing Values",
            ascending=False
        )
        .reset_index(drop=True)
    )


def replace_missing_placeholders(data):
    cleaned_data = data.copy()

    missing_placeholders = [
        "?",
        "NA",
        "N/A",
        "na",
        "n/a",
        "null",
        "NULL",
        "None",
        "none",
        "",
        " "
    ]

    cleaned_data = cleaned_data.replace(
        missing_placeholders,
        np.nan
    )

    return cleaned_data


def replace_zero_with_missing(
    data,
    columns
):
    """
    Optional domain-specific cleaning.

    Zero is NOT treated as missing globally.
    It is replaced with NaN only in columns explicitly selected by the user.
    """
    cleaned_data = data.copy()

    for column in columns:
        if column not in cleaned_data.columns:
            continue

        if not pd.api.types.is_numeric_dtype(
            cleaned_data[column]
        ):
            continue

        cleaned_data[column] = (
            cleaned_data[column]
            .replace(0, np.nan)
        )

    return cleaned_data


def fill_missing_values(data):
    cleaned_data = data.copy()

    numerical_columns = (
        cleaned_data
        .select_dtypes(include="number")
        .columns
        .tolist()
    )

    categorical_columns = (
        cleaned_data
        .select_dtypes(
            include=[
                "object",
                "category",
                "string",
                "bool"
            ]
        )
        .columns
        .tolist()
    )

    for column in numerical_columns:
        if cleaned_data[column].isnull().any():
            median_value = cleaned_data[column].median()

            if pd.notna(median_value):
                cleaned_data[column] = (
                    cleaned_data[column]
                    .fillna(median_value)
                )

    for column in categorical_columns:
        if cleaned_data[column].isnull().any():
            mode_values = cleaned_data[column].mode()

            if not mode_values.empty:
                cleaned_data[column] = (
                    cleaned_data[column]
                    .fillna(mode_values.iloc[0])
                )
            else:
                cleaned_data[column] = (
                    cleaned_data[column]
                    .fillna("Unknown")
                )

    return cleaned_data


def detect_date_columns(data):
    cleaned_data = data.copy()
    date_columns = []

    for column in cleaned_data.columns:
        if pd.api.types.is_datetime64_any_dtype(
            cleaned_data[column]
        ):
            date_columns.append(column)
            continue

        column_name = str(column).lower()

        looks_like_date = (
            "date" in column_name
            or "datetime" in column_name
            or "timestamp" in column_name
            or column_name.endswith("_time")
        )

        if not looks_like_date:
            continue

        original_non_null = int(
            cleaned_data[column].notnull().sum()
        )

        if original_non_null == 0:
            continue

        converted = pd.to_datetime(
            cleaned_data[column],
            errors="coerce"
        )

        valid = int(converted.notnull().sum())

        # Require at least 80% of the existing values to parse.
        if valid / original_non_null >= 0.80:
            cleaned_data[column] = converted
            date_columns.append(column)

    return cleaned_data, date_columns


def detect_outliers(data):
    numerical_columns = (
        data
        .select_dtypes(include="number")
        .columns
        .tolist()
    )

    identifier_columns = set(
        detect_identifier_columns(data)
    )

    outlier_report = []

    for column in numerical_columns:
        if column in identifier_columns:
            continue

        values = data[column].dropna()

        if values.empty:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        lower_limit = q1 - (1.5 * iqr)
        upper_limit = q3 + (1.5 * iqr)

        outliers = values[
            (values < lower_limit)
            |
            (values > upper_limit)
        ]

        outlier_report.append(
            {
                "Column": column,
                "Outlier Count": int(len(outliers)),
                "Lower Limit": round(
                    float(lower_limit),
                    4
                ),
                "Upper Limit": round(
                    float(upper_limit),
                    4
                )
            }
        )

    return pd.DataFrame(outlier_report)


def cap_outliers(data):
    cleaned_data = data.copy()

    numerical_columns = (
        cleaned_data
        .select_dtypes(include="number")
        .columns
        .tolist()
    )

    identifier_columns = set(
        detect_identifier_columns(cleaned_data)
    )

    for column in numerical_columns:
        if column in identifier_columns:
            continue

        values = cleaned_data[column].dropna()

        if values.empty:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        lower_limit = q1 - (1.5 * iqr)
        upper_limit = q3 + (1.5 * iqr)

        cleaned_data[column] = (
            cleaned_data[column]
            .clip(
                lower=lower_limit,
                upper=upper_limit
            )
        )

    return cleaned_data


def clean_dataset(
    data,
    zero_as_missing_columns=None,
    cap_outlier_values=False
):
    cleaned_data = data.copy()

    zero_as_missing_columns = (
        zero_as_missing_columns or []
    )

    original_rows = cleaned_data.shape[0]
    original_columns = cleaned_data.shape[1]

    # Step 1: replace common placeholder strings
    cleaned_data = replace_missing_placeholders(
        cleaned_data
    )

    # Step 2: standardize column names
    cleaned_data = standardize_column_names(
        cleaned_data
    )

    # Convert selected original column names to the standardized names
    selected_zero_columns = []

    for column in zero_as_missing_columns:
        standardized = (
            str(column)
            .strip()
            .lower()
        )

        standardized = (
            pd.Series([standardized])
            .str.replace(
                r"[^a-zA-Z0-9_]+",
                "_",
                regex=True
            )
            .str.replace(
                r"_+",
                "_",
                regex=True
            )
            .str.strip("_")
            .iloc[0]
        )

        if standardized in cleaned_data.columns:
            selected_zero_columns.append(
                standardized
            )

    # Step 3: optional zero-as-missing handling
    cleaned_data = replace_zero_with_missing(
        cleaned_data,
        selected_zero_columns
    )

    missing_before_count = int(
        cleaned_data
        .isnull()
        .sum()
        .sum()
    )

    # Step 4: remove duplicate rows
    cleaned_data, duplicate_count = (
        remove_duplicate_rows(
            cleaned_data
        )
    )

    # Missing report before filling
    missing_before = check_missing_values(
        cleaned_data
    )

    # Step 5: detect date columns
    cleaned_data, date_columns = (
        detect_date_columns(
            cleaned_data
        )
    )

    # Step 6: fill missing values
    cleaned_data = fill_missing_values(
        cleaned_data
    )

    # Step 7: detect IQR outliers
    outliers_before = detect_outliers(
        cleaned_data
    )

    # Step 8: optional outlier capping
    if cap_outlier_values:
        cleaned_data = cap_outliers(
            cleaned_data
        )

    remaining_missing = int(
        cleaned_data
        .isnull()
        .sum()
        .sum()
    )

    cleaning_summary = {
        "Original Rows": original_rows,
        "Original Columns": original_columns,
        "Duplicate Rows Removed": duplicate_count,
        "Missing Before Filling": missing_before_count,
        "Remaining Missing Values": remaining_missing,
        "Detected Date Columns": (
            ", ".join(date_columns)
            if date_columns
            else "None"
        ),
        "Zero-as-Missing Columns": (
            ", ".join(selected_zero_columns)
            if selected_zero_columns
            else "None"
        ),
        "Outliers Capped": (
            "Yes"
            if cap_outlier_values
            else "No"
        )
    }

    return (
        cleaned_data,
        cleaning_summary,
        missing_before,
        outliers_before
    )


# ============================================================
# EDA FUNCTIONS
# ============================================================

def basic_statistical_summary(data):
    numerical_data = (
        data
        .select_dtypes(include="number")
    )

    if numerical_data.empty:
        return None

    return numerical_data.describe().T


def categorical_summary(data):
    categorical_columns = (
        data
        .select_dtypes(
            include=[
                "object",
                "category",
                "string",
                "bool"
            ]
        )
        .columns
        .tolist()
    )

    if not categorical_columns:
        return None

    summary_rows = []

    for column in categorical_columns:
        mode_values = data[column].mode()

        value_counts = (
            data[column]
            .value_counts(dropna=True)
        )

        summary_rows.append(
            {
                "Column": column,
                "Unique Values": int(
                    data[column].nunique(
                        dropna=True
                    )
                ),
                "Most Frequent Value": (
                    mode_values.iloc[0]
                    if not mode_values.empty
                    else None
                ),
                "Most Frequent Count": (
                    int(value_counts.iloc[0])
                    if not value_counts.empty
                    else 0
                )
            }
        )

    return pd.DataFrame(summary_rows)


def correlation_analysis(data):
    numerical_data = (
        data
        .select_dtypes(include="number")
    )

    if numerical_data.shape[1] < 2:
        return None

    return numerical_data.corr()


# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def create_histogram(data, column):
    _require_column(data, column)
    _require_numeric(data, column)

    fig, ax = plt.subplots(figsize=(6.8, 3.8))

    sns.histplot(
        data[column].dropna(),
        bins=20,
        kde=True,
        ax=ax
    )

    ax.set_title(
        f"Distribution of {column}"
    )
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")

    fig.tight_layout()

    return fig


def create_scatter(data, x_column, y_column):
    _require_column(data, x_column)
    _require_column(data, y_column)
    _require_numeric(data, x_column)
    _require_numeric(data, y_column)

    fig, ax = plt.subplots(figsize=(6.8, 3.8))

    sns.scatterplot(
        data=data,
        x=x_column,
        y=y_column,
        s=30,
        alpha=0.75,
        ax=ax
    )

    ax.set_title(
        f"{y_column} vs {x_column}"
    )

    fig.tight_layout()

    return fig


def create_bar_chart(
    data,
    x_column,
    y_column,
    aggregation="mean"
):
    _require_column(data, x_column)
    _require_column(data, y_column)
    _require_numeric(data, y_column)

    if aggregation not in ["mean", "sum"]:
        aggregation = "mean"

    if aggregation == "sum":
        grouped = (
            data
            .groupby(
                x_column,
                dropna=False
            )[y_column]
            .sum()
            .reset_index()
        )
    else:
        grouped = (
            data
            .groupby(
                x_column,
                dropna=False
            )[y_column]
            .mean()
            .reset_index()
        )

    # Keep very high-cardinality charts readable.
    grouped = grouped.head(20)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    sns.barplot(
        data=grouped,
        x=x_column,
        y=y_column,
        ax=ax
    )

    ax.set_title(
        f"{aggregation.title()} {y_column} by {x_column}"
    )

    ax.tick_params(
        axis="x",
        rotation=45
    )

    fig.tight_layout()

    return fig


def create_line_chart(
    data,
    date_column,
    value_column,
    aggregation="sum",
    time_grouping="month"
):
    _require_column(data, date_column)
    _require_column(data, value_column)
    _require_numeric(data, value_column)

    working = data[
        [date_column, value_column]
    ].copy()

    if not pd.api.types.is_datetime64_any_dtype(
        working[date_column]
    ):
        working[date_column] = pd.to_datetime(
            working[date_column],
            errors="coerce"
        )

    working = working.dropna()

    if working.empty:
        raise ValueError(
            f"'{date_column}' could not be used as a date column."
        )

    if time_grouping == "day":
        working["_period"] = (
            working[date_column]
            .dt.to_period("D")
            .dt.to_timestamp()
        )
    elif time_grouping == "year":
        working["_period"] = (
            working[date_column]
            .dt.to_period("Y")
            .dt.to_timestamp()
        )
    else:
        working["_period"] = (
            working[date_column]
            .dt.to_period("M")
            .dt.to_timestamp()
        )

    if aggregation == "mean":
        grouped = (
            working
            .groupby("_period")[value_column]
            .mean()
            .reset_index()
        )
    else:
        grouped = (
            working
            .groupby("_period")[value_column]
            .sum()
            .reset_index()
        )

    fig, ax = plt.subplots(figsize=(7.2, 4.0))

    sns.lineplot(
        data=grouped,
        x="_period",
        y=value_column,
        marker="o",
        ax=ax
    )

    ax.set_title(
        f"{aggregation.title()} {value_column} by {time_grouping}"
    )
    ax.set_xlabel(date_column)
    ax.set_ylabel(value_column)
    ax.tick_params(
        axis="x",
        rotation=45
    )

    fig.tight_layout()

    return fig


def create_boxplot(data, column):
    _require_column(data, column)
    _require_numeric(data, column)

    fig, ax = plt.subplots(figsize=(6.8, 3.0))

    sns.boxplot(
        x=data[column].dropna(),
        ax=ax
    )

    ax.set_title(
        f"Box Plot of {column}"
    )
    ax.set_xlabel(column)

    fig.tight_layout()

    return fig


def create_pie_chart(data, column):
    _require_column(data, column)

    value_counts = (
        data[column]
        .fillna("Missing")
        .astype(str)
        .value_counts()
    )

    if value_counts.empty:
        raise ValueError(
            f"Column '{column}' has no values to plot."
        )

    if len(value_counts) > 10:
        value_counts = value_counts.head(10)

    fig, ax = plt.subplots(figsize=(5.8, 5.2))

    ax.pie(
        value_counts.values,
        labels=value_counts.index,
        autopct="%1.1f%%"
    )

    ax.set_title(
        f"Distribution of {column}"
    )

    fig.tight_layout()

    return fig


def create_count_chart(data, column):
    _require_column(data, column)

    counts = (
        data[column]
        .fillna("Missing")
        .astype(str)
        .value_counts()
        .head(20)
        .reset_index()
    )

    counts.columns = [
        column,
        "count"
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    sns.barplot(
        data=counts,
        x=column,
        y="count",
        ax=ax
    )

    ax.set_title(
        f"Count of {column}"
    )
    ax.set_ylabel("Count")
    ax.tick_params(
        axis="x",
        rotation=45
    )

    fig.tight_layout()

    return fig


def create_correlation_heatmap(data):
    correlation_matrix = (
        correlation_analysis(data)
    )

    if correlation_matrix is None:
        raise ValueError(
            "At least two numerical columns are required "
            "for a correlation heatmap."
        )

    fig, ax = plt.subplots(figsize=(7.3, 5.0))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        annot_kws={
            "size": 8
        },
        ax=ax
    )

    ax.set_title(
        "Correlation Heatmap"
    )

    fig.tight_layout()

    return fig


def _require_column(data, column):
    if column is None:
        raise ValueError(
            "A required column was not provided."
        )

    if column not in data.columns:
        raise ValueError(
            f"Column '{column}' does not exist."
        )


def _require_numeric(data, column):
    if not pd.api.types.is_numeric_dtype(
        data[column]
    ):
        raise ValueError(
            f"Column '{column}' must be numerical."
        )


# ============================================================
# MACHINE LEARNING FUNCTIONS
# Integrated from Finalproject.ipynb
#
# Notebook model families:
# Classification:
#   1. Logistic Regression
#   2. Decision Tree
#   3. Random Forest
#   4. K-Nearest Neighbors
#   5. XGBoost
#
# Regression:
#   1. Linear Regression
#   2. Decision Tree Regressor
#   3. Random Forest Regressor
#   4. K-Nearest Neighbors Regressor
#   5. XGBoost Regressor
# ============================================================


def infer_ml_task(target_series):
    """
    Follow the notebook's automatic ML problem-type logic:
    - non-numeric target -> classification
    - numeric target with <= 10 unique values -> classification
    - continuous numeric target -> regression
    """
    non_null = target_series.dropna()

    if non_null.empty:
        raise ValueError(
            "The target column contains no usable values."
        )

    if not pd.api.types.is_numeric_dtype(non_null):
        return "classification"

    if int(non_null.nunique()) <= 10:
        return "classification"

    return "regression"


def get_classification_models():
    """
    Exactly the five classification models used in Finalproject.ipynb.
    """
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000
        ),
        "decision_tree": DecisionTreeClassifier(
            random_state=42
        ),
        "random_forest": RandomForestClassifier(
            random_state=42
        ),
        "knn": KNeighborsClassifier(),
        "xgboost": XGBClassifier(
            random_state=42,
            eval_metric="logloss"
        )
    }


def get_regression_models():
    """
    Exactly the five regression models used in Finalproject.ipynb.
    """
    return {
        "linear_regression": LinearRegression(),
        "decision_tree_regressor": DecisionTreeRegressor(
            random_state=42
        ),
        "random_forest_regressor": RandomForestRegressor(
            random_state=42
        ),
        "knn_regressor": KNeighborsRegressor(),
        "xgboost_regressor": XGBRegressor(
            random_state=42
        )
    }


def prepare_ml_data(
    data,
    target_column,
    feature_columns=None
):
    """
    Adaptation of the notebook's prepare_ml_data() for Streamlit.

    The notebook:
    - removes obvious identifier columns,
    - separates X and y,
    - one-hot encodes categorical features.

    The Streamlit version also allows optional feature selection.
    """
    if target_column not in data.columns:
        raise ValueError(
            f"Target column '{target_column}' does not exist."
        )

    ml_data = data.copy()

    # Supervised learning cannot use rows with a missing target.
    ml_data = ml_data.dropna(
        subset=[target_column]
    )

    if ml_data.empty:
        raise ValueError(
            "No rows remain after removing missing target values."
        )

    identifier_columns = set(
        detect_identifier_columns(ml_data)
    )

    if feature_columns:
        invalid_features = [
            column
            for column in feature_columns
            if column not in ml_data.columns
        ]

        if invalid_features:
            raise ValueError(
                "Unknown feature columns: "
                + ", ".join(invalid_features)
            )

        selected_features = [
            column
            for column in feature_columns
            if (
                column != target_column
                and column not in identifier_columns
            )
        ]
    else:
        selected_features = [
            column
            for column in ml_data.columns
            if (
                column != target_column
                and column not in identifier_columns
            )
        ]

    if not selected_features:
        raise ValueError(
            "No usable feature columns remain after removing "
            "the target and identifier columns."
        )

    X = ml_data[selected_features].copy()
    y = ml_data[target_column].copy()

    # Datetime predictors are converted into numerical timestamps.
    for column in X.columns:
        if pd.api.types.is_datetime64_any_dtype(X[column]):
            X[column] = (
                X[column]
                .astype("int64")
                .replace(
                    -9223372036854775808,
                    np.nan
                )
            )

    categorical_columns = (
        X
        .select_dtypes(
            include=[
                "object",
                "category",
                "string",
                "bool"
            ]
        )
        .columns
        .tolist()
    )

    X = pd.get_dummies(
        X,
        columns=categorical_columns,
        drop_first=True
    )

    # Defensive numeric conversion / median fill for ML robustness.
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    for column in X.columns:
        if not pd.api.types.is_numeric_dtype(X[column]):
            X[column] = pd.to_numeric(
                X[column],
                errors="coerce"
            )

        if X[column].isnull().any():
            median_value = X[column].median()

            if pd.notna(median_value):
                X[column] = X[column].fillna(
                    median_value
                )
            else:
                X[column] = X[column].fillna(0)

    return X, y, selected_features


def _classification_split_and_scale(
    data,
    target_column,
    feature_columns=None
):
    """
    Shared preparation matching the notebook classification pipeline.
    """
    X, y, selected_features = prepare_ml_data(
        data,
        target_column,
        feature_columns
    )

    label_encoder = None

    if not pd.api.types.is_numeric_dtype(y):
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

    y_series = pd.Series(y)

    if y_series.nunique() < 2:
        raise ValueError(
            "Classification requires at least two target classes."
        )

    class_counts = y_series.value_counts()

    stratify_value = (
        y
        if class_counts.min() >= 2
        else None
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=stratify_value
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "selected_features": selected_features,
        "label_encoder": label_encoder
    }


def train_classification_model(
    data,
    target_column,
    model_name="random_forest",
    feature_columns=None
):
    """
    Train one of the five classification models from the notebook.
    """
    prepared = _classification_split_and_scale(
        data,
        target_column,
        feature_columns
    )

    models = get_classification_models()

    if model_name == "auto":
        model_name = "random_forest"

    if model_name not in models:
        raise ValueError(
            "Unknown classification model. Available models: "
            + ", ".join(models.keys())
        )

    model = models[model_name]

    if model_name in {
        "logistic_regression",
        "knn"
    }:
        train_X = prepared["X_train_scaled"]
        test_X = prepared["X_test_scaled"]
    else:
        train_X = prepared["X_train"]
        test_X = prepared["X_test"]

    model.fit(
        train_X,
        prepared["y_train"]
    )

    predictions = model.predict(
        test_X
    )

    accuracy = accuracy_score(
        prepared["y_test"],
        predictions
    )

    precision = precision_score(
        prepared["y_test"],
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        prepared["y_test"],
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        prepared["y_test"],
        predictions,
        average="weighted",
        zero_division=0
    )

    metrics_table = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Precision (weighted)",
                "Recall (weighted)",
                "F1 Score (weighted)"
            ],
            "Value": [
                accuracy,
                precision,
                recall,
                f1
            ]
        }
    )

    labels = np.unique(
        np.concatenate(
            [
                np.asarray(prepared["y_test"]),
                np.asarray(predictions)
            ]
        )
    )

    cm = confusion_matrix(
        prepared["y_test"],
        predictions,
        labels=labels
    )

    confusion_table = pd.DataFrame(
        cm,
        index=[
            f"Actual {label}"
            for label in labels
        ],
        columns=[
            f"Predicted {label}"
            for label in labels
        ]
    )

    report = classification_report(
        prepared["y_test"],
        predictions,
        output_dict=True,
        zero_division=0
    )

    report_table = (
        pd.DataFrame(report)
        .transpose()
        .reset_index()
        .rename(
            columns={
                "index": "Class"
            }
        )
    )

    return {
        "task": "classification",
        "model_name": model_name,
        "target_column": target_column,
        "feature_columns": prepared["selected_features"],
        "train_rows": len(prepared["X_train"]),
        "test_rows": len(prepared["X_test"]),
        "metrics_table": metrics_table,
        "confusion_matrix": confusion_table,
        "classification_report": report_table,
        "feature_importance": None
    }


def classification_pipeline(
    data,
    target_column,
    feature_columns=None,
    max_models=None
):
    """
    Streamlit-ready form of the notebook's classification_pipeline().
    It compares the same five notebook models.
    """
    model_names = list(
        get_classification_models().keys()
    )

    if max_models is not None:
        model_names = model_names[
            :max(
                1,
                min(
                    int(max_models),
                    len(model_names)
                )
            )
        ]

    rows = []

    for model_name in model_names:
        result = train_classification_model(
            data,
            target_column,
            model_name=model_name,
            feature_columns=feature_columns
        )

        metric_lookup = dict(
            zip(
                result["metrics_table"]["Metric"],
                result["metrics_table"]["Value"]
            )
        )

        rows.append(
            {
                "Model": model_name,
                "Accuracy": metric_lookup["Accuracy"],
                "Precision": metric_lookup["Precision (weighted)"],
                "Recall": metric_lookup["Recall (weighted)"],
                "F1 Score": metric_lookup["F1 Score (weighted)"]
            }
        )

    results_df = pd.DataFrame(
        rows
    )

    metric_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]

    results_df[metric_columns] = (
        results_df[metric_columns]
        .round(4)
    )

    results_df = (
        results_df
        .sort_values(
            by="F1 Score",
            ascending=False
        )
        .reset_index(drop=True)
    )

    return results_df


def _regression_split_and_scale(
    data,
    target_column,
    feature_columns=None
):
    """
    Shared preparation matching the notebook regression pipeline.
    """
    X, y, selected_features = prepare_ml_data(
        data,
        target_column,
        feature_columns
    )

    if not pd.api.types.is_numeric_dtype(y):
        raise ValueError(
            "Regression requires a numerical target column."
        )

    # Keep the Streamlit safety check that prevents binary/discrete
    # classification targets such as Outcome from being used in regression.
    if y.nunique() <= 10:
        raise ValueError(
            f"'{target_column}' has too few distinct values for regression "
            "and looks like a classification target."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "selected_features": selected_features
    }


def train_regression_model(
    data,
    target_column,
    model_name="random_forest_regressor",
    feature_columns=None
):
    """
    Train one of the five regression models from the notebook.
    """
    prepared = _regression_split_and_scale(
        data,
        target_column,
        feature_columns
    )

    models = get_regression_models()

    if model_name == "auto":
        model_name = "random_forest_regressor"

    if model_name not in models:
        raise ValueError(
            "Unknown regression model. Available models: "
            + ", ".join(models.keys())
        )

    model = models[model_name]

    if model_name in {
        "linear_regression",
        "knn_regressor"
    }:
        train_X = prepared["X_train_scaled"]
        test_X = prepared["X_test_scaled"]
    else:
        train_X = prepared["X_train"]
        test_X = prepared["X_test"]

    model.fit(
        train_X,
        prepared["y_train"]
    )

    predictions = model.predict(
        test_X
    )

    mae = mean_absolute_error(
        prepared["y_test"],
        predictions
    )

    mse = mean_squared_error(
        prepared["y_test"],
        predictions
    )

    rmse = float(
        np.sqrt(mse)
    )

    r2 = r2_score(
        prepared["y_test"],
        predictions
    )

    metrics_table = pd.DataFrame(
        {
            "Metric": [
                "MAE",
                "MSE",
                "RMSE",
                "R²"
            ],
            "Value": [
                mae,
                mse,
                rmse,
                r2
            ]
        }
    )

    predictions_table = pd.DataFrame(
        {
            "Actual": np.asarray(
                prepared["y_test"]
            )[:25],
            "Predicted": np.asarray(
                predictions
            )[:25]
        }
    )

    return {
        "task": "regression",
        "model_name": model_name,
        "target_column": target_column,
        "feature_columns": prepared["selected_features"],
        "train_rows": len(prepared["X_train"]),
        "test_rows": len(prepared["X_test"]),
        "metrics_table": metrics_table,
        "predictions_table": predictions_table,
        "feature_importance": None
    }


def regression_pipeline(
    data,
    target_column,
    feature_columns=None,
    max_models=None
):
    """
    Streamlit-ready form of the notebook's regression_pipeline().
    It compares the same five notebook models.
    """
    model_names = list(
        get_regression_models().keys()
    )

    if max_models is not None:
        model_names = model_names[
            :max(
                1,
                min(
                    int(max_models),
                    len(model_names)
                )
            )
        ]

    rows = []

    for model_name in model_names:
        result = train_regression_model(
            data,
            target_column,
            model_name=model_name,
            feature_columns=feature_columns
        )

        metric_lookup = dict(
            zip(
                result["metrics_table"]["Metric"],
                result["metrics_table"]["Value"]
            )
        )

        rows.append(
            {
                "Model": model_name,
                "MAE": metric_lookup["MAE"],
                "MSE": metric_lookup["MSE"],
                "RMSE": metric_lookup["RMSE"],
                "R²": metric_lookup["R²"]
            }
        )

    results_df = pd.DataFrame(
        rows
    )

    metric_columns = [
        "MAE",
        "MSE",
        "RMSE",
        "R²"
    ]

    results_df[metric_columns] = (
        results_df[metric_columns]
        .round(4)
    )

    results_df = (
        results_df
        .sort_values(
            by="R²",
            ascending=False
        )
        .reset_index(drop=True)
    )

    return results_df


def compare_classification_models(
    data,
    target_column,
    feature_columns=None,
    max_models=None
):
    """
    Compatibility wrapper used by the Streamlit chat and ML page.
    """
    return classification_pipeline(
        data,
        target_column,
        feature_columns=feature_columns,
        max_models=max_models
    )


def compare_regression_models(
    data,
    target_column,
    feature_columns=None,
    max_models=None
):
    """
    Compatibility wrapper used by the Streamlit chat and ML page.
    """
    return regression_pipeline(
        data,
        target_column,
        feature_columns=feature_columns,
        max_models=max_models
    )


def validate_ml_target_for_task(
    data,
    target_column,
    task
):
    """
    Validate the target before running classification/regression.
    """
    if target_column is None:
        raise ValueError(
            "Please specify the target column to predict."
        )

    if target_column not in data.columns:
        raise ValueError(
            f"Target column '{target_column}' does not exist."
        )

    target = data[target_column].dropna()

    if target.empty:
        raise ValueError(
            f"Target column '{target_column}' contains no usable values."
        )

    if task == "classification":
        if target.nunique() < 2:
            raise ValueError(
                "Classification requires at least two target classes."
            )

        if infer_ml_task(target) != "classification":
            raise ValueError(
                f"'{target_column}' looks like a continuous target. "
                "Use regression for this target."
            )

    elif task == "regression":
        if not pd.api.types.is_numeric_dtype(target):
            raise ValueError(
                "Regression requires a numerical target column."
            )

        if infer_ml_task(target) != "regression":
            raise ValueError(
                f"'{target_column}' has too few distinct values for regression "
                "and looks like a classification target."
            )

    return True


def run_ml_request(
    data,
    task,
    target_column,
    model_name="auto",
    feature_columns=None,
    compare_models=False,
    max_models=None
):
    """
    Streamlit controller over the notebook ML backend.
    """
    if target_column is None:
        raise ValueError(
            "Please specify the target column to predict."
        )

    if target_column not in data.columns:
        raise ValueError(
            f"Target column '{target_column}' does not exist."
        )

    if task == "auto":
        task = infer_ml_task(
            data[target_column]
        )

    validate_ml_target_for_task(
        data,
        target_column,
        task
    )

    if task == "classification":
        if compare_models:
            return {
                "type": "ml_comparison",
                "task": task,
                "content": (
                    f"Classification model comparison for target "
                    f"<b>{html.escape(str(target_column))}</b>"
                    + (
                        f" using {max_models} models."
                        if max_models
                        else " using all 5 notebook models."
                    )
                ),
                "comparison_table": compare_classification_models(
                    data,
                    target_column,
                    feature_columns,
                    max_models=max_models
                )
            }

        result = train_classification_model(
            data,
            target_column,
            model_name,
            feature_columns
        )

        return {
            "type": "ml_result",
            "content": (
                f"Classification model trained for target "
                f"<b>{html.escape(str(target_column))}</b> using "
                f"<b>{html.escape(result['model_name'])}</b>."
            ),
            **result
        }

    if task == "regression":
        if compare_models:
            return {
                "type": "ml_comparison",
                "task": task,
                "content": (
                    f"Regression model comparison for target "
                    f"<b>{html.escape(str(target_column))}</b>"
                    + (
                        f" using {max_models} models."
                        if max_models
                        else " using all 5 notebook models."
                    )
                ),
                "comparison_table": compare_regression_models(
                    data,
                    target_column,
                    feature_columns,
                    max_models=max_models
                )
            }

        result = train_regression_model(
            data,
            target_column,
            model_name,
            feature_columns
        )

        return {
            "type": "ml_result",
            "content": (
                f"Regression model trained for target "
                f"<b>{html.escape(str(target_column))}</b> using "
                f"<b>{html.escape(result['model_name'])}</b>."
            ),
            **result
        }

    raise ValueError(
        "Task must be classification, regression, or auto."
    )


# ============================================================
# LLM / CHAT WITH DATA
# ============================================================

def create_dataset_metadata(data):
    dataset_metadata = []

    for column in data.columns:
        sample_values = (
            data[column]
            .dropna()
            .astype(str)
            .head(5)
            .tolist()
        )

        dataset_metadata.append(
            {
                "name": str(column),
                "dtype": str(
                    data[column].dtype
                ),
                "missing_values": int(
                    data[column]
                    .isnull()
                    .sum()
                ),
                "unique_values": int(
                    data[column]
                    .nunique(
                        dropna=True
                    )
                ),
                "sample_values": sample_values
            }
        )

    return dataset_metadata


def create_system_prompt():
    return """
You are the interpretation layer for DataTalk, a conversational
analytics and machine-learning application.

Convert the user's natural-language request into ONE structured
JSON analysis plan.

Return JSON only.
Do not return markdown.
Do not return Python.
Do not return explanations.
Do not use code fences.

The JSON must contain exactly these keys:

{
  "operation": "",
  "x_column": null,
  "y_column": null,
  "aggregation": null,
  "time_grouping": null,
  "target_column": null,
  "model_name": null,
  "feature_columns": null,
  "top_n": null,
  "sort_order": null
}

Allowed operations:
row_count
mean
sum
standard_deviation
group_mean
group_sum
top_n
year_over_year
histogram
scatter
bar
line
boxplot
pie
count
heatmap
correlation
missing_values
outliers
describe
classification_model
classification_compare
regression_model
regression_compare
auto_ml
invalid_request

Allowed classification model_name values:
auto
logistic_regression
decision_tree
random_forest
knn
xgboost

Allowed regression model_name values:
auto
linear_regression
decision_tree_regressor
random_forest_regressor
knn_regressor
xgboost_regressor

Rules:
- Use only exact column names from dataset_metadata.
- Never invent a column.
- If a requested column does not exist, use invalid_request.
- Unused scalar fields must be null.
- feature_columns may be null or a list of exact dataset column names.
- top_n must be null or a positive integer. Default to 10 when the user says "top rows" without a number.
- For classification_compare or regression_compare, top_n means how many models to run. For example, "run only 5 classification models" means top_n=5.
- sort_order must be null, "asc", or "desc". Use "desc" for top/highest/largest and "asc" for lowest/smallest.
- mean, sum, and standard_deviation require a numerical x_column.
- group_mean and group_sum use x_column as grouping column and y_column as numerical value column.
- top_n uses x_column as the sort column and returns rows from the whole dataset.
- year_over_year uses x_column as a date/time column and y_column as a numerical value column. aggregation must be mean or sum.
- histogram and boxplot require one numerical x_column.
- scatter and correlation require two numerical columns.
- bar uses x_column as category and y_column as numerical value.
- line uses x_column as a date/time column, y_column as a numerical value, aggregation as mean/sum, and time_grouping as day/month/year.
- pie and count use x_column.
- heatmap, missing_values, outliers, and describe need no columns.
- classification_model trains one classification model and requires target_column.
- classification_compare compares classification models. If the user gives a number such as 5, put that number in top_n; otherwise top_n must be null.
- regression_model trains one regression model and requires target_column.
- regression_compare compares regression models. If the user gives a number such as 5, put that number in top_n; otherwise top_n must be null.
- auto_ml chooses classification or regression automatically from target_column.
- If the user says only "train a model to predict X", use auto_ml.
- The user_payload may include last_valid_ml_target and last_valid_ml_task.
- Reuse the remembered target ONLY when the new follow-up asks for the SAME ML task.
- Example: if last_valid_ml_target is "outcome" and last_valid_ml_task is "classification", then "run only 5 classification models" may reuse outcome.
- NEVER reuse a classification target for regression, and NEVER reuse a regression target for classification.
- If there is no valid remembered target for the requested task, use invalid_request and do not invent a target.
- If the user specifies features, put them in feature_columns.
- If the user does not specify features, feature_columns must be null.
- When the user says "distribution" of a numerical column, use histogram.
- When the user asks about the relationship between two numerical variables or says "x vs y", use scatter unless they explicitly ask for correlation.
- When the user requests "average X by Y", use group_mean with x_column=Y and y_column=X.
- When the user requests "total X by Y", use group_sum with x_column=Y and y_column=X.
- When the user asks "standard deviation of X", use standard_deviation with x_column=X.
- When the user asks "top 10 rows sorted by X", use top_n with x_column=X, top_n=10, sort_order="desc".
- When the user asks for year-over-year or YoY analysis, use year_over_year.
- When the request cannot be mapped safely, use invalid_request.

Examples:

Question: How many rows are there?
{"operation":"row_count","x_column":null,"y_column":null,"aggregation":null,"time_grouping":null,"target_column":null,"model_name":null,"feature_columns":null,"top_n":null,"sort_order":null}

Question: What is the average glucose?
{"operation":"mean","x_column":"glucose","y_column":null,"aggregation":null,"time_grouping":null,"target_column":null,"model_name":null,"feature_columns":null,"top_n":null,"sort_order":null}

Question: What is the standard deviation of glucose?
{"operation":"standard_deviation","x_column":"glucose","y_column":null,"aggregation":null,"time_grouping":null,"target_column":null,"model_name":null,"feature_columns":null,"top_n":null,"sort_order":null}

Question: Show the top 10 rows sorted by revenue
{"operation":"top_n","x_column":"revenue","y_column":null,"aggregation":null,"time_grouping":null,"target_column":null,"model_name":null,"feature_columns":null,"top_n":10,"sort_order":"desc"}

Question: Show year-over-year sales using order_date
{"operation":"year_over_year","x_column":"order_date","y_column":"sales","aggregation":"sum","time_grouping":"year","target_column":null,"model_name":null,"feature_columns":null,"top_n":null,"sort_order":null}

Question: Plot glucose vs bmi
{"operation":"scatter","x_column":"glucose","y_column":"bmi","aggregation":null,"time_grouping":null,"target_column":null,"model_name":null,"feature_columns":null,"top_n":null,"sort_order":null}

Question: Train a classification model to predict outcome
{"operation":"classification_model","x_column":null,"y_column":null,"aggregation":null,"time_grouping":null,"target_column":"outcome","model_name":"auto","feature_columns":null,"top_n":null,"sort_order":null}

Question: Compare classification models for outcome
{"operation":"classification_compare","x_column":null,"y_column":null,"aggregation":null,"time_grouping":null,"target_column":"outcome","model_name":null,"feature_columns":null,"top_n":null,"sort_order":null}

Question: Run only 5 classification models
Context: last_valid_ml_target is outcome and last_valid_ml_task is classification
{"operation":"classification_compare","x_column":null,"y_column":null,"aggregation":null,"time_grouping":null,"target_column":"outcome","model_name":null,"feature_columns":null,"top_n":5,"sort_order":null}

Question: Build a regression model to predict price
{"operation":"regression_model","x_column":null,"y_column":null,"aggregation":null,"time_grouping":null,"target_column":"price","model_name":"auto","feature_columns":null,"top_n":null,"sort_order":null}

Question: Compare regression models for sales
{"operation":"regression_compare","x_column":null,"y_column":null,"aggregation":null,"time_grouping":null,"target_column":"sales","model_name":null,"feature_columns":null,"top_n":null,"sort_order":null}

Question: Train a model to predict outcome
{"operation":"auto_ml","x_column":null,"y_column":null,"aggregation":null,"time_grouping":null,"target_column":"outcome","model_name":"auto","feature_columns":null,"top_n":null,"sort_order":null}
"""

def get_groq_client():
    api_key = None

    try:
        api_key = st.secrets.get(
            "GROQ_API_KEY"
        )
    except Exception:
        api_key = None

    if not api_key:
        api_key = os.getenv(
            "GROQ_API_KEY"
        )

    if not api_key:
        raise ValueError(
            "Groq API key is not configured. "
            "Add GROQ_API_KEY to .streamlit/secrets.toml."
        )

    return Groq(
        api_key=api_key
    )


def interpret_question_with_ai(
    question,
    data
):
    metadata = create_dataset_metadata(
        data
    )

    user_payload = {
        "dataset_metadata": metadata,
        "question": question,
        "last_valid_ml_target": st.session_state.get("last_valid_ml_target"),
        "last_valid_ml_task": st.session_state.get("last_valid_ml_task")
    }

    client = get_groq_client()

    response = (
        client
        .chat
        .completions
        .create(
            model=st.session_state.get("llm_model_name", "openai/gpt-oss-20b"),
            messages=[
                {
                    "role": "system",
                    "content": create_system_prompt()
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        user_payload
                    )
                }
            ],
            temperature=0
        )
    )

    ai_response = (
        response
        .choices[0]
        .message
        .content
    )

    if not ai_response:
        raise ValueError(
            "The AI returned an empty response."
        )

    ai_response = (
        ai_response
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )

    return json.loads(
        ai_response
    )


def validate_analysis_plan(
    plan,
    data
):
    if not isinstance(plan, dict):
        raise ValueError(
            "AI analysis plan must be a dictionary."
        )

    required_keys = {
        "operation",
        "x_column",
        "y_column",
        "aggregation",
        "time_grouping",
        "target_column",
        "model_name",
        "feature_columns",
        "top_n",
        "sort_order"
    }

    if set(plan.keys()) != required_keys:
        raise ValueError(
            "AI analysis plan has invalid keys."
        )

    allowed_operations = {
        "row_count",
        "mean",
        "sum",
        "standard_deviation",
        "group_mean",
        "group_sum",
        "top_n",
        "year_over_year",
        "histogram",
        "scatter",
        "bar",
        "line",
        "boxplot",
        "pie",
        "count",
        "heatmap",
        "correlation",
        "missing_values",
        "outliers",
        "describe",
        "classification_model",
        "classification_compare",
        "regression_model",
        "regression_compare",
        "auto_ml",
        "invalid_request"
    }

    operation = plan["operation"]
    x_column = plan["x_column"]
    y_column = plan["y_column"]
    aggregation = plan["aggregation"]
    time_grouping = plan["time_grouping"]
    target_column = plan["target_column"]
    model_name = plan["model_name"]
    feature_columns = plan["feature_columns"]
    top_n = plan["top_n"]
    sort_order = plan["sort_order"]

    if operation not in allowed_operations:
        raise ValueError(
            f"Unsupported operation: {operation}"
        )

    for column in [
        x_column,
        y_column,
        target_column
    ]:
        if (
            column is not None
            and column not in data.columns
        ):
            raise ValueError(
                f"Column '{column}' does not exist."
            )

    if feature_columns is not None:
        if not isinstance(feature_columns, list):
            raise ValueError(
                "feature_columns must be a list or null."
            )

        invalid_features = [
            column
            for column in feature_columns
            if column not in data.columns
        ]

        if invalid_features:
            raise ValueError(
                "Unknown feature columns: "
                + ", ".join(invalid_features)
            )

    if aggregation not in {
        None,
        "mean",
        "sum"
    }:
        raise ValueError(
            "Aggregation must be mean, sum, or null."
        )

    if time_grouping not in {
        None,
        "day",
        "month",
        "year"
    }:
        raise ValueError(
            "Time grouping must be day, month, year, or null."
        )

    if top_n is not None:
        if not isinstance(top_n, int) or top_n <= 0 or top_n > 100:
            raise ValueError(
                "top_n must be an integer between 1 and 100."
            )

    if sort_order not in {
        None,
        "asc",
        "desc"
    }:
        raise ValueError(
            "sort_order must be asc, desc, or null."
        )

    if operation in {
        "mean",
        "sum",
        "standard_deviation",
        "histogram",
        "boxplot"
    }:
        _require_column(
            data,
            x_column
        )
        _require_numeric(
            data,
            x_column
        )

    if operation in {
        "scatter",
        "correlation"
    }:
        _require_column(
            data,
            x_column
        )
        _require_column(
            data,
            y_column
        )
        _require_numeric(
            data,
            x_column
        )
        _require_numeric(
            data,
            y_column
        )

    if operation in {
        "group_mean",
        "group_sum",
        "bar"
    }:
        _require_column(
            data,
            x_column
        )
        _require_column(
            data,
            y_column
        )
        _require_numeric(
            data,
            y_column
        )

    if operation == "line":
        _require_column(
            data,
            x_column
        )
        _require_column(
            data,
            y_column
        )
        _require_numeric(
            data,
            y_column
        )

    if operation in {
        "pie",
        "count"
    }:
        _require_column(
            data,
            x_column
        )

    if operation == "top_n":
        _require_column(
            data,
            x_column
        )

    if operation == "year_over_year":
        _require_column(
            data,
            x_column
        )
        _require_column(
            data,
            y_column
        )
        _require_numeric(
            data,
            y_column
        )

    if operation in {
        "classification_model",
        "classification_compare",
        "regression_model",
        "regression_compare",
        "auto_ml"
    }:
        _require_column(
            data,
            target_column
        )

    if model_name is not None:
        allowed_model_names = (
            {"auto"}
            | set(get_classification_models().keys())
            | set(get_regression_models().keys())
        )

        if model_name not in allowed_model_names:
            raise ValueError(
                f"Unknown model name: {model_name}"
            )

    return True

def execute_analysis_plan(
    plan,
    data
):
    operation = plan["operation"]
    x_column = plan["x_column"]
    y_column = plan["y_column"]
    aggregation = plan["aggregation"]
    time_grouping = plan["time_grouping"]
    target_column = plan["target_column"]
    model_name = plan["model_name"]
    feature_columns = plan["feature_columns"]
    top_n = plan["top_n"]
    sort_order = plan["sort_order"]

    if operation == "row_count":
        return {
            "type": "text",
            "content": (
                f"The dataset contains "
                f"{len(data):,} rows."
            )
        }

    if operation == "mean":
        result = data[x_column].mean()

        return {
            "type": "text",
            "content": (
                f"The average {x_column} is "
                f"{result:,.2f}."
            )
        }

    if operation == "sum":
        result = data[x_column].sum()

        return {
            "type": "text",
            "content": (
                f"The total {x_column} is "
                f"{result:,.2f}."
            )
        }

    if operation == "standard_deviation":
        result = data[x_column].std()

        return {
            "type": "text",
            "content": (
                f"The standard deviation of {x_column} is "
                f"{result:,.2f}."
            )
        }

    if operation == "top_n":
        rows_to_show = top_n or 10
        ascending = (sort_order == "asc")

        table = (
            data
            .sort_values(
                by=x_column,
                ascending=ascending
            )
            .head(rows_to_show)
            .reset_index(drop=True)
        )

        direction = (
            "lowest"
            if ascending
            else "highest"
        )

        return {
            "type": "table",
            "content": (
                f"Here are the {rows_to_show} rows with the "
                f"{direction} {x_column} values."
            ),
            "table": table
        }

    if operation == "year_over_year":
        working = data[[x_column, y_column]].copy()
        working[x_column] = pd.to_datetime(
            working[x_column],
            errors="coerce"
        )
        working = working.dropna()

        if working.empty:
            raise ValueError(
                f"'{x_column}' could not be used as a date column."
            )

        working["year"] = working[x_column].dt.year

        if aggregation == "mean":
            yearly = (
                working
                .groupby("year")[y_column]
                .mean()
                .reset_index()
            )
        else:
            yearly = (
                working
                .groupby("year")[y_column]
                .sum()
                .reset_index()
            )

        yearly["yoy_change_percent"] = (
            yearly[y_column]
            .pct_change()
            .mul(100)
            .round(2)
        )

        return {
            "type": "table",
            "content": (
                f"Year-over-year {aggregation or 'sum'} of "
                f"{y_column} using {x_column}."
            ),
            "table": yearly
        }

    if operation == "group_mean":
        table = (
            data
            .groupby(
                x_column,
                dropna=False
            )[y_column]
            .mean()
            .reset_index()
        )

        return {
            "type": "table",
            "content": (
                f"Average {y_column} by {x_column}"
            ),
            "table": table
        }

    if operation == "group_sum":
        table = (
            data
            .groupby(
                x_column,
                dropna=False
            )[y_column]
            .sum()
            .reset_index()
        )

        return {
            "type": "table",
            "content": (
                f"Total {y_column} by {x_column}"
            ),
            "table": table
        }

    if operation in {
        "histogram",
        "scatter",
        "bar",
        "line",
        "boxplot",
        "pie",
        "count",
        "heatmap"
    }:
        return {
            "type": "chart",
            "content": _chart_message(
                plan
            ),
            "plan": plan
        }

    if operation == "correlation":
        correlation_value = (
            data[
                [x_column, y_column]
            ]
            .corr()
            .iloc[0, 1]
        )

        return {
            "type": "text",
            "content": (
                f"The correlation between "
                f"{x_column} and {y_column} is "
                f"{correlation_value:.3f}."
            )
        }

    if operation == "missing_values":
        return {
            "type": "table",
            "content": "Missing values report",
            "table": check_missing_values(
                data
            )
        }

    if operation == "outliers":
        return {
            "type": "table",
            "content": "IQR outlier report",
            "table": detect_outliers(
                data
            )
        }

    if operation == "describe":
        summary = basic_statistical_summary(
            data
        )

        if summary is None:
            return {
                "type": "text",
                "content": (
                    "No numerical columns are available "
                    "for descriptive statistics."
                )
            }

        return {
            "type": "table",
            "content": "Descriptive statistics",
            "table": summary
        }

    if operation == "classification_model":
        return run_ml_request(
            data=data,
            task="classification",
            target_column=target_column,
            model_name=(
                model_name or "auto"
            ),
            feature_columns=feature_columns,
            compare_models=False
        )

    if operation == "classification_compare":
        return run_ml_request(
            data=data,
            task="classification",
            target_column=target_column,
            model_name="auto",
            feature_columns=feature_columns,
            compare_models=True,
            max_models=top_n
        )

    if operation == "regression_model":
        return run_ml_request(
            data=data,
            task="regression",
            target_column=target_column,
            model_name=(
                model_name or "auto"
            ),
            feature_columns=feature_columns,
            compare_models=False
        )

    if operation == "regression_compare":
        return run_ml_request(
            data=data,
            task="regression",
            target_column=target_column,
            model_name="auto",
            feature_columns=feature_columns,
            compare_models=True,
            max_models=top_n
        )

    if operation == "auto_ml":
        return run_ml_request(
            data=data,
            task="auto",
            target_column=target_column,
            model_name=(
                model_name or "auto"
            ),
            feature_columns=feature_columns,
            compare_models=False
        )

    if operation == "invalid_request":
        return {
            "type": "text",
            "content": (
                "I could not map that request safely to "
                "the columns in the current dataset. "
                "Please check the column names or rephrase "
                "the question."
            )
        }

    raise ValueError(
        f"Unsupported operation: {operation}"
    )

def chat_with_data(
    question,
    data
):
    """Interpret and execute a conversational data request safely."""
    try:
        question_lower = question.lower().strip()

        # ----------------------------------------------------
        # DIRECT ML FOLLOW-UP HANDLING
        # ----------------------------------------------------
        followup_match = re.search(
            r"(?:run|compare)(?:\s+only)?\s+(\d+)\s+"
            r"(classification|regression)\s+models?",
            question_lower
        )

        if followup_match:
            model_count = int(followup_match.group(1))
            requested_task = followup_match.group(2)
            remembered_target = st.session_state.get(
                "last_valid_ml_target"
            )
            remembered_task = st.session_state.get(
                "last_valid_ml_task"
            )

            if (
                remembered_target is None
                or remembered_task != requested_task
                or remembered_target not in data.columns
            ):
                return {
                    "type": "text",
                    "content": (
                        f"Please specify a valid <b>{requested_task}</b> "
                        "target first. For example: "
                        f"<b>Compare {model_count} {requested_task} models "
                        "for &lt;target_column&gt;</b>."
                    )
                }

            validate_ml_target_for_task(
                data,
                remembered_target,
                requested_task
            )

            result = run_ml_request(
                data=data,
                task=requested_task,
                target_column=remembered_target,
                model_name="auto",
                feature_columns=None,
                compare_models=True,
                max_models=model_count
            )

            st.session_state.last_valid_ml_target = remembered_target
            st.session_state.last_valid_ml_task = requested_task
            return result

        # ----------------------------------------------------
        # NORMAL LLM INTERPRETATION
        # ----------------------------------------------------
        analysis_plan = interpret_question_with_ai(
            question,
            data
        )

        operation = analysis_plan.get("operation")
        ml_operations = {
            "classification_model": "classification",
            "classification_compare": "classification",
            "regression_model": "regression",
            "regression_compare": "regression",
            "auto_ml": "auto"
        }

        ml_words = (
            "classification" in question_lower
            or "regression" in question_lower
            or "model" in question_lower
            or "predict" in question_lower
        )

        if operation == "invalid_request" and ml_words:
            st.session_state.last_valid_ml_target = None
            st.session_state.last_valid_ml_task = None

            available_columns = ", ".join(
                map(str, data.columns.tolist())
            )

            return {
                "type": "text",
                "content": (
                    "I could not find a valid target column for that "
                    "machine-learning request.<br><br>"
                    f"<b>Available columns:</b> "
                    f"{html.escape(available_columns)}<br><br>"
                    "Please specify a target that exists in the dataset. "
                    "For example: <b>Compare classification models for outcome</b> "
                    "or <b>Compare regression models for glucose</b>."
                )
            }

        validate_analysis_plan(
            analysis_plan,
            data
        )

        if operation in ml_operations:
            target_column = analysis_plan.get("target_column")
            requested_task = ml_operations[operation]

            if requested_task == "auto":
                if target_column is None or target_column not in data.columns:
                    raise ValueError(
                        "Please specify a valid target column."
                    )
                requested_task = infer_ml_task(
                    data[target_column]
                )

            validate_ml_target_for_task(
                data,
                target_column,
                requested_task
            )

        result = execute_analysis_plan(
            analysis_plan,
            data
        )

        # Remember ML context only after successful execution.
        if operation in ml_operations:
            target_column = analysis_plan.get("target_column")

            if operation.startswith("classification"):
                successful_task = "classification"
            elif operation.startswith("regression"):
                successful_task = "regression"
            else:
                successful_task = infer_ml_task(
                    data[target_column]
                )

            st.session_state.last_valid_ml_target = target_column
            st.session_state.last_valid_ml_task = successful_task

        return result

    except Exception as error:
        question_lower = question.lower()

        if (
            "classification" in question_lower
            or "regression" in question_lower
            or "model" in question_lower
            or "predict" in question_lower
        ):
            st.session_state.last_valid_ml_target = None
            st.session_state.last_valid_ml_task = None

        return {
            "type": "text",
            "content": (
                "Unable to complete the request: "
                f"{html.escape(str(error))}"
            )
        }


def _chart_message(plan):
    operation = plan["operation"]
    x_column = plan["x_column"]
    y_column = plan["y_column"]

    if operation == "histogram":
        return (
            f"Here is the distribution of {x_column}."
        )

    if operation == "scatter":
        return (
            f"Here is the relationship between "
            f"{x_column} and {y_column}."
        )

    if operation == "bar":
        return (
            f"Here is the requested bar chart."
        )

    if operation == "line":
        return (
            f"Here is the requested trend chart."
        )

    if operation == "boxplot":
        return (
            f"Here is the box plot of {x_column}."
        )

    if operation == "pie":
        return (
            f"Here is the distribution of {x_column}."
        )

    if operation == "count":
        return (
            f"Here are the counts for {x_column}."
        )

    if operation == "heatmap":
        return "Here is the correlation heatmap."

    return "Here is the requested chart."


def render_chart_from_plan(
    plan,
    data
):
    operation = plan["operation"]
    x_column = plan["x_column"]
    y_column = plan["y_column"]
    aggregation = (
        plan["aggregation"]
        or "mean"
    )
    time_grouping = (
        plan["time_grouping"]
        or "month"
    )

    if operation == "histogram":
        return create_histogram(
            data,
            x_column
        )

    if operation == "scatter":
        return create_scatter(
            data,
            x_column,
            y_column
        )

    if operation == "bar":
        return create_bar_chart(
            data,
            x_column,
            y_column,
            aggregation
        )

    if operation == "line":
        return create_line_chart(
            data,
            x_column,
            y_column,
            aggregation,
            time_grouping
        )

    if operation == "boxplot":
        return create_boxplot(
            data,
            x_column
        )

    if operation == "pie":
        return create_pie_chart(
            data,
            x_column
        )

    if operation == "count":
        return create_count_chart(
            data,
            x_column
        )

    if operation == "heatmap":
        return create_correlation_heatmap(
            data
        )

    raise ValueError(
        f"Unsupported chart operation: {operation}"
    )


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "active_chat_index" not in st.session_state:
    st.session_state.active_chat_index = None

if "data" not in st.session_state:
    st.session_state.data = None

if "cleaned_data" not in st.session_state:
    st.session_state.cleaned_data = None

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None

if "cleaning_summary" not in st.session_state:
    st.session_state.cleaning_summary = None

if "missing_report" not in st.session_state:
    st.session_state.missing_report = None

if "outlier_report" not in st.session_state:
    st.session_state.outlier_report = None

if "last_ml_target" not in st.session_state:
    st.session_state.last_ml_target = None

# Valid ML memory is updated only after a successful ML request.
if "last_valid_ml_target" not in st.session_state:
    st.session_state.last_valid_ml_target = None

if "last_valid_ml_task" not in st.session_state:
    st.session_state.last_valid_ml_task = None

if "ml_dashboard_result" not in st.session_state:
    st.session_state.ml_dashboard_result = None

if "llm_model_name" not in st.session_state:
    st.session_state.llm_model_name = "openai/gpt-oss-20b"

if "theme_choice" not in st.session_state:
    st.session_state.theme_choice = "Auto"

if "default_cap_outliers" not in st.session_state:
    st.session_state.default_cap_outliers = False

if "creator_name" not in st.session_state:
    st.session_state.creator_name = "Project Creator"



# ============================================================
# NAVIGATION / PAGE STATE
# ============================================================

if "navigation_page" not in st.session_state:
    st.session_state.navigation_page = "💬 Chat"


# ============================================================
# CHAT WORKSPACE HELPERS
# ============================================================

def make_chat_snapshot():
    return {
        "messages": st.session_state.messages.copy(),
        "data": (
            st.session_state.data.copy()
            if isinstance(st.session_state.data, pd.DataFrame)
            else None
        ),
        "cleaned_data": (
            st.session_state.cleaned_data.copy()
            if isinstance(st.session_state.cleaned_data, pd.DataFrame)
            else None
        ),
        "uploaded_file_name": st.session_state.uploaded_file_name,
        "cleaning_summary": st.session_state.cleaning_summary,
        "last_ml_target": st.session_state.last_ml_target,
        "last_valid_ml_target": st.session_state.last_valid_ml_target,
        "last_valid_ml_task": st.session_state.last_valid_ml_task,
        "missing_report": (
            st.session_state.missing_report.copy()
            if isinstance(st.session_state.missing_report, pd.DataFrame)
            else None
        ),
        "outlier_report": (
            st.session_state.outlier_report.copy()
            if isinstance(st.session_state.outlier_report, pd.DataFrame)
            else None
        ),
        "ml_dashboard_result": st.session_state.ml_dashboard_result,
    }


def restore_chat_snapshot(chat):
    st.session_state.messages = chat.get("messages", []).copy()

    st.session_state.data = (
        chat["data"].copy()
        if isinstance(chat.get("data"), pd.DataFrame)
        else None
    )

    st.session_state.cleaned_data = (
        chat["cleaned_data"].copy()
        if isinstance(chat.get("cleaned_data"), pd.DataFrame)
        else None
    )

    st.session_state.uploaded_file_name = chat.get(
        "uploaded_file_name"
    )
    st.session_state.cleaning_summary = chat.get(
        "cleaning_summary"
    )

    st.session_state.missing_report = (
        chat["missing_report"].copy()
        if isinstance(chat.get("missing_report"), pd.DataFrame)
        else None
    )

    st.session_state.last_ml_target = (
        chat.get("last_ml_target")
    )

    st.session_state.last_valid_ml_target = (
        chat.get("last_valid_ml_target")
    )

    st.session_state.last_valid_ml_task = (
        chat.get("last_valid_ml_task")
    )

    st.session_state.outlier_report = (
        chat["outlier_report"].copy()
        if isinstance(chat.get("outlier_report"), pd.DataFrame)
        else None
    )

    st.session_state.ml_dashboard_result = chat.get(
        "ml_dashboard_result"
    )




def open_recent_chat(index):
    chat = st.session_state.chat_history[index]
    restore_chat_snapshot(chat)
    st.session_state.active_chat_index = index
    st.session_state.navigation_page = "💬 Chat"

def clear_current_chat():
    st.session_state.messages = []
    st.session_state.active_chat_index = None
    st.session_state.data = None
    st.session_state.cleaned_data = None
    st.session_state.uploaded_file_name = None
    st.session_state.cleaning_summary = None
    st.session_state.missing_report = None
    st.session_state.outlier_report = None
    st.session_state.last_ml_target = None
    st.session_state.last_valid_ml_target = None
    st.session_state.last_valid_ml_task = None
    st.session_state.ml_dashboard_result = None


# ============================================================
# SIDEBAR
# All main areas live here as separate pages.
# ============================================================

with st.sidebar:
    st.title("DataTalk")

    if st.button(
        "＋ New Chat",
        use_container_width=True
    ):
        if st.session_state.messages:
            snapshot = make_chat_snapshot()

            if st.session_state.active_chat_index is None:
                first_user_message = next(
                    (
                        message.get("content", "")
                        for message in st.session_state.messages
                        if message.get("role") == "user"
                    ),
                    "New Chat"
                )

                chat_title = (
                    first_user_message[:30] + "..."
                    if len(first_user_message) > 30
                    else first_user_message
                )

                snapshot["title"] = chat_title
                st.session_state.chat_history.insert(
                    0,
                    snapshot
                )
            else:
                index = st.session_state.active_chat_index
                snapshot["title"] = (
                    st.session_state.chat_history[index]["title"]
                )
                st.session_state.chat_history[index] = snapshot

        clear_current_chat()
        st.session_state.navigation_page = "💬 Chat"
        st.rerun()

    st.markdown("### Navigation")

    navigation_options = [
        "💬 Chat",
        "📊 Dataset Preview",
        "🧹 Data Cleaning",
        "📈 EDA Dashboard",
        "🎨 Auto Visualizations",
        "🤖 Machine Learning",
        "⚙️ Settings",
        "ℹ️ About Project",
    ]

    current_page = st.radio(
        "Go to",
        options=navigation_options,
        key="navigation_page",
        label_visibility="collapsed"
    )

    st.divider()
    st.subheader("Recents")

    if not st.session_state.chat_history:
        st.caption("No recent chats yet.")

    for index, chat in enumerate(
        st.session_state.chat_history
    ):
        st.button(
            chat["title"],
            key=f"recent_chat_{index}",
            use_container_width=True,
            on_click=open_recent_chat,
            args=(index,)
        )


apply_selected_theme()


# ============================================================
# PAGE: CHAT
# Text-based control center. The user can ask for rows, metrics,
# visualizations, outliers, classification, regression, etc.
# ============================================================

if current_page == "💬 Chat":

    if st.session_state.data is None:
        st.caption(
            "Attach a CSV or XLSX file in the message box to begin."
        )
    else:
        status_text = (
            f"Dataset: {st.session_state.uploaded_file_name}"
        )

        if st.session_state.cleaned_data is not None:
            status_text += "  •  Ready for analysis"
        else:
            status_text += "  •  Run Data Cleaning before asking analysis questions"

        st.caption(status_text)

    # --------------------------------------------------------
    # Render chat history
    # --------------------------------------------------------

    for message_index, message in enumerate(
        st.session_state.messages
    ):
        role = message.get("role", "assistant")
        message_type = message.get("type", "text")
        content = message.get("content", "")

        if role == "user":
            safe_content = html.escape(str(content))

            st.markdown(
                f"""
                <div style="
                    display:flex;
                    justify-content:flex-end;
                    margin-bottom:18px;
                ">
                    <div style="
                        background-color:#e8f2ff;
                        padding:11px 17px;
                        border-radius:21px;
                        max-width:60%;
                        font-size:16px;
                        line-height:1.5;
                    ">
                        {safe_content}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:
            if content:
                st.markdown(
                    f"""
                    <div style="
                        margin-top:6px;
                        margin-bottom:12px;
                        max-width:78%;
                        font-size:16px;
                        line-height:1.6;
                    ">
                        {content}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            if (
                message_type == "table"
                and isinstance(
                    message.get("table"),
                    pd.DataFrame
                )
            ):
                st.dataframe(
                    message["table"],
                    use_container_width=True
                )

            elif (
                message_type == "chart"
                and st.session_state.cleaned_data is not None
            ):
                try:
                    figure = render_chart_from_plan(
                        message["plan"],
                        st.session_state.cleaned_data
                    )

                    st.pyplot(
                        figure,
                        use_container_width=False
                    )

                    download_chart_button(
                        figure,
                        chart_file_name(message["plan"]),
                        key=(
                            f"download_chat_chart_"
                            f"{message_index}"
                        )
                    )

                    plt.close(figure)

                except Exception as error:
                    st.error(
                        f"Chart could not be rendered: {error}"
                    )

            elif message_type == "ml_result":
                st.caption(
                    f"Task: {message.get('task', '').title()} | "
                    f"Target: {message.get('target_column', '')} | "
                    f"Model: {message.get('model_name', '')}"
                )

                if isinstance(
                    message.get("metrics_table"),
                    pd.DataFrame
                ):
                    st.markdown("**Model Metrics**")
                    st.dataframe(
                        message["metrics_table"],
                        use_container_width=True
                    )

                if isinstance(
                    message.get("confusion_matrix"),
                    pd.DataFrame
                ):
                    st.markdown("**Confusion Matrix**")
                    st.dataframe(
                        message["confusion_matrix"],
                        use_container_width=True
                    )

                if isinstance(
                    message.get("classification_report"),
                    pd.DataFrame
                ):
                    st.markdown("**Classification Report**")
                    st.dataframe(
                        message["classification_report"],
                        use_container_width=True
                    )

                if isinstance(
                    message.get("predictions_table"),
                    pd.DataFrame
                ):
                    st.markdown("**Sample Predictions**")
                    st.dataframe(
                        message["predictions_table"],
                        use_container_width=True
                    )

                if isinstance(
                    message.get("feature_importance"),
                    pd.DataFrame
                ):
                    st.markdown("**Top Feature Importance**")
                    st.dataframe(
                        message["feature_importance"],
                        use_container_width=True
                    )

            elif message_type == "ml_comparison":
                if isinstance(
                    message.get("comparison_table"),
                    pd.DataFrame
                ):
                    st.markdown("**Model Comparison**")
                    st.dataframe(
                        message["comparison_table"],
                        use_container_width=True
                    )

    # --------------------------------------------------------
    # Chat input lives ONLY on the Chat page
    # --------------------------------------------------------

    user_message = st.chat_input(
        "Ask DataTalk",
        accept_file=True,
        file_type=[
            "csv",
            "xlsx",
            "pdf"
        ]
    )

    # --------------------------------------------------------
    # Handle new chat message
    # --------------------------------------------------------

    if user_message:
        message_text = (
            user_message.text or ""
        ).strip()

        uploaded_files = user_message.files

        if message_text:
            st.session_state.messages.append(
                {
                    "role": "user",
                    "type": "text",
                    "content": message_text
                }
            )

        if uploaded_files:
            uploaded_file = uploaded_files[0]
            original_file_name = uploaded_file.name
            file_name = original_file_name.lower()

            if file_name.endswith(".csv"):
                try:
                    data = pd.read_csv(uploaded_file)

                    st.session_state.data = data
                    st.session_state.uploaded_file_name = (
                        original_file_name
                    )
                    st.session_state.cleaned_data = None
                    st.session_state.cleaning_summary = None
                    st.session_state.missing_report = None
                    st.session_state.outlier_report = None
                    st.session_state.ml_dashboard_result = None

                    assistant_result = {
                        "role": "assistant",
                        "type": "text",
                        "content": (
                            "Dataset uploaded successfully."
                            "<br><br>"
                            f"<b>File:</b> "
                            f"{html.escape(original_file_name)}"
                            "<br>"
                            f"<b>Rows:</b> {data.shape[0]}"
                            "<br>"
                            f"<b>Columns:</b> {data.shape[1]}"
                            "<br><br>"
                            "Open <b>Data Cleaning</b> from the left menu, "
                            "run the cleaning pipeline, then return to Chat."
                        )
                    }

                except Exception as error:
                    assistant_result = {
                        "role": "assistant",
                        "type": "text",
                        "content": (
                            "I could not read the CSV file."
                            "<br><br>"
                            f"<b>Error:</b> "
                            f"{html.escape(str(error))}"
                        )
                    }

            elif file_name.endswith(".xlsx"):
                try:
                    data = pd.read_excel(uploaded_file)

                    st.session_state.data = data
                    st.session_state.uploaded_file_name = (
                        original_file_name
                    )
                    st.session_state.cleaned_data = None
                    st.session_state.cleaning_summary = None
                    st.session_state.missing_report = None
                    st.session_state.outlier_report = None
                    st.session_state.ml_dashboard_result = None

                    assistant_result = {
                        "role": "assistant",
                        "type": "text",
                        "content": (
                            "Dataset uploaded successfully."
                            "<br><br>"
                            f"<b>File:</b> "
                            f"{html.escape(original_file_name)}"
                            "<br>"
                            f"<b>Rows:</b> {data.shape[0]}"
                            "<br>"
                            f"<b>Columns:</b> {data.shape[1]}"
                            "<br><br>"
                            "Open <b>Data Cleaning</b> from the left menu, "
                            "run the cleaning pipeline, then return to Chat."
                        )
                    }

                except Exception as error:
                    assistant_result = {
                        "role": "assistant",
                        "type": "text",
                        "content": (
                            "I could not read the Excel file."
                            "<br><br>"
                            f"<b>Error:</b> "
                            f"{html.escape(str(error))}"
                        )
                    }

            elif file_name.endswith(".pdf"):
                assistant_result = {
                    "role": "assistant",
                    "type": "text",
                    "content": (
                        "PDF file detected.<br><br>"
                        "DataTalk's dataset analysis uses CSV and XLSX files."
                    )
                }

            else:
                assistant_result = {
                    "role": "assistant",
                    "type": "text",
                    "content": "This file type is not supported."
                }

            st.session_state.messages.append(
                assistant_result
            )

        elif message_text:
            if st.session_state.cleaned_data is not None:
                result = chat_with_data(
                    message_text,
                    st.session_state.cleaned_data
                )

                assistant_message = {
                    "role": "assistant",
                    "type": result["type"],
                    "content": result.get("content", "")
                }

                if result["type"] == "table":
                    assistant_message["table"] = result["table"]

                elif result["type"] == "chart":
                    assistant_message["plan"] = result["plan"]

                elif result["type"] == "ml_result":
                    for key in [
                        "task",
                        "model_name",
                        "target_column",
                        "feature_columns",
                        "train_rows",
                        "test_rows",
                        "metrics_table",
                        "confusion_matrix",
                        "classification_report",
                        "predictions_table",
                        "feature_importance"
                    ]:
                        if key in result:
                            assistant_message[key] = result[key]

                elif result["type"] == "ml_comparison":
                    assistant_message["task"] = result.get("task")
                    assistant_message["comparison_table"] = (
                        result["comparison_table"]
                    )

                st.session_state.messages.append(
                    assistant_message
                )

            elif st.session_state.data is not None:
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "type": "text",
                        "content": (
                            "Your dataset is loaded. "
                            "Open <b>Data Cleaning</b> from the left menu "
                            "and run it before analysis."
                        )
                    }
                )

            else:
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "type": "text",
                        "content": (
                            "Please attach a CSV or XLSX dataset "
                            "to begin the analysis."
                        )
                    }
                )

        else:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "type": "text",
                    "content": (
                        "Please type a message or attach a file."
                    )
                }
            )

        st.rerun()


# ============================================================
# PAGE: DATASET PREVIEW
# ============================================================

elif current_page == "📊 Dataset Preview":

    st.header("📊 Dataset Preview")

    if st.session_state.data is None:
        st.info(
            "No dataset is loaded. Go to Chat and attach a CSV or XLSX file."
        )
    else:
        data = st.session_state.data

        st.write(
            f"**File:** {st.session_state.uploaded_file_name}"
        )

        col1, col2 = st.columns(2)
        col1.metric("Rows", data.shape[0])
        col2.metric("Columns", data.shape[1])

        st.subheader("First 10 Rows")
        st.dataframe(
            data.head(10),
            use_container_width=True
        )

        st.subheader("Column Metadata")
        metadata_table = pd.DataFrame(
            {
                "Column": data.columns,
                "Data Type": data.dtypes.astype(str).values,
                "Missing Values": data.isnull().sum().values,
                "Unique Values": [
                    int(data[column].nunique(dropna=True))
                    for column in data.columns
                ]
            }
        )
        st.dataframe(
            metadata_table,
            use_container_width=True
        )


# ============================================================
# PAGE: DATA CLEANING
# ============================================================

elif current_page == "🧹 Data Cleaning":

    st.header("🧹 Data Cleaning")

    if st.session_state.data is None:
        st.info(
            "No dataset is loaded. Go to Chat and attach a CSV or XLSX file."
        )
    else:
        st.markdown(
            """
            The cleaning pipeline can:
            - standardize column names,
            - remove duplicate rows,
            - detect and fill missing values,
            - detect date columns,
            - detect IQR outliers,
            - optionally treat selected zero values as missing,
            - optionally cap IQR outliers.
            """
        )

        numeric_original_columns = (
            st.session_state.data
            .select_dtypes(include="number")
            .columns
            .tolist()
        )

        zero_as_missing_columns = st.multiselect(
            "Optional: treat 0 as missing in selected columns",
            options=numeric_original_columns,
            default=[],
            help=(
                "Use this only when zero is impossible or represents "
                "a missing measurement in that specific column."
            )
        )

        cap_outlier_values = st.checkbox(
            "Cap outliers using IQR",
            value=st.session_state.default_cap_outliers,
            help=(
                "Outliers are always detected. Enable this only if "
                "you want extreme numerical values clipped to the "
                "IQR boundaries."
            )
        )

        if st.button(
            "Run Data Cleaning",
            type="primary"
        ):
            try:
                (
                    cleaned_data,
                    cleaning_summary,
                    missing_report,
                    outlier_report
                ) = clean_dataset(
                    st.session_state.data,
                    zero_as_missing_columns=(
                        zero_as_missing_columns
                    ),
                    cap_outlier_values=(
                        cap_outlier_values
                    )
                )

                st.session_state.cleaned_data = cleaned_data
                st.session_state.cleaning_summary = cleaning_summary
                st.session_state.missing_report = missing_report
                st.session_state.outlier_report = outlier_report
                st.session_state.ml_dashboard_result = None

                st.success(
                    "Data cleaning completed successfully."
                )

            except Exception as error:
                st.error(
                    f"Cleaning failed: {error}"
                )

        if st.session_state.cleaned_data is not None:
            summary = st.session_state.cleaning_summary

            st.subheader("Cleaning Summary")

            c1, c2, c3 = st.columns(3)
            c1.metric(
                "Rows",
                st.session_state.cleaned_data.shape[0]
            )
            c2.metric(
                "Columns",
                st.session_state.cleaned_data.shape[1]
            )
            c3.metric(
                "Duplicates Removed",
                summary["Duplicate Rows Removed"]
            )

            c4, c5 = st.columns(2)
            c4.metric(
                "Missing Before Filling",
                summary["Missing Before Filling"]
            )
            c5.metric(
                "Missing After",
                summary["Remaining Missing Values"]
            )

            st.write(
                "**Detected Date Columns:**",
                summary["Detected Date Columns"]
            )
            st.write(
                "**Zero-as-Missing Columns:**",
                summary["Zero-as-Missing Columns"]
            )
            st.write(
                "**Outliers Capped:**",
                summary["Outliers Capped"]
            )

            st.subheader("Cleaned Dataset Preview")
            st.dataframe(
                st.session_state.cleaned_data.head(10),
                use_container_width=True
            )

            st.subheader("Missing Value Report Before Filling")
            st.dataframe(
                st.session_state.missing_report,
                use_container_width=True
            )

            st.subheader("IQR Outlier Report")
            st.dataframe(
                st.session_state.outlier_report,
                use_container_width=True
            )


# ============================================================
# PAGE: EDA DASHBOARD
# ============================================================

elif current_page == "📈 EDA Dashboard":

    st.header("📈 EDA Dashboard")

    if st.session_state.cleaned_data is None:
        st.info(
            "Run Data Cleaning first. The EDA dashboard uses the cleaned dataset."
        )
    else:
        analysis_data = st.session_state.cleaned_data

        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", analysis_data.shape[0])
        c2.metric("Columns", analysis_data.shape[1])
        c3.metric(
            "Missing Values",
            int(analysis_data.isnull().sum().sum())
        )

        st.subheader("Numerical Statistical Summary")
        numerical_summary = basic_statistical_summary(
            analysis_data
        )

        if numerical_summary is not None:
            st.dataframe(
                numerical_summary,
                use_container_width=True
            )
        else:
            st.info("No numerical columns found.")

        st.subheader("Categorical Summary")
        category_report = categorical_summary(
            analysis_data
        )

        if category_report is not None:
            st.dataframe(
                category_report,
                use_container_width=True
            )
        else:
            st.info("No categorical columns found.")

        st.subheader("Missing Values")
        st.dataframe(
            check_missing_values(analysis_data),
            use_container_width=True
        )

        st.subheader("Outlier Analysis")
        st.dataframe(
            detect_outliers(analysis_data),
            use_container_width=True
        )

        st.subheader("Correlation Matrix")
        correlation_matrix = correlation_analysis(
            analysis_data
        )

        if correlation_matrix is not None:
            st.dataframe(
                correlation_matrix,
                use_container_width=True
            )

            st.subheader("Correlation Heatmap")
            figure = create_correlation_heatmap(
                analysis_data
            )
            st.pyplot(
                figure,
                use_container_width=False
            )
            download_chart_button(
                figure,
                "datatalk_eda_correlation_heatmap.png",
                key="download_eda_heatmap"
            )
            plt.close(figure)
        else:
            st.info(
                "At least two numerical columns are required for correlation analysis."
            )

        st.subheader("Column Distribution")
        numeric_columns = (
            analysis_data
            .select_dtypes(include="number")
            .columns
            .tolist()
        )

        if numeric_columns:
            selected_column = st.selectbox(
                "Select a numerical column",
                numeric_columns,
                key="eda_distribution_column"
            )

            histogram = create_histogram(
                analysis_data,
                selected_column
            )
            st.pyplot(
                histogram,
                use_container_width=False
            )
            download_chart_button(
                histogram,
                f"datatalk_histogram_{selected_column}.png",
                key="download_eda_histogram"
            )
            plt.close(histogram)

            boxplot = create_boxplot(
                analysis_data,
                selected_column
            )
            st.pyplot(
                boxplot,
                use_container_width=False
            )
            download_chart_button(
                boxplot,
                f"datatalk_boxplot_{selected_column}.png",
                key="download_eda_boxplot"
            )
            plt.close(boxplot)
        else:
            st.info(
                "No numerical columns are available for distribution analysis."
            )


# ============================================================
# PAGE: AUTO VISUALIZATIONS
# Manual chart creation is kept separate from Chat.
# ============================================================

elif current_page == "🎨 Auto Visualizations":

    st.header("🎨 Auto Visualizations")

    if st.session_state.cleaned_data is None:
        st.info(
            "Run Data Cleaning first. Manual visualizations use the cleaned dataset."
        )
    else:
        visualization_data = st.session_state.cleaned_data

        chart_type = st.selectbox(
            "Chart type",
            [
                "Histogram",
                "Scatter Plot",
                "Bar Chart",
                "Line Chart",
                "Boxplot",
                "Pie Chart",
                "Count Chart",
                "Correlation Heatmap"
            ]
        )

        all_columns = visualization_data.columns.tolist()
        numeric_columns = (
            visualization_data
            .select_dtypes(include="number")
            .columns
            .tolist()
        )

        manual_figure = None
        manual_file_name = "datatalk_chart.png"

        try:
            if chart_type == "Histogram":
                column = st.selectbox(
                    "Numerical column",
                    numeric_columns,
                    key="manual_hist_column"
                )
                manual_figure = create_histogram(
                    visualization_data,
                    column
                )
                manual_file_name = (
                    f"datatalk_histogram_{column}.png"
                )

            elif chart_type == "Scatter Plot":
                x_column = st.selectbox(
                    "X-axis",
                    numeric_columns,
                    key="manual_scatter_x"
                )
                y_column = st.selectbox(
                    "Y-axis",
                    numeric_columns,
                    key="manual_scatter_y"
                )
                manual_figure = create_scatter(
                    visualization_data,
                    x_column,
                    y_column
                )
                manual_file_name = (
                    f"datatalk_scatter_{x_column}_{y_column}.png"
                )

            elif chart_type == "Bar Chart":
                x_column = st.selectbox(
                    "Category / X-axis",
                    all_columns,
                    key="manual_bar_x"
                )
                y_column = st.selectbox(
                    "Numerical / Y-axis",
                    numeric_columns,
                    key="manual_bar_y"
                )
                aggregation = st.selectbox(
                    "Aggregation",
                    ["mean", "sum"],
                    key="manual_bar_agg"
                )
                manual_figure = create_bar_chart(
                    visualization_data,
                    x_column,
                    y_column,
                    aggregation
                )
                manual_file_name = (
                    f"datatalk_bar_{x_column}_{y_column}.png"
                )

            elif chart_type == "Line Chart":
                x_column = st.selectbox(
                    "Date / X-axis",
                    all_columns,
                    key="manual_line_x"
                )
                y_column = st.selectbox(
                    "Numerical / Y-axis",
                    numeric_columns,
                    key="manual_line_y"
                )
                aggregation = st.selectbox(
                    "Aggregation",
                    ["sum", "mean"],
                    key="manual_line_agg"
                )
                time_grouping = st.selectbox(
                    "Time grouping",
                    ["day", "month", "year"],
                    key="manual_line_group"
                )
                manual_figure = create_line_chart(
                    visualization_data,
                    x_column,
                    y_column,
                    aggregation,
                    time_grouping
                )
                manual_file_name = (
                    f"datatalk_line_{x_column}_{y_column}.png"
                )

            elif chart_type == "Boxplot":
                column = st.selectbox(
                    "Numerical column",
                    numeric_columns,
                    key="manual_box_column"
                )
                manual_figure = create_boxplot(
                    visualization_data,
                    column
                )
                manual_file_name = (
                    f"datatalk_boxplot_{column}.png"
                )

            elif chart_type == "Pie Chart":
                column = st.selectbox(
                    "Column",
                    all_columns,
                    key="manual_pie_column"
                )
                manual_figure = create_pie_chart(
                    visualization_data,
                    column
                )
                manual_file_name = (
                    f"datatalk_pie_{column}.png"
                )

            elif chart_type == "Count Chart":
                column = st.selectbox(
                    "Column",
                    all_columns,
                    key="manual_count_column"
                )
                manual_figure = create_count_chart(
                    visualization_data,
                    column
                )
                manual_file_name = (
                    f"datatalk_count_{column}.png"
                )

            elif chart_type == "Correlation Heatmap":
                manual_figure = create_correlation_heatmap(
                    visualization_data
                )
                manual_file_name = (
                    "datatalk_correlation_heatmap.png"
                )

            if manual_figure is not None:
                st.pyplot(
                    manual_figure,
                    use_container_width=False
                )

                download_chart_button(
                    manual_figure,
                    manual_file_name,
                    key="download_manual_chart"
                )

                plt.close(manual_figure)

        except Exception as error:
            st.warning(
                f"Chart could not be created: {error}"
            )


# ============================================================
# PAGE: MACHINE LEARNING
# ============================================================

elif current_page == "🤖 Machine Learning":

    st.header("🤖 Machine Learning")

    if st.session_state.cleaned_data is None:
        st.info(
            "Run Data Cleaning first. Machine learning uses the cleaned dataset."
        )
    else:
        ml_data = st.session_state.cleaned_data

        st.write(
            "Train or compare classification and regression models manually. "
            "You can also request these tasks in Chat using plain English."
        )

        target_column = st.selectbox(
            "Target column",
            options=ml_data.columns.tolist(),
            key="ml_target_column"
        )

        task_choice = st.selectbox(
            "Task type",
            options=[
                "Auto Detect",
                "Classification",
                "Regression"
            ],
            key="ml_task_choice"
        )

        available_features = [
            column
            for column in ml_data.columns
            if column != target_column
        ]

        selected_features = st.multiselect(
            "Feature columns (leave empty to use all eligible features)",
            options=available_features,
            default=[],
            key="ml_feature_columns"
        )

        if task_choice == "Classification":
            task_key = "classification"
            model_options = [
                "auto",
                *get_classification_models().keys()
            ]

        elif task_choice == "Regression":
            task_key = "regression"
            model_options = [
                "auto",
                *get_regression_models().keys()
            ]

        else:
            task_key = "auto"
            inferred = infer_ml_task(
                ml_data[target_column]
            )

            st.info(
                f"Auto-detected task: {inferred.title()}"
            )

            if inferred == "classification":
                model_options = [
                    "auto",
                    *get_classification_models().keys()
                ]
            else:
                model_options = [
                    "auto",
                    *get_regression_models().keys()
                ]

        selected_model = st.selectbox(
            "Model",
            options=model_options,
            key="ml_model_choice"
        )

        ml_col1, ml_col2 = st.columns(2)

        if ml_col1.button(
            "Train Selected Model",
            type="primary",
            use_container_width=True
        ):
            try:
                st.session_state.ml_dashboard_result = run_ml_request(
                    data=ml_data,
                    task=task_key,
                    target_column=target_column,
                    model_name=selected_model,
                    feature_columns=(
                        selected_features
                        if selected_features
                        else None
                    ),
                    compare_models=False
                )
            except Exception as error:
                st.session_state.ml_dashboard_result = {
                    "type": "text",
                    "content": f"Model training failed: {error}"
                }

        if ml_col2.button(
            "Compare All Models",
            use_container_width=True
        ):
            try:
                comparison_task = task_key

                if comparison_task == "auto":
                    comparison_task = infer_ml_task(
                        ml_data[target_column]
                    )

                st.session_state.ml_dashboard_result = run_ml_request(
                    data=ml_data,
                    task=comparison_task,
                    target_column=target_column,
                    model_name="auto",
                    feature_columns=(
                        selected_features
                        if selected_features
                        else None
                    ),
                    compare_models=True
                )
            except Exception as error:
                st.session_state.ml_dashboard_result = {
                    "type": "text",
                    "content": f"Model comparison failed: {error}"
                }

        ml_result = st.session_state.ml_dashboard_result

        if ml_result:
            st.markdown(
                ml_result.get("content", ""),
                unsafe_allow_html=True
            )

            if ml_result.get("type") == "ml_result":
                st.caption(
                    f"Task: {ml_result.get('task', '').title()} | "
                    f"Target: {ml_result.get('target_column', '')} | "
                    f"Model: {ml_result.get('model_name', '')} | "
                    f"Train rows: {ml_result.get('train_rows', '')} | "
                    f"Test rows: {ml_result.get('test_rows', '')}"
                )

                st.subheader("Model Metrics")
                st.dataframe(
                    ml_result["metrics_table"],
                    use_container_width=True
                )

                if isinstance(
                    ml_result.get("confusion_matrix"),
                    pd.DataFrame
                ):
                    st.subheader("Confusion Matrix")
                    st.dataframe(
                        ml_result["confusion_matrix"],
                        use_container_width=True
                    )

                if isinstance(
                    ml_result.get("classification_report"),
                    pd.DataFrame
                ):
                    st.subheader("Classification Report")
                    st.dataframe(
                        ml_result["classification_report"],
                        use_container_width=True
                    )

                if isinstance(
                    ml_result.get("predictions_table"),
                    pd.DataFrame
                ):
                    st.subheader("Sample Predictions")
                    st.dataframe(
                        ml_result["predictions_table"],
                        use_container_width=True
                    )

                if isinstance(
                    ml_result.get("feature_importance"),
                    pd.DataFrame
                ):
                    st.subheader("Top Feature Importance")
                    st.dataframe(
                        ml_result["feature_importance"],
                        use_container_width=True
                    )

            elif ml_result.get("type") == "ml_comparison":
                st.subheader("Model Comparison")
                st.dataframe(
                    ml_result["comparison_table"],
                    use_container_width=True
                )


# ============================================================
# PAGE: SETTINGS
# ============================================================

elif current_page == "⚙️ Settings":

    st.header("⚙️ Settings")

    st.text_input(
        "Groq model",
        key="llm_model_name",
        help=(
            "Default: openai/gpt-oss-20b. Change this only to a "
            "model available in your Groq account."
        )
    )

    st.selectbox(
        "Theme",
        options=[
            "Auto",
            "Light",
            "Dark"
        ],
        key="theme_choice"
    )

    st.checkbox(
        "Cap outliers by default",
        key="default_cap_outliers",
        help=(
            "Outliers are always detected. This setting only changes "
            "the default state of IQR capping on the Data Cleaning page."
        )
    )

    st.caption(
        "Your Groq API key should stay in .streamlit/secrets.toml and "
        "should not be committed to GitHub."
    )


# ============================================================
# PAGE: ABOUT PROJECT
# Replaces the previous creator section.
# ============================================================

elif current_page == "ℹ️ About Project":

    st.header("ℹ️ About DataTalk")

    st.markdown(
        """
        **DataTalk: Conversational Data Analysis & Auto-Visualization System**
        is designed to help users explore datasets without needing to write
        Python, SQL, or BI queries manually.

        ### Project Goal
        A user can upload a dataset, clean it, explore the data, ask questions
        in natural language, generate visualizations, and optionally train
        machine-learning models from one Streamlit application.

        ### Main Capabilities
        - CSV and XLSX dataset upload
        - Dataset preview and metadata
        - Missing-value handling
        - Duplicate removal
        - Date-column detection
        - IQR outlier detection and optional capping
        - Numerical and categorical EDA
        - Correlation analysis and distributions
        - LLM-powered conversational analytics
        - Row-based questions such as sorting and Top-N
        - Standard deviation and year-over-year analysis
        - Automatic chart selection from natural language
        - Manual visualization builder
        - Downloadable charts
        - Classification and regression model training/comparison

        ### Recommended Workflow
        **Chat upload → Data Cleaning → Dataset Preview / EDA → Chat analysis →
        Auto Visualizations or Machine Learning as needed.**

        ### Technology
        Python, Streamlit, Pandas, NumPy, Matplotlib, Seaborn,
        scikit-learn and Groq LLM integration.
        """
    )
