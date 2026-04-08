import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def add_features(df):
    df = df.copy()

    df["INSR_BEGIN"] = pd.to_datetime(
        df["INSR_BEGIN"],
        format="%d-%b-%y",
        errors="coerce"
    )

    df["INSR_END"] = pd.to_datetime(
        df["INSR_END"],
        format="%d-%b-%y",
        errors="coerce"
    )

    df["insurance_duration"] = (df["INSR_END"] - df["INSR_BEGIN"]).dt.days
    df["car_age"] = df["INSR_BEGIN"].dt.year - df["PROD_YEAR"]

    df = df.drop(columns=["INSR_BEGIN", "INSR_END"], errors="ignore")

    return df


def feature_engineering(df):
    return add_features(df)


def build_preprocessor(X):
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    return preprocessor