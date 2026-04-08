from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from src.preprocessing.preprocess import build_preprocessor, feature_engineering
from sklearn.preprocessing import FunctionTransformer


def build_model(X, params):

    X_transformed = feature_engineering(X)
    preprocessor = build_preprocessor(X_transformed)

    rf = RandomForestRegressor(
        warm_start=True,
        n_estimators=params.get("n_estimators", 50)
    )

    model = Pipeline([
        ("features", FunctionTransformer(feature_engineering)),
        ("prep", preprocessor),
        ("model", rf)
    ])

    return model


def train_model(model, X, y, is_first_batch=False, add_trees=10):
    rf = model.named_steps["model"]
    prep = model.named_steps["prep"]
    feat = model.named_steps["features"]

    X_feat = feat.transform(X)

    if is_first_batch:
        X_transformed = prep.fit_transform(X_feat)
        rf.fit(X_transformed, y)
    else:
        X_transformed = prep.transform(X_feat)
        rf.n_estimators += add_trees
        rf.fit(X_transformed, y)
    return model