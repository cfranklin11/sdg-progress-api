import os

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.experimental import (
    enable_iterative_imputer,
)  # pylint: disable=unused-import
from sklearn.impute import IterativeImputer
import joblib

from src import country_data
from src.settings import BASE_DIR


LABELS = [
    "neonatal_mortality_rate",
    "u5_mortality_rate",
    "maternal_mortality_rate",
    "modern_contraceptive_rate",
    "adolescent_fertility_rate",
    "safely_managed_water_use_rate",
]
FEATURES = [
    "country",
    "year",
    "defence_budget",
    "economic_affairs_budget",
    "education_budget",
    "environment_protection_budget",
    "general_public_services_budget",
    "health_budget",
    "housing_and_community_amenities_budget",
    "public_order_and_safety_budget",
    "recreation_culture_and_religion_budget",
    "social_protection_budget",
    "gross_debt",
    "total_expenditure",
    "total_revenue",
    "gini_index",
    "population",
]
START_TEST_YEAR = 2017


def prepare_data(df):
    prepared_df = df.reset_index().drop("code", axis=1)

    features = prepared_df[FEATURES]
    labels = prepared_df[LABELS]

    return features, labels


def split_data(X, y):
    in_test = X["year"] >= START_TEST_YEAR

    y_train = y[~in_test]
    y_test = y[in_test]

    # There are a whole bunch of missing label data for some countries such that we
    # can't even forward- or backfill them. Making up your own labels is bad, but
    # we need some form of data and don't have time to find sources for what's missing
    imputer = IterativeImputer(max_iter=100)
    imputer.fit(y_train)

    return (
        X[~in_test],
        X[in_test],
        imputer.transform(y_train),
        imputer.transform(y_test),
    )


def pipeline():
    pipeline_steps = [
        ColumnTransformer(
            [
                (
                    "onehotencoder",
                    OneHotEncoder(sparse=False, handle_unknown="ignore"),
                    ["country"],
                )
            ],
            remainder=StandardScaler(),
        ),
        ExtraTreesRegressor(n_estimators=100),
    ]

    return make_pipeline(*pipeline_steps)


def save_model():
    df = country_data.load_combined()
    X, y = prepare_data(df)

    X_train, _X_test, y_train, _y_test = split_data(X, y)
    ml_pipeline = pipeline()
    ml_pipeline.fit(X_train, y_train)

    joblib.dump(ml_pipeline, os.path.join(BASE_DIR, "src/ml_model.pkl"))


def load_model():
    return joblib.load(os.path.join(BASE_DIR, "src/ml_model.pkl"))
