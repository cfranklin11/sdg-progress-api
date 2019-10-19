import os

from sklearn.linear_model import ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.externals import joblib

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
START_TEST_YEAR = 2017


def prepare_data(df):
    # TODO: There is a whole bunch of missing data among the labels,
    # and filling with 0 is terrible, so, time permitting, we'll come back
    # and use a better imputation strategy
    prepared_df = df.fillna(0).reset_index().drop("code", axis=1)

    features = prepared_df.drop(LABELS, axis=1)
    labels = prepared_df[LABELS]

    return features, labels


def split_data(X, y):
    in_test = X["year"] >= START_TEST_YEAR

    return X[~in_test], X[in_test], y[~in_test], y[in_test]


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
        ElasticNet(),
    ]

    return make_pipeline(*pipeline_steps)


def save_model():
    df = country_data.load_combined()
    X, y = prepare_data(df)

    X_train, _X_test, y_train, _y_test = split_data(X, y)
    ml_pipeline = pipeline()
    ml_pipeline.fit(X_train, y_train)

    joblib.dump(ml_pipeline, os.path.join(BASE_DIR, "src/ml_model.pkl"))
