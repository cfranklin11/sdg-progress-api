from sklearn.linear_model import ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


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
