import os
import sys
import json

import pandas as pd
from sklearn.externals import joblib


BASE_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)))

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)


from src import country_data as countries
from src import ml_model


def _clean_col_names(col_name):
    JSON_NAMES_MAP = {
        "code": "countryCode",
        "country": "country",
        "time": "year",
        "economic_affairs": "economicAffairs",
        "environment_protection": "environmentProtection",
        "general_public_services": "generalPublicServices",
        "housing_and_community_amenities": "housingAndCommunityAmenities",
        "public_order_and_safety": "publicOrderAndSafety",
        "recreation_culture_and_religion": "recreationCultureAndReligion",
        "social_protection": "socialProtection",
        "social_benefits_and_social_transfers_in_kind": "socialBenefits",
        "gross_debt": "totalDebt",
        "total_expenditure": "totalBudget",
        "total_revenue": "totalRevenue",
        "neonatal_mortality_rate": "neonatalMortalityRate",
        "u5_mortality_rate": "u5MortalityRate",
        "maternal_mortality_rate": "maternalMortalityRate",
        "modern_contraceptive_rate": "modernContraceptiveRate",
        "adolescent_fertility_rate": "adolescentFertilityRate",
        "safely_managed_water_use_rate": "safelyManagedWaterUseRate",
    }

    return JSON_NAMES_MAP.get(col_name) or col_name


def country_data(_request):
    data = (
        countries.load_combined()
        .loc[(slice(None), 2018), :]
        .reset_index()
        .rename(columns=_clean_col_names)
        .fillna("")
        .to_dict("records")
    )

    return json.dumps({"data": data})


def sdg_predictions(request):
    DEFAULT_PARAMS = {"year": 2020}
    REQUIRED_PARAMS = [
        "country",
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
    ]

    params = request.get_json()

    country = params["country"]  # pylint: disable=unused-variable

    features, _ = ml_model.prepare_data(countries.load_combined())
    default_data = features.query("country == @country").iloc[-1, :].to_dict()
    param_data = {param: params[param] for param in REQUIRED_PARAMS}

    X_test = pd.DataFrame([{**default_data, **DEFAULT_PARAMS, **param_data}])
    model = joblib.load(os.path.join(BASE_DIR, "src/ml_model.pkl"))
    y_pred = model.predict(X_test)

    predictions = (
        pd.DataFrame(y_pred, columns=ml_model.LABELS).fillna("").to_dict("records")
    )

    return {"data": predictions}


if __name__ == "__main__":
    test_params = {
        "country": "Australia",
        "defence_budget": 810.0,
        "economic_affairs_budget": 2305.06,
        "education_budget": 2703.66,
        "environment_protection_budget": 278.31,
        "general_public_services_budget": 2594.59,
        "health_budget": 3054.12,
        "housing_and_community_amenities_budget": 420.02,
        "public_order_and_safety_budget": 828.84,
        "recreation_culture_and_religion_budget": 468.08,
        "social_protection_budget": 5515.92,
    }

    print(sdg_predictions(test_params))
