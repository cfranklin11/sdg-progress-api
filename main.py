import os
import sys
import json

import pandas as pd


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
        "environment_protection_budget": "environmentProtectionBudget",
        "general_public_services_budget": "generalPublicServicesBudget",
        "health_budget": "healthBudget",
        "housing_and_community_amenities_budget": "housingAndCommunityAmenitiesBudget",
        "public_order_and_safety_budget": "publicOrderAndSafetyBudget",
        "recreation_culture_and_religion_budget": "recreationCultureAndReligionBudget",
        "social_protection_budget": "socialProtectionBudget",
        "education_budget": "educationBudget",
        "economic_affairs_budget": "economicAffairsBudget",
        "defence_budget": "defenceBudget",
        "total_expenditure": "totalBudget",
        "neonatal_mortality_rate": "neonatalMortalityRate",
        "u5_mortality_rate": "u5MortalityRate",
        "maternal_mortality_rate": "maternalMortalityRate",
        "modern_contraceptive_rate": "modernContraceptiveRate",
        "adolescent_fertility_rate": "adolescentFertilityRate",
        "safely_managed_water_use_rate": "safelyManagedWaterUseRate",
    }

    return JSON_NAMES_MAP.get(col_name) or col_name


def _prepare_response(df):
    return (
        df.fillna("")
        .rename(columns=_clean_col_names)
        .filter(regex="Budget$|Rate$|^country")
        .to_dict("records")
    )


def country_data(_request):
    data = (
        countries.load_combined()
        .loc[(slice(None), 2018), :]
        .reset_index()
        .pipe(_prepare_response)
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
    model = ml_model.load_model()
    y_pred = model.predict(X_test)

    predictions = pd.DataFrame(y_pred, columns=ml_model.LABELS).pipe(_prepare_response)

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

    print(country_data({}))
