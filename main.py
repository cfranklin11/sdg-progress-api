import os
import sys
import json


BASE_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)))

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)


from src.country_data import combined


def _clean_col_names(col_name):
    JSON_NAMES_MAP = {
        "environment_protection_budget": "environmentProtection_budget",
        "general_public_services_budget": "generalPublicServicesBudget",
        "health_budget": "healthBudget",
        "housing_and_community_amenities_budget": "housingAndCommunityAmenitiesBudget",
        "public_order_and_safety_budget": "publicOrderAndSafetyBudget",
        "recreation_culture_and_religion_budget": "recreationCultureAndReligionBudget",
        "social_protection_budget": "socialProtectionBudget",
        "education_budget": "educationBudget",
        "economic_affairs_budget": "economicAffairsBudget",
        "defence_budget": "defenceBudget",
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
        combined()
        .loc[(slice(None), 2018), :]
        .reset_index()
        .rename(columns=_clean_col_names)
        .fillna("")
        .to_dict("records")
    )

    return json.dumps({"data": data})


if __name__ == "__main__":
    country_data({})
