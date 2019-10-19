import os
import sys
import json


BASE_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)))

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)


from src import country_data


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
    }

    return JSON_NAMES_MAP.get(col_name) or col_name


def country_budgets(_request):
    data = (
        country_data.country_budgets.loc[(slice(None), slice(None), 2018), :]
        .reset_index()
        .rename(columns=_clean_col_names)
        .to_dict("records")
    )

    return json.dumps({"data": data})


if __name__ == "__main__":
    country_budgets({})
