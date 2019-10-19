import os
import sys
import json

import pandas as pd
import numpy as np

BASE_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)))

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)


def _clean_col_names(col_name):
    JSON_NAMES_MAP = {
        "Code": "countryCode",
        "Country": "country",
        "Time": "year",
        "economic affairs": "economicAffairs",
        "environment protection": "environmentProtection",
        "general public services": "generalPublicServices",
        "housing and community amenities": "housingAndCommunityAmenities",
        "public order and safety": "publicOrderAndSafety",
        "recreation, culture and religion": "recreationCultureAndReligion",
        "social protection": "socialProtection",
        "Social benefits and social transfers in kind": "socialBenefits",
        "Gross debt of general government": "totalDebt",
        "Total expenditure of general government": "totalBudget",
        "Total general government (GG) revenue": "totalRevenue",
    }

    trimmed_name = col_name.replace(
        "General government expenditure by function, ", ""
    ).replace(", percentage of GDP", "")

    return JSON_NAMES_MAP.get(trimmed_name) or trimmed_name


def country_budgets(_request):
    OECD_FILEPATH = os.path.join(
        BASE_DIR, "data/country_stats/oecd/NAAG_13102019054548637.csv"
    )

    country_df = (
        pd.read_csv(OECD_FILEPATH)
        .drop(
            [
                "TIME",
                "INDICATOR",
                "Unit Code",
                "Unit",
                "PowerCode Code",
                "PowerCode",
                "Reference Period Code",
                "Reference Period",
                "Flag Codes",
                "Flags",
            ],
            axis=1,
        )
        .rename(columns={"LOCATION": "Code"})
        .pivot_table(
            index=["Code", "Country", "Time"], columns=["Indicator"], values=["Value"]
        )
        .droplevel(level=0, axis=1)
        .groupby(level=0)
        # Many countries are missing budget data for recent years
        # (I'm looking at you, Australia), so using forward fill to have a reasonable
        # approximation to match up with the SDG data
        .ffill()
        # Social benefits seems to double-up with other budgets,
        # because removing it gets the budget sum close to the total budget figures
        .drop(
            ["Social benefits and social transfers in kind, percentage of GDP"], axis=1
        )
        # A few countries (namely, South Africa) don't have any budget data,
        # so we'll just drop them
        .dropna()
    )

    gdp = country_df["Gross domestic product (GDP), current PPPs, billions US dollars"]
    budgets = country_df.drop(
        "Gross domestic product (GDP), current PPPs, billions US dollars", axis=1
    )

    budgets_dollar = (
        budgets.apply(lambda col: np.asarray(col) * np.asarray(gdp))
        .round(2)
        .loc[(slice(None), slice(None), 2018), :]
        .reset_index()
        .rename(columns=_clean_col_names)
        .to_dict("records")
    )

    return json.dumps({"data": budgets_dollar})


if __name__ == "__main__":
    country_budgets({})
