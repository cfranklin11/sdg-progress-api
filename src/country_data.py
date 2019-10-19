import os
import re

import pandas as pd
import numpy as np

from .settings import BASE_DIR


def _clean_col_names(col_name):
    cleaned_name = (
        col_name.replace("General government expenditure by function, ", "")
        .replace(", percentage of GDP", "")
        .replace(" of general government", "")
        .replace(",", "")
        .replace(" general government (GG)", "")
        .replace(" ", "_")
        .lower()
    )

    if re.search("gross|total", cleaned_name):
        return cleaned_name

    return cleaned_name + "_budget"


def country_budgets():
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

    return (
        budgets.apply(lambda col: np.asarray(col) * np.asarray(gdp))
        .round(2)
        .rename(columns=_clean_col_names)
        .reset_index()
        .rename(columns=lambda col: {"Time": "year"}.get(col) or col.lower())
        .set_index(["code", "country", "year"])
    )
