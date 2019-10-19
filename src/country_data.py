import os
import re

import pandas as pd
import numpy as np

from .settings import BASE_DIR

INDEX_COLS = ["code", "country", "year"]


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
        .set_index(INDEX_COLS)
    )


def _unicef_data(filepath, value_label):
    UNICEF_COL_MAP = {
        "ISO Code": "code",
        "Country Name": "country",
        "variable": "year",
        "Uncertainty bounds*": "uncertainty",
    }
    UNICEF_ID_COLS = ["ISO Code", "Country Name", "Uncertainty bounds*"]
    UNICEF_SHEET = "Country estimates"
    UNICEF_HEADER_IDX = 11
    UNICEF_YEAR_RANGE = (1950.5, 2019)

    return (
        pd.read_excel(filepath, sheet_name=UNICEF_SHEET, header=UNICEF_HEADER_IDX)
        .melt(id_vars=UNICEF_ID_COLS, value_vars=np.arange(*UNICEF_YEAR_RANGE))
        .dropna(subset=UNICEF_ID_COLS)
        .rename(columns={**UNICEF_COL_MAP, **{"value": value_label}})
        # Using 'Median' uncertainty across the board, because might as well hew
        # to the middle-ground
        .query('uncertainty == "Median"')
        # Years all have .5 added to them
        .assign(year=lambda df: df["year"].astype(int))
        .set_index(INDEX_COLS)
        .drop("uncertainty", axis=1)
    )


def neonatal_mortality():
    NN_MORT_FILEPATH = os.path.join(
        BASE_DIR, "data/health_well_being/child_mortality/NMR_mortality_rate_2019.xlsx"
    )
    return _unicef_data(NN_MORT_FILEPATH, "neonatal_mortality_rate")


def u5_mortality():
    U5_MORT_FILEPATH = os.path.join(
        BASE_DIR,
        "data/health_well_being/child_mortality/U5MR_mortality_rate_2019-1.xlsx",
    )
    return _unicef_data(U5_MORT_FILEPATH, "u5_mortality_rate")


def maternal_mortality():
    MATERNAL_MORT_FILEPATH = os.path.join(
        BASE_DIR,
        "data/health_well_being/maternal_mortality/maternal_mortality/countryresults_all.csv",
    )
    COL_NAME_MAP = {
        "name": "country",
        "iso": "code",
        "value": "maternal_mortality_rate",
    }
    UNUSED_COLS = ["estimate", "rounded", "indicator"]

    return (
        pd.read_csv(MATERNAL_MORT_FILEPATH)
        .query('estimate == "point estimate" & indicator == "mmr" & rounded == False')
        .rename(columns=COL_NAME_MAP)
        .drop(UNUSED_COLS, axis=1)
        .set_index(INDEX_COLS)
    )
