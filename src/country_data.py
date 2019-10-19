import os
import re

import pandas as pd
import numpy as np

from .settings import BASE_DIR

INDEX_COLS = ["country", "year"]


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
    UNUSED_COLS = ["uncertainty", "code"]

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
        .drop(UNUSED_COLS, axis=1)
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
    UNUSED_COLS = ["estimate", "rounded", "indicator", "code"]

    return (
        pd.read_csv(MATERNAL_MORT_FILEPATH)
        .query('estimate == "point estimate" & indicator == "mmr" & rounded == False')
        .rename(columns=COL_NAME_MAP)
        .drop(UNUSED_COLS, axis=1)
        .set_index(INDEX_COLS)
    )


def _join_cols(col_names):
    filtered_col_names = [
        col
        for col in col_names
        if any(col) and "Unnamed" not in col and "DRINKING WATER" not in col
    ]

    if len(set(filtered_col_names)) == 1:
        return filtered_col_names[0].replace("\n", " ")

    return ": ".join(filtered_col_names).replace("\n", " ")


def modern_contraceptive_use_rate():
    FILEPATH = os.path.join(
        BASE_DIR,
        "data/health_well_being/family_planning/UNPD_WCU2019_Country_Data_Survey-Based.xlsx",
    )
    COL_MAP = {
        "Country or area": "country",
        "Survey start year": "year",
        "Age group": "age_group",
        "Contraceptive prevalence (per cent): Any modern method": "modern_contraceptive_rate",
    }

    df = pd.read_excel(FILEPATH, sheet_name="By methods", header=[3, 4])
    df.columns = [_join_cols(col_pair) for col_pair in df.columns.values]

    return (
        df.rename(columns=COL_MAP)
        .loc[:, list(COL_MAP.values())]
        # There are about 75 duplicates, using 'Survey end year' to fill
        # some duplicates reduces it to 53, but we'll just drop duplicates for now
        .drop_duplicates(subset=INDEX_COLS, keep="first")
        .drop("age_group", axis=1)
        .set_index(INDEX_COLS)
    )


def adolescent_fertility_rate():
    FILEPATH = os.path.join(
        BASE_DIR, "data/health_well_being/family_planning/UNPD_WFD_2017_FERTILITY.xlsx"
    )
    COL_MAP = {
        "Country or area": "country",
        "YearStart": "year",
        "AgeGroup": "age_group",
        # Fertility rate is per 1,000:
        # https://www.un.org/en/development/desa/population/publications/dataset/fertility/total-fertility.asp
        "DataValue": "adolescent_fertility_rate",
    }

    df = (
        pd.read_excel(FILEPATH, sheet_name="FERTILITY_INDICATORS", header=2)
        .rename(columns=COL_MAP)
        .query('age_group == "[15-19]"')
    )

    # Some country/year combos don't have YearStart or YearEnd values,
    # but they do have TimeMid values, so we'll use that
    years = df["year"].fillna(df["TimeMid"].round())

    return (
        df.assign(year=years)
        .set_index(INDEX_COLS)
        .loc[:, ["adolescent_fertility_rate"]]
    )


def safe_drinking_water():
    FILEPATH = os.path.join(BASE_DIR, "data/water_sanitation/JMP_2019_WLD.xlsx")
    COL_MAP = {
        "COUNTRY, AREA OR TERRITORY": "country",
        "NATIONAL: Proportion of population using  improved water supplies: Safely managed": "safely_managed_water_use_rate",  # pylint: disable=line-too-long
        "Year": "year",
    }

    drink_water_df = pd.read_excel(FILEPATH, sheet_name="Water", header=[0, 1, 2])

    drink_water_df.columns = [
        _join_cols(cols) for cols in drink_water_df.columns.values
    ]
    drink_water_df = drink_water_df.loc[:, ~drink_water_df.columns.duplicated()]

    return pd.to_numeric(
        drink_water_df.rename(columns=COL_MAP)
        .set_index(["country", "year"])
        .loc[:, "safely_managed_water_use_rate"]
        .dropna()
        .str.replace("<", "")
        .str.replace(">", ""),
        errors="coerce",
    )


def combined():
    return (
        country_budgets()
        .join(
            [
                neonatal_mortality(),
                u5_mortality(),
                maternal_mortality(),
                modern_contraceptive_use_rate(),
                adolescent_fertility_rate(),
                safe_drinking_water(),
            ],
            how="left",
        )
        .groupby("country")
        # We forwardfill & backfill to make sure not to leave NaNs at the start
        # of a country's data
        .ffill()
        .groupby("country")
        .bfill()
    )
