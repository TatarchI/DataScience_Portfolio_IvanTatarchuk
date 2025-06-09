"""
📦 Module: Data Parsing and Preparation
Author: Ivan Tatarchuk

This module performs preprocessing for the credit scoring pipeline:
- Parses raw Excel files with client applications and metadata.
- Verifies structure and handles missing values.
- Performs feature engineering (age, address duration, employment duration).
- Cleans up pseudo-values and drops noisy indicators.
"""

import pandas as pd
import numpy as np


def parsing(filename_sample_data, filename_data_description):
    """
    Parses and prepares the input dataset for credit scoring.

    Args:
        filename_sample_data (str): Path to Excel file with client application data.
        filename_data_description (str): Path to Excel file with indicator metadata.

    Returns:
        pd.DataFrame: Cleaned and feature-enriched DataFrame.
    """

    # 1.1 Load raw application data
    d_sample_data = pd.read_excel(filename_sample_data, parse_dates=['applied_at'])
    print('d_sample_data=', d_sample_data)  # Print entire DataFrame
    Title_d_sample_data = d_sample_data.columns  # Column names

    # 1.2 Analyze structure
    print('-------------  Column Names  -----------')
    print(Title_d_sample_data)
    print('---------  Column Data Types  -----------')
    print(d_sample_data.dtypes)
    print('---------  Missing Values Summary  ------')
    print(d_sample_data.isnull().sum())

    # 1.3 Load indicator metadata
    d_data_description = pd.read_excel(filename_data_description)
    print('---------------  d_data_description  ---------------')
    print('d_data_description=', d_data_description)
    print('----------------------------------------------------')

    # 1.4 Initial filtering of indicators (client or product-level)
    d_segment_data_description_client_bank = d_data_description[
        (d_data_description.Place_of_definition == 'Вказує позичальник') |
        (d_data_description.Place_of_definition == 'параметри, повязані з виданим продуктом')]
    # Reset index manually
    d_segment_data_description_client_bank.index = range(0,
                                                         len(d_segment_data_description_client_bank))
    print('---------  d_segment_data_description_client_bank  -----------')
    print('d_segment_data_description_client_bank=', d_segment_data_description_client_bank)
    print('----------------------------------------------------')

    # 1.5 Validation of matches
    print('-----------------------------------------')
    b = d_segment_data_description_client_bank['Field_in_data']  # column names to match

    if set(b).issubset(d_sample_data.columns):
        Flag_b = 'Flag_True'
    else:
        Flag_b = 'Flag_False'
    print('⚠️ Column match flag:', Flag_b)

    # ------ Count number of matches
    n_columns = d_segment_data_description_client_bank['Field_in_data'].size
    j = 0
    for i in range(0, n_columns):
        a = d_segment_data_description_client_bank['Field_in_data'][i]
        if set([a]).issubset(d_sample_data.columns):
            j = j + 1
    print('j = ', j)

    # ------ Indices of matching columns
    Columns_Flag_True = np.zeros((j))
    j = 0
    for i in range(0, n_columns):
        a = d_segment_data_description_client_bank['Field_in_data'][i]
        if set([a]).issubset(d_sample_data.columns):
            Columns_Flag_True[j] = i
            j = j + 1
    print('✔ Matching column indices:', Columns_Flag_True)

    # 1.5.2 Keep only matched indicators
    d_segment_data_description_client_bank_True = d_segment_data_description_client_bank.iloc[
        Columns_Flag_True]
    d_segment_data_description_client_bank_True.index = range(0,
                                                              len(d_segment_data_description_client_bank_True))
    print('------------ Filtered Indicator Metadata -------------')
    print(d_segment_data_description_client_bank_True)
    print('------------------------------------------------------')

    # 1.5.2 (extended) Feature Engineering
    print('--------- 1.5.2 (extended) – Feature Engineering -----------')

    # Age = application date - birth date
    d_sample_data['age'] = round((d_sample_data['applied_at'] - d_sample_data['birth_date']).dt.days / 365.25, 1)

    # Address duration: fill missing address start dates using mean per birth year
    d_sample_data['birth_year'] = d_sample_data['birth_date'].dt.year
    addr_mean_by_year = d_sample_data.groupby('birth_year')['fact_addr_start_date'].transform('mean')
    d_sample_data['fact_addr_start_date'] = d_sample_data['fact_addr_start_date'].fillna(addr_mean_by_year)
    d_sample_data['addr_duration_years'] = round((d_sample_data['applied_at'] -
                                                  d_sample_data['fact_addr_start_date']).dt.days / 365.25, 1)

    # Employment duration: same logic
    emp_mean_by_year = d_sample_data.groupby('birth_year')['employment_date'].transform('mean')
    d_sample_data['employment_date'] = d_sample_data['employment_date'].fillna(emp_mean_by_year)
    d_sample_data['employment_duration_years'] = round((d_sample_data['applied_at'] -
                                                        d_sample_data['employment_date']).dt.days / 365.25, 1)

    # Fix pseudo-values: 999 → default
    d_sample_data['organization_type_id'] = d_sample_data['organization_type_id'].replace(999, 5)
    d_sample_data['organization_branch_id'] = d_sample_data['organization_branch_id'].replace(999, 11)
    d_sample_data['income_frequency_id'] = d_sample_data['income_frequency_id'].replace(999, 3)
    d_sample_data['income_source_id'] = d_sample_data['income_source_id'].replace(999, 6)

    # 1.5.3 Final cleanup
    b = d_segment_data_description_client_bank_True['Field_in_data']
    d_segment_sample_data_client_bank = d_sample_data[b].copy()

    # Add engineered features
    d_segment_sample_data_client_bank['age'] = d_sample_data['age']
    d_segment_sample_data_client_bank['addr_duration_years'] = d_sample_data['addr_duration_years']
    d_segment_sample_data_client_bank['employment_duration_years'] = d_sample_data['employment_duration_years']

    # Drop temporary columns
    d_segment_sample_data_client_bank = d_segment_sample_data_client_bank.drop(
        columns=['birth_date', 'fact_addr_start_date', 'employment_date'])

    print('---- Missing values in final sample segment --------')
    print(d_segment_sample_data_client_bank.isnull().sum())
    print('----------------------------------------------------')

    # Drop poor indicators
    d_segment_data_description_cleaning = d_segment_data_description_client_bank_True.loc[
        d_segment_data_description_client_bank_True['Field_in_data'] != 'position_id']
    d_segment_data_description_cleaning = d_segment_data_description_cleaning.loc[
        d_segment_data_description_cleaning['Field_in_data'] != 'has_prior_employment']
    d_segment_data_description_cleaning = d_segment_data_description_cleaning.loc[
        d_segment_data_description_cleaning['Field_in_data'] != 'prior_employment_start_date']
    d_segment_data_description_cleaning = d_segment_data_description_cleaning.loc[
        d_segment_data_description_cleaning['Field_in_data'] != 'prior_employment_end_date']
    d_segment_data_description_cleaning = d_segment_data_description_cleaning.loc[
        d_segment_data_description_cleaning['Field_in_data'] != 'income_frequency_other']

    d_segment_data_description_cleaning.index = range(0,
                                                      len(d_segment_data_description_cleaning))
    d_segment_data_description_cleaning.to_excel(
        '../check_files/d_segment_data_description_cleaning.xlsx')

    # Drop same indicators from data
    d_segment_sample_cleaning = d_segment_sample_data_client_bank.drop(
        columns=['position_id', 'has_prior_employment', 'prior_employment_start_date',
                 'prior_employment_end_date', 'income_frequency_other'])

    d_segment_sample_cleaning.index = range(0, len(d_segment_sample_cleaning))
    d_segment_sample_cleaning.to_excel('../check_files/d_segment_sample_cleaning.xlsx')
    print('--- Missing values in cleaned indicator data ---')
    print(d_segment_sample_cleaning.isnull().sum())
    print('---------- Final scoring dataset -----------')
    print(d_segment_sample_cleaning)
    print('----------------- Final indicator metadata  ----------------')
    print(d_segment_data_description_cleaning)
    print('------------------------------------------------------------')

    return d_segment_sample_cleaning