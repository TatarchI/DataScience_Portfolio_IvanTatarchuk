"""
📦 Module: Data Normalization for Scoring Model
Author: Ivan Tatarchuk

This module:
- Performs Voronin-style age transformation for peak scoring at ~40 years old.
- Selects criteria for multicriteria analysis (min/max logic).
- Parses weights and normalizes the dataset based on min/max scaling.
- Outputs a NumPy matrix ready for analytical modeling.
"""

import pandas as pd
import numpy as np


def normalization(d_segment_sample_cleaning):
    """
    Normalize the input dataset according to min/max strategy with weight support.

    Args:
        d_segment_sample_cleaning (pd.DataFrame): Cleaned dataset with features.

    Returns:
        tuple: (normalized_matrix, criteria_description, raw_values, weights)
    """

    # ----------------------------------- II. BUILDING THE SCORING MODEL ------------------------------

    # 2.1.0 Age transformation for Voronin model:
    # Peak score at age ≈ 40 → allows use as "max" criterion
    age_opt = 40  # optimal age empirically used in banking
    age_min = d_segment_sample_cleaning['age'].min()
    age_max = d_segment_sample_cleaning['age'].max()

    d_segment_sample_cleaning['age_transformed'] = 1 - abs(d_segment_sample_cleaning['age'] - age_opt) / (
        age_max - age_min)
    d_segment_sample_cleaning['age_transformed'] = d_segment_sample_cleaning['age_transformed'].clip(0, 1)

    print('✅ age_transformed generated — used as "max" criterion in Voronin model')

    # 2.1 Parse scoring indicators with Minimax logic
    d_data_description_minimax = pd.read_excel('../check_files/d_segment_data_description_cleaning_minimax.xlsx')

    d_segment_data_description_minimax = d_data_description_minimax.loc[
        (d_data_description_minimax['Minimax'] == 'min') |
        (d_data_description_minimax['Minimax'] == 'max')]
    d_segment_data_description_minimax.index = range(0, len(d_segment_data_description_minimax))

    print('📊 Selected Min/Max Criteria Description:')
    print(d_segment_data_description_minimax)

    # Read weights (with comma-to-dot conversion)
    raw_weights = d_segment_data_description_minimax['Weights (sum=1.0)'].astype(str).str.replace(',', '.')
    weights = raw_weights.astype(float).to_numpy()

    print('⚖ Raw Weights from Excel:')
    print(weights)

    # Save filtered description
    d_segment_data_description_minimax.to_excel('../check_files/d_segment_data_description_minimax.xlsx', index=False)

    # Extract relevant columns from cleaned data
    d = d_segment_data_description_minimax['Field_in_data']
    cols = d.values.tolist()
    d_segment_sample_minimax = d_segment_sample_cleaning[cols]

    print('📄 Selected Feature Columns:')
    print(cols)
    print(d_segment_sample_minimax)

    d_segment_sample_minimax.to_excel('../check_files/d_segment_sample_minimax.xlsx')

    # 2.2 Calculate min/max values for each criterion
    d_segment_sample_min = d_segment_sample_minimax[cols].min()
    d_segment_sample_max = d_segment_sample_minimax[cols].max()

    print('📉 Min values per column:')
    print(d_segment_sample_min)
    print('📈 Max values per column:')
    print(d_segment_sample_max)

    # 2.3 Normalize values into matrix
    m = d_segment_sample_minimax['loan_amount'].size
    n = d_segment_data_description_minimax['Field_in_data'].size
    d_segment_sample_minimax_Normal = np.zeros((m, n))  # Final normalized matrix

    delta_d = 0.3  # buffer coefficient for stability

    for j in range(0, len(d_segment_data_description_minimax)):
        columns_d = d_segment_data_description_minimax['Minimax'][j]
        if columns_d == 'min':
            columns_m = d_segment_data_description_minimax['Field_in_data'][j]
            for i in range(0, len(d_segment_sample_minimax)):
                max_max = d_segment_sample_max[j] + (2 * delta_d)
                d_segment_sample_minimax_Normal[i, j] = (delta_d + d_segment_sample_minimax[columns_m][i]) / (max_max)
        else:
            for i in range(0, len(d_segment_sample_minimax)):
                min_min = d_segment_sample_max[j] + (2 * delta_d)
                d_segment_sample_minimax_Normal[i, j] = (1 / (delta_d + d_segment_sample_minimax[columns_m][i])) / (
                    min_min)

    print('✅ Normalized Minimax Matrix (Preview):')
    print(d_segment_sample_minimax_Normal)

    np.savetxt('../check_files/d_segment_sample_minimax_Normal.txt', d_segment_sample_minimax_Normal)

    # -------------------------- Matrix Description --------------------------
    '''
    d_segment_sample_minimax_Normal[i, j]:
    m = number of applicants (rows)
    n = number of selected scoring indicators (columns)
    Normalized to 0–1 range per min/max rule, ready for multi-criteria scoring.
    '''

    return d_segment_sample_minimax_Normal, d_segment_data_description_minimax, d_segment_sample_minimax, weights