"""
📦 Module: Multi-Criteria Scoring Model (Voronin Method)
Author: Ivan Tatarchuk

Implements an interpretable scoring system using a modified Voronin-style multi-criteria formula
with weight-adjusted contributions and risk flagging. Includes two types of visualizations.

Outputs:
- Integro_Scor.txt: Full scoring results
- Integro_Scor_for_plot.txt: Clipped version for plotting
"""

import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------------------------------------

def Voronin(d_segment_sample_minimax_Normal, d_segment_data_description_minimax, d_segment_sample_minimax, weights):
    """
    Calculate integrated multi-criteria score using Voronin's method.

    The score is a sum of inversely transformed normalized criteria, weighted by domain knowledge.

    Two visualizations:
    - Full scoring graph
    - Clipped version (max 80) for visual interpretability

    Also includes a diagnosis of high-risk clients (possible fraud suspicion).

    Args:
        d_segment_sample_minimax_Normal (np.ndarray): Normalized input data [m x n]
        d_segment_data_description_minimax (pd.DataFrame): Description of selected criteria
        d_segment_sample_minimax (pd.DataFrame): Raw input values for same clients
        weights (np.ndarray): Array of criterion weights (sum ≈ 1.0)

    Returns:
        np.ndarray: Final scoring values (1D array)
    """

    n = d_segment_data_description_minimax['Field_in_data'].size
    k = d_segment_sample_minimax_Normal.shape[0]

    Integro = np.zeros(k)

    # --- Scoring Loop (Voronin formula) ---
    for i in range(k):
        Sum_Voronin = 0
        for j in range(n):
            Sum_Voronin += weights[j] * ((1 - d_segment_sample_minimax_Normal[i, j]) ** -1)
        Integro[i] = Sum_Voronin

    # --- Clip values for visual purposes ---
    Integro_vis = np.where(Integro > 80, 80, Integro)

    # --- Save to file ---
    np.savetxt('../check_files/Integro_Scor.txt', Integro)
    np.savetxt('../check_files/Integro_Scor_for_plot.txt', Integro_vis)

    # --- Top-5 high scorers diagnostic (risk detection) ---
    print("\n🔍 Top-5 Clients with Highest Scor (Potential Fraud Alert):")
    top_idx = np.argsort(Integro)[-5:][::-1]

    for idx in top_idx:
        print(f"\n▶ Client {idx} — Scor = {Integro[idx]:.2f}")
        suspicious_count = 0

        for j in range(n):
            norm_value = d_segment_sample_minimax_Normal[idx, j]
            if norm_value > 0.99:
                suspicious_count += 1
                feature_name = d_segment_data_description_minimax['Field_in_data'].iloc[j]
                real_value = d_segment_sample_minimax.iloc[idx][feature_name]
                contribution = weights[j] * ((1 - norm_value) ** -1)

                print(f"  ▶ Feature: {feature_name}")
                print(f"     Real Value = {real_value}")
                print(f"     Contribution to Scor = {contribution:.2f} (Weight {weights[j]:.3f})")

        if suspicious_count >= 3:
            print(f"⚠️ Alert: Client has {suspicious_count} indicators > 0.99 — High Fraud Risk")
        elif suspicious_count >= 1:
            print(f"⚠️ Warning: Client has {suspicious_count} indicators with high potential risk")
        else:
            print("ℹ️ No risk indicators detected.")

    # --- Visualization 1: Original Scoring ---
    plt.figure(figsize=(10, 4))
    plt.title("Multi-criteria Integrated Scor (Original)")
    plt.plot(Integro, 'b-', label='Scor')
    plt.axhline(y=20, color='orange', linestyle='--', label='Threshold')
    plt.xlabel("Client Index")
    plt.ylabel("Integrated Score")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Visualization 2: Clipped Scoring (max=80) ---
    plt.figure(figsize=(10, 4))
    plt.title("Multi-criteria Integrated Scor (Clipped max=80)")
    plt.plot(Integro_vis, 'b-', label='Scor (clipped)')
    plt.axhline(y=20, color='orange', linestyle='--', label='Threshold')
    plt.xlabel("Client Index")
    plt.ylabel("Integrated Score (clipped)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return Integro