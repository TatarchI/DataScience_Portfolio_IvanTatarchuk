"""
📦 Module: Label Generation
Author: Ivan Tatarchuk

This module builds both binary and multiclass target labels from integrated SCOR values,
allowing downstream supervised learning (e.g. classification via neural networks or gradient boosting).

It uses simple heuristics based on SCOR thresholds to define risk classes and outputs them as `.txt` files.
"""

import numpy as np


def load_scor(file_path='check_files/Integro_Scor.txt'):
    """
    Load the integrated SCOR array from a saved .txt file.

    Args:
        file_path (str): Path to the text file with SCOR values.

    Returns:
        np.ndarray: Array of SCOR values (float)
    """
    return np.loadtxt(file_path, dtype=float)


def get_y_binary(scor_array, threshold=20):
    """
    Generate a binary target variable from SCOR:
    0 — Reject, 1 — Approve

    Args:
        scor_array (np.ndarray): Integrated SCOR values
        threshold (float): Approval cutoff

    Returns:
        np.ndarray: Binary target array
    """
    return np.where(scor_array >= threshold, 1, 0)


def get_y_multi(scor_array):
    """
    Generate a multiclass target variable from SCOR:
    0 — Reject (<20)
    1 — Accept type 1 (20–29.99)
    2 — Accept type 2 (30–49.99)
    3 — Suspicious (≥50)

    Args:
        scor_array (np.ndarray): Integrated SCOR values

    Returns:
        np.ndarray: Multiclass target array
    """
    y_class = []
    for score in scor_array:
        if score < 20:
            y_class.append(0)
        elif score < 30:
            y_class.append(1)
        elif score < 50:
            y_class.append(2)
        else:
            y_class.append(3)
    return np.array(y_class)


def save_labels(y_bin, y_multi):
    """
    Save the generated labels to text files.

    Args:
        y_bin (np.ndarray): Binary labels
        y_multi (np.ndarray): Multiclass labels
    """
    np.savetxt("y_bin.txt", y_bin, fmt="%d")
    np.savetxt("y_multi.txt", y_multi, fmt="%d")


def generate_all_labels():
    """
    Full pipeline to:
    1. Load SCOR values
    2. Generate both binary and multiclass labels
    3. Save them to disk

    Returns:
        Tuple[np.ndarray, np.ndarray]: y_bin, y_multi
    """
    scor_array = load_scor()
    y_bin = get_y_binary(scor_array)
    y_multi = get_y_multi(scor_array)
    save_labels(y_bin, y_multi)
    print("✅ Labels successfully generated and saved.")
    return y_bin, y_multi