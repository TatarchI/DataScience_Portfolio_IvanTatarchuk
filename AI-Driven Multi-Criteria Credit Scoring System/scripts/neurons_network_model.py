"""
🧠 Module: Neural Network Model
Author: Ivan Tatarchuk

This module implements a flexible Keras-based feedforward neural network to classify loan applicants
into risk categories based on SCOR labels. Includes model building, evaluation, training, and hyperparameter tuning.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import itertools
from time import time
from tabulate import tabulate

from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop, Nadam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve

from scoring_labeling import get_y_multi

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings


def build_model(config, input_dim, output_dim=4):
    """
    Construct a flexible feedforward neural network based on the given configuration.

    Args:
        config (dict): Model parameters: lyrs, activation, dropout, optimizer, lr
        input_dim (int): Feature dimensionality
        output_dim (int): Number of output classes (default: 4)

    Returns:
        keras.Model: Compiled Keras model
    """
    model = Sequential()
    lyrs = config.get('lyrs', [32, 16])
    activation = config.get('activation', 'relu')
    dropout = config.get('dropout', 0.2)
    optimizer_name = config.get('optimizer', 'adam')
    lr = config.get('lr', 0.001)

    model.add(Dense(lyrs[0], input_dim=input_dim, activation=activation))
    if dropout > 0:
        model.add(Dropout(dropout))

    for units in lyrs[1:]:
        model.add(Dense(units, activation=activation))
        if dropout > 0:
            model.add(Dropout(dropout))

    model.add(Dense(output_dim, activation='softmax'))

    optimizers = {'adam': Adam, 'rmsprop': RMSprop, 'nadam': Nadam}
    optimizer_cls = optimizers.get(optimizer_name, Adam)
    optimizer = optimizer_cls(learning_rate=lr)

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance and visualize ROC curves per class.

    Args:
        model (keras.Model): Trained or loaded model
        X_test (np.ndarray): Test features
        y_test (np.ndarray): True class labels
    """
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)

    acc = accuracy_score(y_test, y_pred_classes)
    auc = roc_auc_score(y_test, y_pred, average='macro', multi_class='ovo')
    cm = confusion_matrix(y_test, y_pred_classes)

    print("🎯 Accuracy (Test):", round(acc, 4))
    print("📉 Loss (Test):", round(test_loss, 4))
    print("📈 ROC AUC (macro, OVO):", round(auc, 4))
    print("🧩 Confusion matrix:\n", cm)

    # ROC Curve (1-vs-rest)
    plt.figure(figsize=(6, 5))
    for i in range(y_pred.shape[1]):
        fpr, tpr, _ = roc_curve((y_test == i).astype(int), y_pred[:, i])
        plt.plot(fpr, tpr, label=f"Class {i}")
    plt.plot([0, 1], [0, 1], linestyle="--", color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves by Class")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def train_model_nn(X, scor_array, config, mode=1, model_path="../saved_model/model_nn_best.h5", verbose=0):
    """
    Train or load a neural network model depending on mode.

    Args:
        X (np.ndarray): Feature matrix
        scor_array (np.ndarray): SCOR values
        config (dict): Neural network configuration
        mode (int): 1 — train, 2 — load & evaluate
        model_path (str): Path to save/load the model
        verbose (int): Keras verbosity

    Returns:
        Tuple with evaluation results or None
    """
    y_class = get_y_multi(scor_array)

    X_temp, X_test, y_temp, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

    if mode == 1:
        model = build_model(config, input_dim=X.shape[1])
        history = model.fit(
            X_train, y_train,
            epochs=config.get("epochs", 50),
            batch_size=config.get("batch_size", 32),
            validation_data=(X_val, y_val),
            verbose=verbose
        )

        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        val_acc = history.history['val_accuracy'][-1]
        val_loss = history.history['val_loss'][-1]

        y_pred = model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)

        acc = accuracy_score(y_test, y_pred_classes)
        auc = roc_auc_score(y_test, y_pred, average='macro', multi_class='ovo')

        return acc, auc, val_acc, val_loss, test_loss, config, model

    elif mode == 2:
        if not os.path.exists(model_path):
            print(f"❌ Model not found at {model_path}")
            return
        model = load_model(model_path)

        y_class = get_y_multi(scor_array)
        X_temp, X_test, y_temp, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)

        evaluate_model(model, X_test, y_test)

def optimize_model_nn(X, scor_array):
    """
    Perform exhaustive hyperparameter search for the best neural network configuration.

    Args:
        X (np.ndarray): Normalized feature matrix
        scor_array (np.ndarray): Integrated SCOR scores
    """
    param_grid = {
        'lyrs': [[128, 64, 32, 16], [64, 32, 16], [32, 16, 8]],
        'activation': ['relu', 'tanh'],
        'dropout': [0.0, 0.1],
        'optimizer': ['Adam', 'Nadam'],
        'lr': [0.001, 0.0005],
        'batch_size': [16, 32],
        'epochs': [100]
    }

    keys = list(param_grid.keys())
    combinations = list(itertools.product(*(param_grid[k] for k in keys)))

    results = []
    best_auc = -1
    best_config = None

    print(f"🔍 Running {len(combinations)} hyperparameter combinations...\n")

    for i, values in enumerate(combinations):
        config = dict(zip(keys, values))
        start = time()
        acc, auc, val_acc, val_loss, test_loss, cfg, trained_model = train_model_nn(
            X, scor_array, config, mode=1, model_path="../saved_model/model_nn_best.h5", verbose=0)
        end = time()

        elapsed = round(end - start, 2)
        print(
            f"#{i + 1:02d} | acc: {acc:.4f} | val_acc: {val_acc:.4f} | val_loss: {val_loss:.4f} | "
            f"test_loss: {test_loss:.4f} | auc: {auc:.4f} | ⏱ {elapsed}s")

        results.append({
            '№': i + 1,
            'accuracy': round(acc, 4),
            'val_acc': round(val_acc, 4),
            'val_loss': round(val_loss, 4),
            'test_loss': round(test_loss, 4),
            'roc_auc': round(auc, 4),
            'config': cfg
        })

        if auc > best_auc:
            best_auc = auc
            best_config = cfg
            os.makedirs("../saved_model", exist_ok=True)
            trained_model.save("../saved_model/model_nn_best.h5")
            with open("../saved_model/model_nn_best_config.json", "w") as f:
                json.dump(cfg, f, indent=4)

    print("\n📊 Grid Search Results:")
    print(tabulate(results, headers='keys', tablefmt='github'))
    print("\n🏆 Best Configuration:")
    print(best_config)