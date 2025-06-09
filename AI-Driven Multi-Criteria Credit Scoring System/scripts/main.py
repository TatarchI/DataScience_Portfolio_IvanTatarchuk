"""
🏁 Entry Point — Risk Scoring Pipeline Controller

This script coordinates the full pipeline:
- Parsing and preprocessing client data
- Performing multi-criteria scoring using the Voronin model
- Generating synthetic labels
- Training or evaluating a neural network model on the labeled data

Author: Ivan Tatarchuk

Package                      Version
---------------------------- -----------
pip                          25.0.1
matplotlib                   3.7.5
pandas                       2.0.3
numpy                        1.24.3
scikit-learn                 1.3.2
tensorflow                   2.13.0
"""

import os
import json
import warnings

from data_parsing import parsing
from data_normalization import normalization
from multi_criteria_model import Voronin
from neurons_network_model import optimize_model_nn, train_model_nn

warnings.filterwarnings("ignore", category=UserWarning)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__':
    d_segment_sample_cleaning = parsing('../original_dataset/sample_data.xlsx',
                                        '../original_dataset/data_description.xlsx') # data parsing

    (d_segment_sample_minimax_Normal, d_segment_data_description_minimax,
     d_segment_sample_minimax, weights) = normalization(d_segment_sample_cleaning)  # data normalization

    # Multi-criteria scoring for all clients, including fraud suspects
    Integro = Voronin(d_segment_sample_minimax_Normal, d_segment_data_description_minimax,
                      d_segment_sample_minimax, weights)

    # Neural Network setup
    X = d_segment_sample_minimax_Normal
    scor_array = Integro

    # User chooses execution mode
    print("\n⚠️ Warning: Full optimization may take 10–15 minutes. Recommended for first-time only.\n")
    mode_input = input("Enter mode (1 — optimization, 2 — use best model): ")
    mode_nn = int(mode_input.strip())

    if mode_nn == 1:
        print("Hyperparameter optimization in progress.")
        optimize_model_nn(X, scor_array)

    elif mode_nn == 2:
        print("🚀 Using best saved model (Mode 2)")

        # Load saved best configuration
        with open("../saved_model/model_nn_best_config.json", "r") as f:
            config_best = json.load(f)

        # Use pre-trained model from file
        train_model_nn(X, scor_array, config=config_best,
                       mode=2, model_path="../saved_model/model_nn_best.h5")



'''
## ✅ Analytical Summary & Model Validation
--------------------------------------------------------------------

### 1. Data Preprocessing and Feature Engineering  
During preprocessing, we focused not only on removing service or duplicate columns but also on proper type conversion 
and handling of missing values. Instead of dropping rows with missing data, we attempted smart imputation where 
feasible — for example, based on the year of birth. This allowed us to retain more records. Additionally, we optimized 
the `age` feature to be suitable for multi-criteria analysis: we treated 40 years as the optimal value (score = 1), 
with both younger and older ages being penalized.

### 2. Weight Integration and Voronin Scoring Model  
The Voronin model was adapted to support feature weights, which were read from an external Excel file.  
This enabled an expert-based assessment of each feature’s importance.  
The model is highly sensitive to features whose normalized values approach 1.0 — such features have the strongest 
influence. Only well-calibrated weights can shift the final outcome.

### 3. Label Generation and Risk Clustering  
Due to the absence of ground truth labels, we generated synthetic targets based on the Voronin model:  
- A 4-class version:  
  `0 = rejected`, `1 = low-risk loan`, `2 = medium-risk loan`, `3 = suspected fraud`.  
- A binary version:  
  `0 = reject`, `1 = approve`.  
This approach allowed us to apply supervised learning without relying on real default labels.  
Additionally, we visualized risk clusters based on scoring values, which enabled intuitive profiling of borrower 
groups with similar risk profiles.

### 4. Top-5 Risk Pattern Analysis  
We performed a detailed review of the five clients with the highest integrated scores. Findings included:  
- Most had a combination of high-risk indicators, such as large loan size, short term, low income, or irregular 
employment.  
- It is the **combination** of such factors (rather than isolated flags) that indicates elevated fraud potential.  
- Clients with multi-dimensional risk factors are the true challenge — and the primary target of this scoring system, 
as they often bypass standard rule-based filters.

### 5. Neural Network Trained on Heuristic Labels  
We built a complete TensorFlow neural model and performed grid search across key hyperparameters  
(neurons, batch size, epochs, optimizer, activation, dropout, learning rate).  
The final model achieved 98% accuracy and ROC AUC = 1.0.  
While the confusion matrix showed some misclassifications in class 2, the model **correctly ranked the probabilities**, 
showing stable behavior. The high AUC can be explained by the fact that even when misclassified, the predicted 
probabilities are near decision thresholds — especially for neighboring classes (2 and 3).  
This shows that even with heuristic labeling, a well-tuned neural net can effectively learn risk patterns and 
should be considered for future scoring automation.

### 6. Code Structure, Modularity, and Scalability  
We redesigned the entire project architecture with modularity and scalability in mind:  
- Separate scripts were created for each functional layer:  
  - `data_parsing.py` — parsing and cleaning input data  
  - `data_normalization.py` — normalization based on criterion types  
  - `scoring_labeling.py` — score calculation and label generation  
  - `multi_criteria_model.py` — Voronin model with weights and visualization  
  - `neurons_network_model.py` — model building, training, and evaluation  
  - `main.py` — central execution control  
- All files are logically organized into directories: `scripts`, `original_dataset`, `check_files`, `saved_model`  
- A dedicated `Mode 2` enables inference from a pre-trained model without repeating the time-consuming
optimization step. This makes the system easily integratable as a CRM module or a plug-in to enterprise 
credit analytics workflows.
'''