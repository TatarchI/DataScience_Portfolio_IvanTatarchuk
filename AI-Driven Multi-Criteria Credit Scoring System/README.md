
# 🧠 Multi-Criteria Scoring System with Neural Network for Credit Risk Analysis

This project implements a full data science pipeline for building a credit scoring system that simulates a real-life scenario of detecting fraudulent clients and automating loan approvals. We integrate multi-criteria decision-making (MCDM) methods (Voronin model) and train a custom neural network on heuristic labels.

---

## 📌 Project Objectives

- Develop a scoring pipeline using a **completely unlabeled dataset**, ensuring that all logic is derived independently of any historical decisions or human bias.
- Avoid reliance on pre-existing labels to ensure **model autonomy** and **objectivity**, enabling fair risk assessment without inheriting past systemic patterns.
- Clean and parse raw client data from loan applications.
- Normalize features considering their nature (positive/negative influence).
- Implement Voronin MCDM model for scoring clients based on multiple risk factors.
- Generate heuristic binary and multiclass labels based on calculated scores.
- Train a supervised neural network to approximate expert scoring derived from MCDM.
- Visualize results, identify high-risk profiles, and assess model quality using confusion matrix and ROC AUC.

---

## 🧩 Dataset and Structure

- `Sample_data.xlsx`: Client loan applications with multiple features.
- `Data_description.xlsx`: Metadata with explanations and criterion polarity.

Directory structure:

```
Scoring_models/
├── original_dataset/
│   ├── sample_data.xlsx
│   └── data_description.xlsx
├── scripts/
│   ├── data_parsing.py
│   ├── data_normalization.py
│   ├── multi_criteria_model.py
│   ├── scoring_labeling.py
│   ├── neurons_network_model.py
│   └── main.py
├── check_files/
│   └── Outputs and intermediate results
├── saved_model/
│   └── model_nn_best.h5
├── visualizations/
│   └── scoring graphics + roc/auc (Neural network)
├── console_output (main w/o optimization block for NN)
├── console_output (optimization with gridsearch for NN)
├── README.md
```

---

## 🧪 Methodology & Workflow

### 1. **Data Parsing & Cleaning**

- Removal of irrelevant fields and columns with excessive missing values.
- Custom logic to impute missing values (e.g., age inference from birth year).
- Type casting and sanity checks.

### 2. **Normalization**

- Min-max scaling per feature.
- Handling negative/positive polarity using `Data_description.xlsx`.
- Weight loading from external Excel file (d_segment_data_description_cleaning_minimax in check_files).

### 3. **Voronin Multi-Criteria Model**

- Integrated scoring based on weighted normalized features.
- Features close to 1.0 have high impact.
- Final score stored in `Integro_Scor.txt` and clipped version for plotting.

### 4. **Label Generation**

- Binary label: `0 = reject`, `1 = approve`
- Multiclass label:
  - `0`: Reject (<20)
  - `1`: Low-risk approval (20–29.99)
  - `2`: Medium-risk approval (30–49.99)
  - `3`: Fraud suspicion (50+)

### 5. **Neural Network Model**

- Architecture tuning: grid search over neurons, batch size, dropout, etc.
- Mode 1: Full hyperparameter optimization (takes ~10-15 min)
- Mode 2: Load best saved model and run prediction
- ROC AUC: 1.0 on heuristic labels, accuracy ~98%

---

## 📊 Visual Outputs

- Top-5 clients with suspicious scores and breakdown by feature contributions.
- Integrated score visualizations before and after clipping.
- Confusion matrix and ROC curve for neural network classifier.

---

## 🎯 Business Value

- Label-Free Risk Segmentation: In the absence of historical ground truth or labeled outcomes, we use the Voronin multi-criteria model as a surrogate decision-making engine. This allows us to cluster clients by risk profiles based on feature importance and expert-weighted evaluation — a crucial capability during the early stages of projects with strict timelines or limited data availability.

- Automation: Once heuristic labels are generated through scoring, we train a neural network to replicate the decision boundaries, creating a fully automated scoring system that mimics expert logic without further human input.

- Fraud Detection: The scoring model flags clients with anomalous high-risk patterns, allowing early identification of fraud-prone profiles without needing legacy cases or labeled fraud examples.

- Scalability & Integration: The modular design enables integration into CRM or ERP systems with minimal adaptation. Labels, scores, and model outputs can easily be piped into other analytics or decision engines.

- Explainability & Trust: Unlike black-box scoring, the Voronin model offers transparent and interpretable reasoning for each score, helping business stakeholders understand why decisions are made.

- Adaptability: Applicable to domains beyond banking — including insurance underwriting, peer-to-peer lending platforms, leasing, or any sector where structured risk assessment is critical.

---

## 🛠 Technologies Used

| Tool / Library     | Purpose                         |
| ------------------ | ------------------------------- |
| Python 3.8+        | Core language                   |
| Pandas / NumPy     | Data manipulation               |
| Matplotlib         | Visualization                   |
| TensorFlow / Keras | Neural network training         |
| Scikit-learn       | Model evaluation (ROC, metrics) |

---

## ⚙️ How to Run

1. Clone the repo and install dependencies.
2. Place data files into `original_dataset/` folder.
3. Run `main.py` and choose mode:
   - `1`: Full model tuning
   - `2`: Use best pre-trained model

```bash
python scripts/main.py
```

Output logs and scores will be saved to `check_files/`.

---

## 📌 Future Improvements

- Integration of real labeled data: If ground truth becomes available in the future (e.g., actual loan defaults or repayment behavior), the pipeline can be extended to support reinforcement learning or continuous supervised retraining of the neural network model, improving predictive power over time.

- Model Ensembling & Voting Strategy: Beyond a single neural network, we can employ a model ensemble approach, combining predictions from multiple classifiers to improve robustness. A voting mechanism can be introduced — where decisions are based on the majority opinion of several models.

- Alternative Classifiers: The following traditional and advanced ML models can be incorporated, tested, or combined:

	- Discriminant Analysis (LDA/QDA) — useful when class distributions are Gaussian.

	- Logistic Regression — a simple and interpretable baseline for binary classification.

	- Decision Trees — for rule-based scoring and explainability.

	- Support Vector Machines (SVM) — strong performance in high-dimensional spaces.

	- Naive Bayes Classifier — particularly effective with categorical or text features.


- Explainability Techniques:

Integration of SHAP or LIME for feature contribution visualization and trust-building with stakeholders.

- Interactive Dashboard:

A web-based dashboard (e.g., using Streamlit, Dash) could be implemented to allow business analysts to explore:

	- Scores by customer segment,

	- Risk clusters,

	- Neural network confidence levels,

	- Audit trail for each scoring decision.

- Model Monitoring & Alerts:

Set up model drift detection, automated alerts if inputs change over time (e.g., concept drift), and auto-trigger retraining protocols.

---

## 📬 Contact

Feel free to reach out or fork the project for adaptation to your own city or business sector.

**© 2025 Ivan Tatarchuk (Telegram - @Ivan_Tatarchuk; LinkedIn - https://www.linkedin.com/in/ivan-tatarchuk/)**
