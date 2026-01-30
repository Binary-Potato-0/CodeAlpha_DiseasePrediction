# Disease Prediction from Medical Data

**CodeAlpha Machine Learning Internship - Task 4**

---

##  Project Overview

This project implements a comprehensive **Disease Prediction System** that predicts the possibility of heart disease based on patient medical data. The system uses multiple machine learning classification algorithms including:

- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **Random Forest**
- **XGBoost**

The project demonstrates end-to-end machine learning workflow including data preprocessing, model training, evaluation, and visualization.

---

##  Objectives

- Predict heart disease presence using patient medical data
- Compare multiple classification algorithms
- Evaluate model performance using comprehensive metrics
- Visualize results and feature importance
- Provide a production-ready prediction system

---

##  Dataset

**Source:** UCI Machine Learning Repository - Heart Disease Dataset (Cleveland)

**Features (13 attributes):**
1. `age` - Age in years
2. `sex` - Sex (1 = male; 0 = female)
3. `cp` - Chest pain type (1-4)
4. `trestbps` - Resting blood pressure (mm Hg)
5. `chol` - Serum cholesterol (mg/dl)
6. `fbs` - Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
7. `restecg` - Resting electrocardiographic results (0-2)
8. `thalach` - Maximum heart rate achieved
9. `exang` - Exercise induced angina (1 = yes; 0 = no)
10. `oldpeak` - ST depression induced by exercise
11. `slope` - Slope of peak exercise ST segment (1-3)
12. `ca` - Number of major vessels colored by fluoroscopy (0-3)
13. `thal` - Thalassemia (3 = normal; 6 = fixed defect; 7 = reversible defect)

**Target:**
- `target` - Heart disease diagnosis (0 = no disease, 1 = disease present)

---

##  Technologies Used

- **Python 3.8+**
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation and analysis
- **Scikit-learn** - Machine learning algorithms
- **XGBoost** - Gradient boosting framework
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical visualization

---

##  Project Structure

```
CodeAlpha_DiseasePrediction/
│
├── disease_prediction.py      # Main implementation file
├── download_data.py           # Dataset download script
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
│
├── heart_disease.csv          # Dataset 
│
└── Output Visualizations/
    ├── class_distribution.png
    ├── correlation_matrix.png
    ├── model_comparison.png
    ├── roc_curves.png
    ├── confusion_matrices.png
    └── feature_importance.png
```

---

##  Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Binary-Potato-0/CodeAlpha_DiseasePrediction.git
cd CodeAlpha_DiseasePrediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Dataset

```bash
python download_data.py
```

### 4. Run the Main Program

```bash
python disease_prediction.py
```

---

##  Usage

### Basic Usage

```python
from disease_prediction import DiseasePredictionModel

# Initialize model
model = DiseasePredictionModel('heart_disease.csv')

# Load and analyze data
model.load_data()
model.exploratory_data_analysis()

# Preprocess data
model.preprocess_data(test_size=0.2)

# Train models
model.train_models()

# Evaluate models
model.evaluate_models()

# Visualize results
model.visualize_results()

# Get best model
best_model_name, best_model = model.get_best_model()
```

### Making Predictions

```python
# Sample patient data: [age, sex, cp, trestbps, chol, fbs, restecg, 
#                       thalach, exang, oldpeak, slope, ca, thal]
sample_patient = [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]

# Predict
prediction, probability = model.predict_new_sample(sample_patient)
```

---

##  Model Performance

The system trains and evaluates four different algorithms:

### Evaluation Metrics
- **Accuracy** - Overall correctness
- **Precision** - Positive prediction accuracy
- **Recall** - True positive detection rate
- **F1-Score** - Harmonic mean of precision and recall
- **ROC-AUC** - Area under ROC curve

### Expected Performance
All models typically achieve:
- Accuracy: 75-85%
- ROC-AUC: 0.80-0.90

*(Actual results may vary based on data split)*

---

##  Visualizations

The project generates six comprehensive visualizations:

1. **Class Distribution** - Target variable balance
2. **Correlation Matrix** - Feature correlations heatmap
3. **Model Comparison** - Performance metrics across all models
4. **ROC Curves** - Receiver Operating Characteristic curves
5. **Confusion Matrices** - True/False Positive/Negative breakdown
6. **Feature Importance** - Most influential features (Random Forest & XGBoost)

---

##  Key Features

### Data Preprocessing
- Missing value handling
- Feature scaling using StandardScaler
- Train-test split with stratification

### Model Training
- Multiple algorithm implementation
- Cross-validation (5-fold)
- Hyperparameter optimization ready

### Model Evaluation
- Comprehensive metric calculation
- Confusion matrix analysis
- ROC curve comparison
- Classification reports

### Visualization
- Professional, publication-ready plots
- Comparative analysis charts
- Feature importance analysis

---

##  Results Interpretation

### Understanding Predictions

When the model predicts for a new patient:

```
Prediction: Disease Detected
Confidence: 85.23%
Probability of Disease: 85.23%
Probability of No Disease: 14.77%
```

- **High confidence (>80%)**: Strong prediction
- **Medium confidence (60-80%)**: Moderate certainty
- **Low confidence (<60%)**: Uncertain, may need more features

---

##  Future Enhancements

- [ ] Deep learning models (Neural Networks)
- [ ] Feature engineering and selection
- [ ] Hyperparameter tuning with GridSearchCV
- [ ] Support for multiple disease types
- [ ] Web interface for predictions
- [ ] Model deployment using Flask/FastAPI
- [ ] Real-time prediction API
- [ ] Mobile application integration

---


##  Contact

Mostafa Eldeeb 
Email: mostafa.eldeeb912@gmail.com  
LinkedIn: https://www.linkedin.com/in/mostafa--eldeeb/ 
GitHub: https://github.com/Binary-Potato-0

---

##  License

This project is part of CodeAlpha Machine Learning Internship program.

---

## 🙏 Acknowledgments

- **CodeAlpha** for the internship opportunity
- **UCI Machine Learning Repository** for the dataset
- **Scikit-learn** community for excellent documentation
- All mentors and peers who provided guidance

---

##  References

1. [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease)
2. [Scikit-learn Documentation](https://scikit-learn.org/)
3. [XGBoost Documentation](https://xgboost.readthedocs.io/)
4. [Machine Learning Mastery](https://machinelearningmastery.com/)

---

**⭐ If you find this project helpful, please star the repository!**

---

*Developed with ❤️ for CodeAlpha Machine Learning Internship*
