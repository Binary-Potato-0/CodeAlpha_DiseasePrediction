"""
Download Heart Disease Dataset from UCI ML Repository
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml

def download_heart_disease_data():
    """Download and prepare heart disease dataset"""
    
    print("Downloading Heart Disease dataset from UCI ML Repository...")
    
    # Using the Cleveland Heart Disease dataset
    # Features: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
    # Target: presence of heart disease (0 = no disease, 1-4 = disease)
    
    # Download dataset
    try:
        # Using a pre-processed version of the dataset
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        
        column_names = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
        ]
        
        data = pd.read_csv(url, names=column_names, na_values='?')
        
        # Handle missing values
        data = data.dropna()
        
        # Convert target to binary (0 = no disease, 1 = disease)
        data['target'] = (data['target'] > 0).astype(int)
        
        # Save to CSV
        data.to_csv('/home/claude/CodeAlpha_DiseasePrediction/heart_disease.csv', index=False)
        
        print(f"✓ Dataset downloaded successfully!")
        print(f"  Shape: {data.shape}")
        print(f"  Features: {data.shape[1] - 1}")
        print(f"  Samples: {data.shape[0]}")
        print(f"  Saved to: heart_disease.csv")
        
        # Display feature descriptions
        print("\n" + "=" * 80)
        print("FEATURE DESCRIPTIONS")
        print("=" * 80)
        feature_descriptions = {
            'age': 'Age in years',
            'sex': 'Sex (1 = male; 0 = female)',
            'cp': 'Chest pain type (1-4)',
            'trestbps': 'Resting blood pressure (mm Hg)',
            'chol': 'Serum cholesterol (mg/dl)',
            'fbs': 'Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)',
            'restecg': 'Resting electrocardiographic results (0-2)',
            'thalach': 'Maximum heart rate achieved',
            'exang': 'Exercise induced angina (1 = yes; 0 = no)',
            'oldpeak': 'ST depression induced by exercise',
            'slope': 'Slope of peak exercise ST segment (1-3)',
            'ca': 'Number of major vessels colored by fluoroscopy (0-3)',
            'thal': 'Thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect)',
            'target': 'Heart disease diagnosis (0 = no disease, 1 = disease)'
        }
        
        for feature, description in feature_descriptions.items():
            print(f"{feature:12s} : {description}")
        
        print("\n" + "=" * 80)
        print(f"Target Distribution:")
        print(data['target'].value_counts())
        print("=" * 80)
        
        return data
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nCreating a synthetic dataset for demonstration...")
        
        # Create synthetic dataset as fallback
        np.random.seed(42)
        n_samples = 300
        
        data = pd.DataFrame({
            'age': np.random.randint(30, 80, n_samples),
            'sex': np.random.randint(0, 2, n_samples),
            'cp': np.random.randint(1, 5, n_samples),
            'trestbps': np.random.randint(90, 200, n_samples),
            'chol': np.random.randint(120, 400, n_samples),
            'fbs': np.random.randint(0, 2, n_samples),
            'restecg': np.random.randint(0, 3, n_samples),
            'thalach': np.random.randint(70, 200, n_samples),
            'exang': np.random.randint(0, 2, n_samples),
            'oldpeak': np.random.uniform(0, 6, n_samples).round(1),
            'slope': np.random.randint(1, 4, n_samples),
            'ca': np.random.randint(0, 4, n_samples),
            'thal': np.random.choice([3, 6, 7], n_samples),
            'target': np.random.randint(0, 2, n_samples)
        })
        
        data.to_csv('/home/claude/CodeAlpha_DiseasePrediction/heart_disease.csv', index=False)
        print("✓ Synthetic dataset created successfully!")
        
        return data

if __name__ == "__main__":
    download_heart_disease_data()
