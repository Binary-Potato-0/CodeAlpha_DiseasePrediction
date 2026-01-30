"""
Disease Prediction from Medical Data
CodeAlpha Machine Learning Internship - Task 4

This project predicts the possibility of heart disease based on patient medical data
using various classification algorithms including SVM, Logistic Regression, 
Random Forest, and XGBoost.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc, roc_auc_score)
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

class DiseasePredictionModel:
    """
    A comprehensive disease prediction system using multiple ML algorithms
    """
    
    def __init__(self, data_path):
        """
        Initialize the model with data path
        
        Parameters:
        -----------
        data_path : str
            Path to the dataset file
        """
        
        self.data_path = os.path.join(os.path.dirname(__file__), "heart_disease.csv")
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Load and display basic information about the dataset"""
        print("=" * 80)
        print("LOADING DATASET")
        print("=" * 80)
        
        self.data = pd.read_csv(self.data_path)
        
        print(f"\nDataset Shape: {self.data.shape}")
        print(f"Number of Samples: {self.data.shape[0]}")
        print(f"Number of Features: {self.data.shape[1]}")
        
        print("\nFirst few rows:")
        print(self.data.head())
        
        print("\nDataset Info:")
        print(self.data.info())
        
        print("\nStatistical Summary:")
        print(self.data.describe())
        
        return self.data
    
    def exploratory_data_analysis(self):
        """Perform exploratory data analysis"""
        print("\n" + "=" * 80)
        print("EXPLORATORY DATA ANALYSIS")
        print("=" * 80)
        
        # Check for missing values
        print("\nMissing Values:")
        missing = self.data.isnull().sum()
        if missing.sum() == 0:
            print("No missing values found!")
        else:
            print(missing[missing > 0])
        
        # Target variable distribution
        target_col = self.data.columns[-1]
        print(f"\nTarget Variable Distribution ({target_col}):")
        print(self.data[target_col].value_counts())
        
        # Class distribution visualization
        plt.figure(figsize=(10, 6))
        self.data[target_col].value_counts().plot(kind='bar', color=['#3498db', '#e74c3c'])
        plt.title('Disease Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Disease Presence', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("\n✓ Class distribution plot saved!")
        
        # Correlation matrix
        plt.figure(figsize=(14, 10))
        correlation = self.data.corr()
        sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', 
                    center=0, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Correlation matrix saved!")
        
        return correlation
    
    def preprocess_data(self, test_size=0.2, random_state=42):
        """
        Preprocess the data: split and scale
        
        Parameters:
        -----------
        test_size : float
            Proportion of dataset to include in test split
        random_state : int
            Random state for reproducibility
        """
        print("\n" + "=" * 80)
        print("DATA PREPROCESSING")
        print("=" * 80)
        
        # Separate features and target
        X = self.data.iloc[:, :-1]
        y = self.data.iloc[:, -1]
        
        print(f"\nFeatures shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\nTraining set size: {self.X_train.shape[0]}")
        print(f"Testing set size: {self.X_test.shape[0]}")
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("\n✓ Data preprocessing completed!")
        
    def train_models(self):
        """Train multiple classification models"""
        print("\n" + "=" * 80)
        print("MODEL TRAINING")
        print("=" * 80)
        
        # Define models
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'SVM': SVC(kernel='rbf', probability=True, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        # Train each model
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.fit(self.X_train_scaled, self.y_train)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5)
            print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            
            print(f"✓ {name} training completed!")
        
        print("\n✓ All models trained successfully!")
    
    def evaluate_models(self):
        """Evaluate all trained models"""
        print("\n" + "=" * 80)
        print("MODEL EVALUATION")
        print("=" * 80)
        
        for name, model in self.models.items():
            print(f"\n{'=' * 40}")
            print(f"{name}")
            print(f"{'=' * 40}")
            
            # Make predictions
            y_pred = model.predict(self.X_test_scaled)
            y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            
            # Store results
            self.results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            # Print metrics
            print(f"\nAccuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")
            print(f"ROC-AUC:   {roc_auc:.4f}")
            
            # Classification report
            print(f"\nClassification Report:")
            print(classification_report(self.y_test, y_pred))
            
            # Confusion Matrix
            cm = confusion_matrix(self.y_test, y_pred)
            print(f"Confusion Matrix:")
            print(cm)
        
        return self.results
    
    def visualize_results(self):
        """Create comprehensive visualizations of model performance"""
        print("\n" + "=" * 80)
        print("GENERATING VISUALIZATIONS")
        print("=" * 80)
        
        # 1. Model Comparison - Metrics
        metrics_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Accuracy': [r['accuracy'] for r in self.results.values()],
            'Precision': [r['precision'] for r in self.results.values()],
            'Recall': [r['recall'] for r in self.results.values()],
            'F1-Score': [r['f1_score'] for r in self.results.values()],
            'ROC-AUC': [r['roc_auc'] for r in self.results.values()]
        })
        
        # Metrics comparison plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Model Performance Comparison', fontsize=18, fontweight='bold', y=1.02)
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            bars = ax.bar(metrics_df['Model'], metrics_df[metric], color=colors[idx], alpha=0.8)
            ax.set_title(f'{metric}', fontsize=14, fontweight='bold')
            ax.set_ylabel('Score', fontsize=11)
            ax.set_ylim([0, 1.1])
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Hide the last subplot
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("\n✓ Model comparison plot saved!")
        
        # 2. ROC Curves
        plt.figure(figsize=(12, 8))
        
        for name in self.results.keys():
            fpr, tpr, _ = roc_curve(self.y_test, self.results[name]['y_pred_proba'])
            roc_auc = self.results[name]['roc_auc']
            plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=13)
        plt.ylabel('True Positive Rate', fontsize=13)
        plt.title('ROC Curves - All Models', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ ROC curves saved!")
        
        # 3. Confusion Matrices
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('Confusion Matrices', fontsize=18, fontweight='bold', y=0.995)
        
        for idx, (name, result) in enumerate(self.results.items()):
            ax = axes[idx // 2, idx % 2]
            cm = confusion_matrix(self.y_test, result['y_pred'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       cbar_kws={'label': 'Count'}, annot_kws={'size': 14})
            ax.set_title(name, fontsize=14, fontweight='bold')
            ax.set_ylabel('Actual', fontsize=12)
            ax.set_xlabel('Predicted', fontsize=12)
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Confusion matrices saved!")
        
        # 4. Feature Importance (for Random Forest and Gradient Boosting)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold')
        
        # Random Forest
        rf_model = self.models['Random Forest']
        feature_names = self.data.columns[:-1]
        rf_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        axes[0].barh(rf_importance['feature'][:10], rf_importance['importance'][:10], color='#2ecc71')
        axes[0].set_xlabel('Importance', fontsize=12)
        axes[0].set_title('Random Forest - Top 10 Features', fontsize=13, fontweight='bold')
        axes[0].invert_yaxis()
        
        # Gradient Boosting
        gb_model = self.models['Gradient Boosting']
        gb_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': gb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        axes[1].barh(gb_importance['feature'][:10], gb_importance['importance'][:10], color='#e74c3c')
        axes[1].set_xlabel('Importance', fontsize=12)
        axes[1].set_title('Gradient Boosting - Top 10 Features', fontsize=13, fontweight='bold')
        axes[1].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Feature importance plot saved!")
        
        print("\n✓ All visualizations generated successfully!")
    
    def get_best_model(self):
        """Determine the best performing model"""
        print("\n" + "=" * 80)
        print("BEST MODEL SELECTION")
        print("=" * 80)
        
        best_model_name = max(self.results.items(), key=lambda x: x[1]['accuracy'])[0]
        best_accuracy = self.results[best_model_name]['accuracy']
        
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"   Accuracy: {best_accuracy:.4f}")
        print(f"   Precision: {self.results[best_model_name]['precision']:.4f}")
        print(f"   Recall: {self.results[best_model_name]['recall']:.4f}")
        print(f"   F1-Score: {self.results[best_model_name]['f1_score']:.4f}")
        print(f"   ROC-AUC: {self.results[best_model_name]['roc_auc']:.4f}")
        
        return best_model_name, self.models[best_model_name]
    
    def predict_new_sample(self, sample_data):
        """
        Make prediction for new patient data
        
        Parameters:
        -----------
        sample_data : array-like
            New patient features
        """
        best_model_name, best_model = self.get_best_model()
        
        sample_scaled = self.scaler.transform([sample_data])
        prediction = best_model.predict(sample_scaled)[0]
        probability = best_model.predict_proba(sample_scaled)[0]
        
        print("\n" + "=" * 80)
        print("NEW PATIENT PREDICTION")
        print("=" * 80)
        print(f"\nUsing model: {best_model_name}")
        print(f"\nPrediction: {'Disease Detected' if prediction == 1 else 'No Disease'}")
        print(f"Confidence: {probability[prediction] * 100:.2f}%")
        print(f"Probability of Disease: {probability[1] * 100:.2f}%")
        print(f"Probability of No Disease: {probability[0] * 100:.2f}%")
        
        return prediction, probability

def main():
    """Main execution function"""
    print("\n" + "=" * 80)
    print("DISEASE PREDICTION FROM MEDICAL DATA")
    print("CodeAlpha Machine Learning Internship - Task 4")
    print("=" * 80)
    
    # Initialize the model
    model = DiseasePredictionModel('/home/claude/CodeAlpha_DiseasePrediction/heart_disease.csv')
    
    # Load and explore data
    model.load_data()
    model.exploratory_data_analysis()
    
    # Preprocess data
    model.preprocess_data(test_size=0.2, random_state=42)
    
    # Train models
    model.train_models()
    
    # Evaluate models
    model.evaluate_models()
    
    # Visualize results
    model.visualize_results()
    
    # Get best model
    best_model_name, best_model = model.get_best_model()
    
    # Example prediction on new data
    print("\n" + "=" * 80)
    print("EXAMPLE: PREDICTING FOR NEW PATIENT")
    print("=" * 80)
    
    # Sample patient data (modify based on your dataset features)
    sample_patient = [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]
    model.predict_new_sample(sample_patient)
    
    print("\n" + "=" * 80)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nAll results and visualizations have been saved to:")
    print("  D:\CodeAlpha_DiseasePrediction")
    print("\nGenerated files:")
    print("  ✓ class_distribution.png")
    print("  ✓ correlation_matrix.png")
    print("  ✓ model_comparison.png")
    print("  ✓ roc_curves.png")
    print("  ✓ confusion_matrices.png")
    print("  ✓ feature_importance.png")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
