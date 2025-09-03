#!/usr/bin/env python3
"""
PPMI Machine Learning Benchmarking Script
Benchmarks XGBoost and classical models for Parkinson's Disease classification
using SBR features
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            confusion_matrix, classification_report, roc_auc_score, roc_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

class MLBenchmarker:
    def __init__(self, dataset_path="ppmi_sbr_dataset.csv"):
        self.dataset_path = dataset_path
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
    def load_and_prepare_data(self):
        """Load dataset and prepare features for machine learning"""
        print("Loading and preparing dataset...")
        
        # Load dataset
        self.data = pd.read_csv(self.dataset_path)
        print(f"Dataset loaded: {self.data.shape}")
        
        # Display basic info
        print(f"\nDataset Info:")
        print(f"Total patients: {len(self.data)}")
        print(f"Control patients: {len(self.data[self.data['Group'] == 'Control'])}")
        print(f"PD patients: {len(self.data[self.data['Group'] == 'PD'])}")
        
        # Select SBR features for modeling
        sbr_features = [col for col in self.data.columns if col.endswith('_SBR')]
        print(f"\nSBR features available: {sbr_features}")
        
        # Prepare features (X) and target (y)
        self.X = self.data[sbr_features].copy()
        self.y = self.data['Diagnosis'].copy()
        
        # Remove any rows with NaN values
        valid_indices = ~(self.X.isna().any(axis=1) | self.y.isna())
        self.X = self.X[valid_indices]
        self.y = self.y[valid_indices]
        
        print(f"Features shape after cleaning: {self.X.shape}")
        print(f"Target distribution: {self.y.value_counts().to_dict()}")
        
        # Split data (since we have 40 samples, use 70-30 split)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42, stratify=self.y
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        
        return self.X, self.y
    
    def initialize_models(self):
        """Initialize all models to be benchmarked"""
        print("\nInitializing models...")
        
        # Classical Models
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'SVM (Linear)': SVC(kernel='linear', random_state=42, probability=True),
            'SVM (RBF)': SVC(kernel='rbf', random_state=42, probability=True),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Naive Bayes': GaussianNB(),
            'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
            'Quadratic Discriminant Analysis': QuadraticDiscriminantAnalysis(),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        }
        
        print(f"Initialized {len(self.models)} models")
        return self.models
    
    def train_and_evaluate_models(self):
        """Train and evaluate all models"""
        print("\nTraining and evaluating models...")
        
        # Cross-validation setup
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model in self.models.items():
            print(f"\n--- {name} ---")
            
            try:
                # Use scaled data for models that benefit from it
                if name in ['SVM (Linear)', 'SVM (RBF)', 'K-Nearest Neighbors', 'Linear Discriminant Analysis']:
                    X_train_model = self.X_train_scaled
                    X_test_model = self.X_test_scaled
                else:
                    X_train_model = self.X_train
                    X_test_model = self.X_test
                
                # Train model
                model.fit(X_train_model, self.y_train)
                
                # Make predictions
                y_pred = model.predict(X_test_model)
                y_pred_proba = model.predict_proba(X_test_model)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                accuracy = accuracy_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred, zero_division=0)
                recall = recall_score(self.y_test, y_pred, zero_division=0)
                f1 = f1_score(self.y_test, y_pred, zero_division=0)
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_train_model, self.y_train, cv=cv, scoring='accuracy')
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                # ROC AUC (if probabilities available)
                roc_auc = roc_auc_score(self.y_test, y_pred_proba) if y_pred_proba is not None else None
                
                # Store results
                self.results[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'roc_auc': roc_auc,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba
                }
                
                print(f"  Accuracy: {accuracy:.3f}")
                print(f"  Precision: {precision:.3f}")
                print(f"  Recall: {recall:.3f}")
                print(f"  F1-Score: {f1:.3f}")
                print(f"  CV Accuracy: {cv_mean:.3f} ± {cv_std:.3f}")
                if roc_auc:
                    print(f"  ROC AUC: {roc_auc:.3f}")
                
            except Exception as e:
                print(f"  Error with {name}: {e}")
                self.results[name] = None
        
        return self.results
    
    def create_performance_comparison(self):
        """Create comprehensive performance comparison"""
        print("\nCreating performance comparison...")
        
        # Filter out failed models
        valid_results = {k: v for k, v in self.results.items() if v is not None}
        
        # Create comparison DataFrame
        comparison_data = []
        for name, results in valid_results.items():
            comparison_data.append({
                'Model': name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1'],
                'CV_Accuracy': results['cv_mean'],
                'CV_Std': results['cv_std'],
                'ROC_AUC': results['roc_auc'] if results['roc_auc'] else np.nan
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by F1-Score (best overall metric)
        comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
        
        print("\n=== Model Performance Ranking (by F1-Score) ===")
        print(comparison_df.round(3))
        
        # Save comparison
        comparison_df.to_csv('model_performance_comparison.csv', index=False)
        print(f"\nPerformance comparison saved to: model_performance_comparison.csv")
        
        return comparison_df
    
    def plot_performance_comparison(self, comparison_df):
        """Plot performance comparison charts"""
        print("\nCreating performance visualization...")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Machine Learning Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # 1. Accuracy comparison
        ax1 = axes[0, 0]
        bars1 = ax1.bar(range(len(comparison_df)), comparison_df['Accuracy'], 
                        color='skyblue', alpha=0.7)
        ax1.set_title('Model Accuracy Comparison')
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Accuracy')
        ax1.set_xticks(range(len(comparison_df)))
        ax1.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars1, comparison_df['Accuracy']):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # 2. F1-Score comparison
        ax2 = axes[0, 1]
        bars2 = ax2.bar(range(len(comparison_df)), comparison_df['F1-Score'], 
                        color='lightgreen', alpha=0.7)
        ax2.set_title('Model F1-Score Comparison')
        ax2.set_xlabel('Models')
        ax2.set_ylabel('F1-Score')
        ax2.set_xticks(range(len(comparison_df)))
        ax2.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        for bar, value in zip(bars2, comparison_df['F1-Score']):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # 3. Cross-validation accuracy with error bars
        ax3 = axes[1, 0]
        x_pos = np.arange(len(comparison_df))
        bars3 = ax3.bar(x_pos, comparison_df['CV_Accuracy'], 
                        yerr=comparison_df['CV_Std'], capsize=5, 
                        color='orange', alpha=0.7)
        ax3.set_title('Cross-Validation Accuracy (with std)')
        ax3.set_xlabel('Models')
        ax3.set_ylabel('CV Accuracy')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # 4. ROC AUC comparison (if available)
        ax4 = axes[1, 1]
        roc_data = comparison_df.dropna(subset=['ROC_AUC'])
        if not roc_data.empty:
            bars4 = ax4.bar(range(len(roc_data)), roc_data['ROC_AUC'], 
                            color='purple', alpha=0.7)
            ax4.set_title('ROC AUC Comparison')
            ax4.set_xlabel('Models')
            ax4.set_ylabel('ROC AUC')
            ax4.set_xticks(range(len(roc_data)))
            ax4.set_xticklabels(roc_data['Model'], rotation=45, ha='right')
            ax4.grid(True, alpha=0.3)
            
            for bar, value in zip(bars4, roc_data['ROC_AUC']):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.3f}', ha='center', va='bottom')
        else:
            ax4.text(0.5, 0.5, 'ROC AUC not available\nfor all models', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('ROC AUC Comparison')
        
        plt.tight_layout()
        plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
        print("Performance visualization saved to: model_performance_comparison.png")
        plt.show()
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for top performing models"""
        print("\nCreating confusion matrices...")
        
        # Get top 4 models by F1-Score
        valid_results = {k: v for k, v in self.results.items() if v is not None}
        top_models = sorted(valid_results.items(), 
                           key=lambda x: x[1]['f1'], reverse=True)[:4]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Confusion Matrices - Top 4 Models', fontsize=16, fontweight='bold')
        
        for i, (name, results) in enumerate(top_models):
            row = i // 2
            col = i % 2
            ax = axes[row, col]
            
            # Create confusion matrix
            cm = confusion_matrix(self.y_test, results['y_pred'])
            
            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Control', 'PD'], yticklabels=['Control', 'PD'])
            ax.set_title(f'{name}\nAccuracy: {results["accuracy"]:.3f}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        print("Confusion matrices saved to: confusion_matrices.png")
        plt.show()
    
    def plot_roc_curves(self):
        """Plot ROC curves for models with probability predictions"""
        print("\nCreating ROC curves...")
        
        # Get models with probability predictions
        proba_models = {k: v for k, v in self.results.items() 
                       if v is not None and v['y_pred_proba'] is not None}
        
        if not proba_models:
            print("No models with probability predictions available")
            return
        
        plt.figure(figsize=(10, 8))
        
        for name, results in proba_models.items():
            fpr, tpr, _ = roc_curve(self.y_test, results['y_pred_proba'])
            auc = results['roc_auc']
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
        print("ROC curves saved to: roc_curves.png")
        plt.show()
    
    def feature_importance_analysis(self):
        """Analyze feature importance for tree-based models"""
        print("\nAnalyzing feature importance...")
        
        # Get tree-based models
        tree_models = {
            'Random Forest': self.models.get('Random Forest'),
            'Gradient Boosting': self.models.get('Gradient Boosting'),
            'XGBoost': self.models.get('XGBoost'),
            'Decision Tree': self.models.get('Decision Tree')
        }
        
        # Filter out None models
        tree_models = {k: v for k, v in tree_models.items() if v is not None}
        
        if not tree_models:
            print("No tree-based models available for feature importance analysis")
            return
        
        # Create feature importance plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold')
        
        feature_names = self.X.columns
        
        for i, (name, model) in enumerate(tree_models.items()):
            row = i // 2
            col = i % 2
            ax = axes[row, col]
            
            try:
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    
                    # Sort features by importance
                    indices = np.argsort(importances)[::-1]
                    
                    # Plot feature importance
                    ax.bar(range(len(indices)), importances[indices])
                    ax.set_title(f'{name} - Feature Importance')
                    ax.set_xlabel('Features')
                    ax.set_ylabel('Importance')
                    ax.set_xticks(range(len(indices)))
                    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
                    ax.grid(True, alpha=0.3)
                    
                    # Add value labels
                    for j, importance in enumerate(importances[indices]):
                        ax.text(j, importance + 0.001, f'{importance:.3f}', 
                               ha='center', va='bottom', fontsize=8)
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Error: {str(e)}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{name} - Feature Importance')
        
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        print("Feature importance analysis saved to: feature_importance.png")
        plt.show()
    
    def generate_comprehensive_report(self):
        """Generate comprehensive machine learning report"""
        print("\nGenerating comprehensive report...")
        
        # Create report
        report = []
        report.append("PPMI Machine Learning Benchmarking Report")
        report.append("=" * 50)
        report.append("")
        
        # Dataset summary
        report.append("DATASET SUMMARY")
        report.append("-" * 20)
        report.append(f"Total patients: {len(self.data)}")
        report.append(f"Control patients: {len(self.data[self.data['Group'] == 'Control'])}")
        report.append(f"PD patients: {len(self.data[self.data['Group'] == 'PD'])}")
        report.append(f"Features used: {list(self.X.columns)}")
        report.append(f"Training set size: {self.X_train.shape[0]}")
        report.append(f"Test set size: {self.X_test.shape[0]}")
        report.append("")
        
        # Model performance summary
        report.append("MODEL PERFORMANCE SUMMARY")
        report.append("-" * 30)
        
        # Filter out failed models
        valid_results = {k: v for k, v in self.results.items() if v is not None}
        
        # Sort by F1-Score
        sorted_results = sorted(valid_results.items(), key=lambda x: x[1]['f1'], reverse=True)
        
        for i, (name, results) in enumerate(sorted_results, 1):
            report.append(f"{i}. {name}")
            report.append(f"   Accuracy: {results['accuracy']:.3f}")
            report.append(f"   Precision: {results['precision']:.3f}")
            report.append(f"   Recall: {results['recall']:.3f}")
            report.append(f"   F1-Score: {results['f1']:.3f}")
            report.append(f"   CV Accuracy: {results['cv_mean']:.3f} ± {results['cv_std']:.3f}")
            if results['roc_auc']:
                report.append(f"   ROC AUC: {results['roc_auc']:.3f}")
            report.append("")
        
        # Key findings
        report.append("KEY FINDINGS")
        report.append("-" * 15)
        
        best_model_name = sorted_results[0][0]
        best_model_results = sorted_results[0][1]
        
        report.append(f"Best performing model: {best_model_name}")
        report.append(f"Best F1-Score: {best_model_results['f1']:.3f}")
        report.append(f"Best Accuracy: {best_model_results['accuracy']:.3f}")
        report.append("")
        
        # Feature analysis
        report.append("FEATURE ANALYSIS")
        report.append("-" * 18)
        report.append("SBR features used for classification:")
        for feature in self.X.columns:
            report.append(f"  - {feature}")
        report.append("")
        
        # Save report
        report_text = "\n".join(report)
        with open('ml_benchmarking_report.txt', 'w') as f:
            f.write(report_text)
        
        print("Comprehensive report saved to: ml_benchmarking_report.txt")
        print("\n" + report_text)
        
        return report_text

def main():
    """Main function to run the complete ML benchmarking"""
    print("PPMI Machine Learning Benchmarking")
    print("=" * 50)
    
    # Initialize benchmarker
    benchmarker = MLBenchmarker()
    
    # Load and prepare data
    benchmarker.load_and_prepare_data()
    
    # Initialize models
    benchmarker.initialize_models()
    
    # Train and evaluate models
    benchmarker.train_and_evaluate_models()
    
    # Create performance comparison
    comparison_df = benchmarker.create_performance_comparison()
    
    # Create visualizations
    benchmarker.plot_performance_comparison(comparison_df)
    benchmarker.plot_confusion_matrices()
    benchmarker.plot_roc_curves()
    benchmarker.feature_importance_analysis()
    
    # Generate comprehensive report
    benchmarker.generate_comprehensive_report()
    
    print("\n" + "=" * 50)
    print("MACHINE LEARNING BENCHMARKING COMPLETE!")
    print("=" * 50)

if __name__ == "__main__":
    main()
