#!/usr/bin/env python3
"""
Quick EDA and baseline analysis for PPMI dataset.
"""

import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run EDA and baseline analysis."""
    
    logger.info("🚀 Starting PPMI EDA & Baseline Analysis")
    logger.info("=" * 50)
    
    try:
        # Import custom PPMI loader
        sys.path.append(str(Path(__file__).parent / 'src'))
        from data.ppmi_custom_loader import load_ppmi_data
        
        # Load PPMI data
        logger.info("Loading PPMI data...")
        mapping_df, summary = load_ppmi_data()
        
        logger.info(f"✅ Dataset loaded: {summary['total_images']} images from {summary['unique_patients']} patients")
        
        # Show data distribution
        logger.info("\n📊 Data Distribution:")
        for folder, count in summary['data_folders'].items():
            logger.info(f"  {folder}: {count} images")
            
        # Show clinical data summary
        if 'sex_distribution' in summary:
            logger.info(f"\n👥 Sex Distribution:")
            for sex, count in summary['sex_distribution'].items():
                logger.info(f"  {sex}: {count} images")
                
        if 'age_stats' in summary:
            age_stats = summary['age_stats']
            logger.info(f"\n📅 Age Statistics:")
            logger.info(f"  Mean: {age_stats['mean']:.1f} years")
            logger.info(f"  Range: {age_stats['min']:.1f} - {age_stats['max']:.1f} years")
            
        # Calculate SBR features for baseline
        logger.info("\n🔬 Calculating SBR features...")
        from features.sbr_calculator import SBRCalculator
        from utils.config import get_config
        
        config = get_config()
        sbr_calculator = SBRCalculator(config)
        
        # Use a sample for quick analysis
        sample_size = min(20, len(mapping_df))
        sample_mapping = mapping_df.sample(n=sample_size, random_state=42)
        
        logger.info(f"Calculating features for {sample_size} sample images...")
        sample_features = sbr_calculator.calculate_sbr_dataset(sample_mapping)
        
        # Prepare for baseline model
        feature_cols = [col for col in sample_features.columns 
                       if col not in ['series_path', 'patient_id', 'label']]
        
        X = sample_features[feature_cols].fillna(0)
        y = sample_features['label']
        
        logger.info(f"✅ Features prepared: {X.shape}")
        logger.info(f"✅ Labels: {y.shape}")
        
        # Quick baseline model
        logger.info("\n🎯 Training baseline model...")
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score, accuracy_score
        from sklearn.preprocessing import StandardScaler
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        baseline_model = RandomForestClassifier(n_estimators=100, random_state=42)
        baseline_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = baseline_model.predict(X_test_scaled)
        y_pred_proba = baseline_model.predict_proba(X_test_scaled)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        logger.info("\n🏆 Baseline Model Results:")
        logger.info("=" * 30)
        logger.info(f"Accuracy: {accuracy:.3f}")
        logger.info(f"ROC AUC: {auc:.3f}")
        
        # Feature importance
        feature_importance = sorted(
            zip(feature_cols, baseline_model.feature_importances_),
            key=lambda x: x[1], reverse=True
        )
        
        logger.info("\n🔝 Top 5 Most Important Features:")
        for i, (feature, importance) in enumerate(feature_importance[:5]):
            logger.info(f"  {i+1}. {feature}: {importance:.3f}")
            
        # Save results
        logger.info("\n💾 Saving results...")
        output_dir = Path("data/metadata")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save baseline results
        baseline_results = {
            'sample_size': sample_size,
            'accuracy': accuracy,
            'roc_auc': auc,
            'top_features': feature_importance[:10]
        }
        
        import json
        with open(output_dir / "baseline_results.json", 'w') as f:
            json.dump(baseline_results, f, indent=2, default=str)
            
        logger.info(f"✅ Baseline results saved to {output_dir / 'baseline_results.json'}")
        
        # Final summary
        logger.info("\n🎉 Analysis Complete!")
        logger.info("=" * 30)
        logger.info(f"📊 Dataset: {summary['total_images']} images from {summary['unique_patients']} patients")
        logger.info(f"🔬 Features: {len(feature_cols)} calculated")
        logger.info(f"🎯 Baseline ROC AUC: {auc:.3f}")
        
        logger.info("\n📚 Next Steps:")
        logger.info("1. Review detailed EDA: jupyter notebook notebooks/ppmi_eda.ipynb")
        logger.info("2. Run full pipeline: python src/main.py")
        logger.info("3. Train CNN models for comparison")
        
        return True
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
