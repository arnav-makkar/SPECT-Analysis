"""
Configuration management utilities for PPMI project.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration manager for PPMI project."""
    
    def __init__(self, config_path: str = "config/data_config.yaml"):
        """Initialize configuration from YAML file.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file.
        
        Returns:
            Dictionary containing configuration parameters
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation like 'data.raw_path')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_data_path(self, path_type: str) -> Path:
        """Get data path by type.
        
        Args:
            path_type: Type of data path ('raw', 'processed', 'clinical', 'metadata')
            
        Returns:
            Path object for the specified data type
        """
        path_key = f"data.{path_type}_path"
        path_str = self.get(path_key)
        
        if not path_str:
            raise ValueError(f"Data path not found for type: {path_type}")
            
        path = Path(path_str)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_output_path(self, output_type: str) -> Path:
        """Get output path by type.
        
        Args:
            output_type: Type of output path ('models', 'plots', 'reports', 'predictions')
            
        Returns:
            Path object for the specified output type
        """
        path_key = f"output.{output_type}"
        path_str = self.get(path_key)
        
        if not path_str:
            raise ValueError(f"Output path not found for type: {output_type}")
            
        path = Path(path_str)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_preprocessing_params(self) -> Dict[str, Any]:
        """Get preprocessing parameters.
        
        Returns:
            Dictionary of preprocessing parameters
        """
        return self.get('preprocessing', {})
    
    def get_model_params(self, model_type: str) -> Dict[str, Any]:
        """Get model parameters by type.
        
        Args:
            model_type: Type of model ('classical' or 'cnn')
            
        Returns:
            Dictionary of model parameters
        """
        return self.get(f'models.{model_type}', {})
    
    def get_splitting_params(self) -> Dict[str, Any]:
        """Get data splitting parameters.
        
        Returns:
            Dictionary of splitting parameters
        """
        return self.get('splitting', {})
    
    def get_evaluation_params(self) -> Dict[str, Any]:
        """Get evaluation parameters.
        
        Returns:
            Dictionary of evaluation parameters
        """
        return self.get('evaluation', {})
    
    def get_sbr_params(self) -> Dict[str, Any]:
        """Get SBR calculation parameters.
        
        Returns:
            Dictionary of SBR parameters
        """
        return self.get('sbr', {})
    
    def validate_config(self) -> bool:
        """Validate configuration file.
        
        Returns:
            True if configuration is valid
        """
        required_keys = [
            'data.raw_dicom_path',
            'data.clinical_data_path',
            'data.processed_path',
            'data.metadata_path',
            'splitting.train_ratio',
            'splitting.val_ratio',
            'splitting.test_ratio'
        ]
        
        for key in required_keys:
            if self.get(key) is None:
                raise ValueError(f"Required configuration key missing: {key}")
        
        # Validate splitting ratios sum to 1
        train_ratio = self.get('splitting.train_ratio', 0)
        val_ratio = self.get('splitting.val_ratio', 0)
        test_ratio = self.get('splitting.test_ratio', 0)
        
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Splitting ratios must sum to 1.0")
            
        return True


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get global configuration instance.
    
    Returns:
        Global configuration instance
    """
    return config
