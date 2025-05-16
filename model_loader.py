"""
Deep Prototype Model Loader

This module provides functionality for loading a saved deep prototype model
and making predictions with it. It's designed to be standalone and not dependent
on the original few_shot_prediction_prototype.py file.
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Any, Optional, Union
from scipy.ndimage import gaussian_filter1d


class EmbeddingNet(nn.Module):
    """
    A multi-layer perceptron that embeds input features into a latent space.
    Replicates the architecture from the original model.
    """

    def __init__(self, input_dim: int, embedding_dim: int = 64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class DeepPrototypeModelLoader:
    """
    Standalone class for loading and using a saved deep prototype model.
    """
    
    def __init__(self, model_dir: str, device: str = "cpu"):
        """
        Initialize the model loader by loading all necessary components.
        
        Args:
            model_dir: Directory containing the saved model artifacts
            device: Device to load the model onto ("cpu" or "cuda")
        """
        self.model_dir = model_dir
        self.device = torch.device(device)
        
        # Check if model directory exists
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        # Load feature names
        feature_names_path = os.path.join(model_dir, "feature_names.pkl")
        if not os.path.exists(feature_names_path):
            raise FileNotFoundError(f"Feature names file not found: {feature_names_path}")
        with open(feature_names_path, "rb") as f:
            self.feature_names = pickle.load(f)
        
        # Load scaler
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        
        # Load model configuration
        config_path = os.path.join(model_dir, "model_config.pkl")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Model config file not found: {config_path}")
        with open(config_path, "rb") as f:
            model_config = pickle.load(f)
        
        input_dim = model_config["input_dim"]
        embedding_dim = model_config["embedding_dim"]
        
        # Initialize and load model
        model_path = os.path.join(model_dir, "model.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights file not found: {model_path}")
            
        self.model = EmbeddingNet(input_dim, embedding_dim).to(self.device)
        self.model.load_state_dict(torch.load(
            model_path, 
            map_location=self.device
        ))
        self.model.eval()
        
        # Load prototypes
        prototypes_path = os.path.join(model_dir, "prototypes.pkl")
        if not os.path.exists(prototypes_path):
            raise FileNotFoundError(f"Prototypes file not found: {prototypes_path}")
        with open(prototypes_path, "rb") as f:
            prototype_dict = pickle.load(f)
            # Convert numpy arrays back to torch tensors
            self.prototypes = {k: torch.tensor(v, device=self.device) for k, v in prototype_dict.items()}
        
        # Create a mapping for prototype labels
        self.prototype_labels = ['Negative']
        for key in self.prototypes.keys():
            if key != 0:  # 0 is the negative prototype
                self.prototype_labels.append(str(key))
    
    def preprocess_features(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Preprocess the input features using the saved scaler.
        Ensures all expected features are present, adding zero columns for missing features.
        
        Args:
            features_df: DataFrame containing raw features
            
        Returns:
            Scaled feature array ready for model input
        """
        # Create a copy to avoid modifying the original DataFrame
        features_df = features_df.copy()
        
        # Check and add missing features all at once using a dictionary
        missing_features = {}
        for feature in self.feature_names:
            if feature not in features_df.columns:
                missing_features[feature] = 0
                
        # Add all missing columns at once if any
        if missing_features:
            for feature, value in missing_features.items():
                features_df[feature] = value
        
        # Keep only needed features and in the right order
        X = features_df[self.feature_names]
        
        # Scale the features
        X_scaled = self.scaler.transform(X)
        return X_scaled
    
    def predict(self, features_df: pd.DataFrame, smoothing_sigma: float = 2.0) -> pd.DataFrame:
        """
        Generate predictions for input features.
        
        Args:
            features_df: DataFrame containing features for prediction
            smoothing_sigma: Sigma parameter for Gaussian smoothing of probabilities
            
        Returns:
            DataFrame with prediction results
        """
        # Preprocess features
        X_scaled = self.preprocess_features(features_df)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        
        # Get timestamps if available, otherwise create index-based timestamps
        if "timestamp" in features_df.columns:
            timestamps = features_df["timestamp"].values
        else:
            timestamps = np.arange(len(features_df))
        
        # Generate embeddings
        with torch.no_grad():
            embeddings = self.model(X_tensor)
        
        # Prepare prototype tensors for distance calculation
        all_prototype_tensors = []
        prototype_labels = []
        
        # Add the negative prototype first
        all_prototype_tensors.append(self.prototypes[0].unsqueeze(0))
        prototype_labels.append('Negative')
        
        # Add all positive prototypes
        for category, proto in self.prototypes.items():
            if category != 0:
                all_prototype_tensors.append(proto.unsqueeze(0))
                prototype_labels.append(str(category))
        
        # Stack all prototypes
        all_prototypes_stacked = torch.cat(all_prototype_tensors, dim=0)
        
        # Compute distances to all prototypes
        dists = torch.cdist(embeddings, all_prototypes_stacked, p=2) ** 2
        
        # Apply softmax to get probabilities across all prototypes
        probs = nn.functional.softmax(-dists, dim=1)
        
        # Sum probabilities of all positive prototypes (all except 'Negative' which is at index 0)
        prob_positive = 1 - probs[:, 0].cpu().numpy()
        
        # Apply Gaussian smoothing if requested
        if smoothing_sigma > 0:
            prob_positive = gaussian_filter1d(prob_positive, sigma=smoothing_sigma)
        
        # Get the most likely prototype for each embedding
        most_likely_prototype_idx = torch.argmin(dists, dim=1).cpu().numpy()
        most_likely_prototype = [prototype_labels[idx] for idx in most_likely_prototype_idx]
        
        # Get the most likely POSITIVE prototype (ignoring the negative prototype)
        # First, create a version of distances with the negative prototype set to infinity
        positive_only_dists = dists.clone()
        positive_only_dists[:, 0] = float('inf')  # Set distance to negative prototype to infinity
        
        # Now find the most likely positive prototype
        most_likely_positive_idx = torch.argmin(positive_only_dists, dim=1).cpu().numpy()
        most_likely_positive = [prototype_labels[idx] for idx in most_likely_positive_idx]
        
        # Get the probability of the most likely positive prototype
        most_likely_positive_prob = probs.gather(1, torch.tensor(most_likely_positive_idx, device=self.device).unsqueeze(1)).squeeze().cpu().numpy()
        
        # Create predictions DataFrame
        predictions = pd.DataFrame({
            "timestamp": timestamps,
            "prob_positive": prob_positive,
            "most_likely_prototype": most_likely_prototype,
            "most_likely_positive": most_likely_positive,
            "most_likely_positive_prob": most_likely_positive_prob,
            "is_positive": (most_likely_prototype_idx != 0).astype(int)
        })
        
        return predictions


def example_usage():
    """Example of how to use the model loader."""
    try:
        # Path to the saved model - make sure this directory exists
        model_dir = "models"
        
        # Check if model directory exists first
        if not os.path.exists(model_dir):
            print(f"Model directory not found: {model_dir}")
            print("Please ensure the model has been trained and saved correctly.")
            
            # List available model directories if any
            models_base_dir = "models"
            if os.path.exists(models_base_dir):
                available_models = os.listdir(models_base_dir)
                if available_models:
                    print("\nAvailable model directories:")
                    for model in available_models:
                        print(f"  - {model}")
                else:
                    print(f"\nNo models found in {models_base_dir} directory.")
            return
            
        # Load the model
        model_loader = DeepPrototypeModelLoader(model_dir, device="cpu")
        
        # Create example features all at once to avoid fragmentation
        timestamp_values = [0, 5, 10, 15, 20]
        
        # Build a dictionary of features first
        feature_dict = {"timestamp": timestamp_values}
        
        # Add all feature columns at once with a default value
        for feature in model_loader.feature_names:
            feature_dict[feature] = [0.5] * len(timestamp_values)
        print(model_loader.feature_names)
        
        # Create DataFrame from the complete dictionary (no fragmentation)
        example_features = pd.DataFrame(feature_dict)
        
        # Make predictions
        predictions = model_loader.predict(example_features)
        
        # Print predictions
        print(predictions)
        
        # In a real application, you might want to save the predictions
        # predictions.to_csv("predictions.csv", index=False)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Please check that the model files exist and are correctly formatted.")


if __name__ == "__main__":
    example_usage() 