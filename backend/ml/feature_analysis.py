"""Feature analysis utilities for understanding feature importance and patterns."""

import numpy as np
from typing import Dict, List, Tuple
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.ensemble import RandomForestClassifier
import shap
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class FeatureImportance:
    """Container for feature importance results."""
    name: str
    importance: float
    category: str
    temporal_stability: float

class FeatureAnalyzer:
    """Analyze feature importance and patterns across modalities."""
    
    def __init__(
        self,
        eeg_feature_names: List[str],
        bio_feature_names: List[str],
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.eeg_feature_names = eeg_feature_names
        self.bio_feature_names = bio_feature_names
        self.device = device
        self.feature_cache = {}
    
    def analyze_mutual_information(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        feature_names: List[str]
    ) -> List[FeatureImportance]:
        """Calculate mutual information between features and labels."""
        mi_scores = mutual_info_classif(features, labels)
        
        # Calculate temporal stability
        temporal_stability = self._calculate_temporal_stability(features)
        
        return [
            FeatureImportance(
                name=name,
                importance=score,
                category=self._get_feature_category(name),
                temporal_stability=stability
            )
            for name, score, stability in zip(
                feature_names, mi_scores, temporal_stability
            )
        ]
    
    def _calculate_temporal_stability(
        self,
        features: np.ndarray,
        window_size: int = 10
    ) -> np.ndarray:
        """Calculate temporal stability of features."""
        n_samples = features.shape[0]
        n_windows = n_samples - window_size + 1
        
        stabilities = []
        for i in range(features.shape[1]):
            window_means = []
            for j in range(n_windows):
                window = features[j:j+window_size, i]
                window_means.append(np.mean(window))
            stabilities.append(1.0 / (np.std(window_means) + 1e-6))
        
        return np.array(stabilities)
    
    def _get_feature_category(self, feature_name: str) -> str:
        """Determine category of feature based on name."""
        if any(cat in feature_name.lower() for cat in ['fft', 'psd', 'wavelet']):
            return 'frequency_domain'
        elif any(cat in feature_name.lower() for cat in ['entropy', 'complexity']):
            return 'complexity'
        elif any(cat in feature_name.lower() for cat in ['hrv', 'bp', 'gsr']):
            return 'biometric'
        elif 'connectivity' in feature_name.lower():
            return 'connectivity'
        else:
            return 'time_domain'
    
    def analyze_shap_values(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        background_size: int = 100
    ) -> Dict[str, np.ndarray]:
        """Calculate SHAP values for model interpretability."""
        model.eval()
        
        # Get background dataset
        background = []
        for i, batch in enumerate(dataloader):
            if i >= background_size:
                break
            background.append(batch)
        background = torch.cat(background, dim=0)
        
        # Create explainer
        explainer = shap.DeepExplainer(model, background.to(self.device))
        
        # Calculate SHAP values
        shap_values = []
        for batch in dataloader:
            batch = batch.to(self.device)
            shap_values.append(
                explainer.shap_values(batch)
            )
        
        return np.concatenate(shap_values, axis=0)
    
    def analyze_feature_correlations(
        self,
        features: np.ndarray,
        feature_names: List[str]
    ) -> np.ndarray:
        """Analyze correlations between features."""
        return np.corrcoef(features.T)
    
    def get_random_forest_importance(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        feature_names: List[str]
    ) -> List[FeatureImportance]:
        """Get feature importance using Random Forest."""
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(features, labels)
        
        temporal_stability = self._calculate_temporal_stability(features)
        
        return [
            FeatureImportance(
                name=name,
                importance=importance,
                category=self._get_feature_category(name),
                temporal_stability=stability
            )
            for name, importance, stability in zip(
                feature_names,
                rf.feature_importances_,
                temporal_stability
            )
        ]
    
    def plot_feature_importance(
        self,
        importance_list: List[FeatureImportance],
        title: str = 'Feature Importance Analysis',
        save_path: Optional[str] = None
    ):
        """Plot feature importance analysis results."""
        # Sort by importance
        sorted_features = sorted(
            importance_list,
            key=lambda x: x.importance,
            reverse=True
        )
        
        # Prepare data for plotting
        names = [f.name for f in sorted_features]
        importances = [f.importance for f in sorted_features]
        categories = [f.category for f in sorted_features]
        
        # Create plot
        plt.figure(figsize=(12, 6))
        bars = plt.barh(names, importances)
        
        # Color bars by category
        category_colors = {
            'frequency_domain': 'skyblue',
            'complexity': 'lightgreen',
            'biometric': 'salmon',
            'connectivity': 'purple',
            'time_domain': 'orange'
        }
        
        for bar, category in zip(bars, categories):
            bar.set_color(category_colors[category])
        
        plt.title(title)
        plt.xlabel('Importance Score')
        
        # Add legend
        legend_elements = [
            plt.Rectangle((0,0),1,1, facecolor=color, label=cat)
            for cat, color in category_colors.items()
        ]
        plt.legend(handles=legend_elements, loc='lower right')
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def analyze_temporal_patterns(
        self,
        features: np.ndarray,
        feature_names: List[str],
        window_sizes: List[int] = [10, 50, 100]
    ) -> Dict[str, Dict[int, float]]:
        """Analyze temporal patterns at different scales."""
        results = {}
        
        for feature_idx, feature_name in enumerate(feature_names):
            feature_data = features[:, feature_idx]
            window_results = {}
            
            for window_size in window_sizes:
                # Calculate rolling statistics
                n_windows = len(feature_data) - window_size + 1
                rolling_mean = np.array([
                    np.mean(feature_data[i:i+window_size])
                    for i in range(n_windows)
                ])
                rolling_std = np.array([
                    np.std(feature_data[i:i+window_size])
                    for i in range(n_windows)
                ])
                
                # Calculate stability metric
                stability = 1.0 / (np.std(rolling_mean) + 1e-6)
                window_results[window_size] = stability
            
            results[feature_name] = window_results
        
        return results
    
    def get_optimal_feature_subset(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        n_features: int
    ) -> Tuple[np.ndarray, List[int]]:
        """Select optimal feature subset using mutual information."""
        selector = SelectKBest(
            score_func=mutual_info_classif,
            k=n_features
        )
        
        selected_features = selector.fit_transform(features, labels)
        selected_indices = selector.get_support(indices=True)
        
        return selected_features, selected_indices
    
    def analyze_all(
        self,
        eeg_features: np.ndarray,
        bio_features: np.ndarray,
        labels: np.ndarray,
        save_dir: Optional[str] = None
    ) -> Dict:
        """Run comprehensive feature analysis."""
        # Combine features
        all_features = np.hstack([eeg_features, bio_features])
        all_feature_names = self.eeg_feature_names + self.bio_feature_names
        
        results = {
            'mutual_information': self.analyze_mutual_information(
                all_features, labels, all_feature_names
            ),
            'random_forest': self.get_random_forest_importance(
                all_features, labels, all_feature_names
            ),
            'correlations': self.analyze_feature_correlations(
                all_features, all_feature_names
            ),
            'temporal_patterns': self.analyze_temporal_patterns(
                all_features, all_feature_names
            )
        }
        
        if save_dir:
            # Plot and save visualizations
            self.plot_feature_importance(
                results['mutual_information'],
                title='Mutual Information Analysis',
                save_path=f'{save_dir}/mutual_info.png'
            )
            
            self.plot_feature_importance(
                results['random_forest'],
                title='Random Forest Feature Importance',
                save_path=f'{save_dir}/random_forest.png'
            )
            
            # Plot correlation matrix
            plt.figure(figsize=(12, 12))
            sns.heatmap(
                results['correlations'],
                xticklabels=all_feature_names,
                yticklabels=all_feature_names
            )
            plt.title('Feature Correlations')
            plt.tight_layout()
            plt.savefig(f'{save_dir}/correlations.png')
            plt.close()
        
        return results