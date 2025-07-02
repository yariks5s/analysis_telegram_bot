#!/usr/bin/env python3
"""
Machine Learning Signal Enhancement Module

This module provides functionality to enhance trading signals using machine learning techniques.
It integrates with the backtesting system to provide improved signal quality and accuracy.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Import ML models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Feature engineering and selection tools
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.decomposition import PCA

class MLSignalEnhancer:
    """
    Machine Learning Signal Enhancer class for improving trading signal quality
    """
    
    def __init__(
        self,
        model_type: str = 'random_forest',
        feature_selection: str = 'none',
        n_features: int = 10,
        model_params: Optional[Dict[str, Any]] = None,
        output_dir: str = "./models"
    ):
        """
        Initialize the ML Signal Enhancer
        
        Args:
            model_type: Type of ML model to use (random_forest, gradient_boosting, logistic, svm, mlp)
            feature_selection: Feature selection method (none, kbest, rfe, pca)
            n_features: Number of features to select
            model_params: Additional parameters for the ML model
            output_dir: Directory to save trained models
        """
        self.model_type = model_type
        self.feature_selection = feature_selection
        self.n_features = n_features
        self.model_params = model_params or {}
        self.output_dir = output_dir
        self.model = None
        self.feature_selector = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def _create_model(self) -> Any:
        """Create the ML model based on the specified type"""
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                **self.model_params
            )
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                **self.model_params
            )
        elif self.model_type == 'logistic':
            return LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42,
                **self.model_params
            )
        elif self.model_type == 'svm':
            return SVC(
                C=1.0,
                kernel='rbf',
                probability=True,
                random_state=42,
                **self.model_params
            )
        elif self.model_type == 'mlp':
            return MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                activation='relu',
                random_state=42,
                **self.model_params
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _create_feature_selector(self) -> Any:
        """Create a feature selector based on the specified method"""
        if self.feature_selection == 'none':
            return None
        elif self.feature_selection == 'kbest':
            return SelectKBest(f_classif, k=self.n_features)
        elif self.feature_selection == 'rfe':
            model = self._create_model()
            return RFE(model, n_features_to_select=self.n_features)
        elif self.feature_selection == 'pca':
            return PCA(n_components=self.n_features)
        else:
            raise ValueError(f"Unsupported feature selection method: {self.feature_selection}")
    
    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate features for the ML model from price data.
        
        Args:
            data: DataFrame containing OHLCV price data
            
        Returns:
            DataFrame with engineered features
        """
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Technical indicators as features
        # Moving averages
        for window in [5, 10, 20, 50, 100]:
            df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'close_ma_{window}_ratio'] = df['close'] / df[f'ma_{window}']
            
        # Exponential moving averages
        for window in [5, 10, 20, 50, 100]:
            df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
            df[f'close_ema_{window}_ratio'] = df['close'] / df[f'ema_{window}']
        
        # Bollinger Bands
        for window in [20]:
            mid = df['close'].rolling(window=window).mean()
            std = df['close'].rolling(window=window).std()
            df[f'bb_upper_{window}'] = mid + 2 * std
            df[f'bb_lower_{window}'] = mid - 2 * std
            df[f'bb_width_{window}'] = (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}']) / mid
            df[f'bb_position_{window}'] = (df['close'] - df[f'bb_lower_{window}']) / \
                                        (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}'])
        
        # RSI (Relative Strength Index)
        for window in [14, 21]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            df[f'rsi_{window}'] = 100 - (100 / (1 + rs))
        
        # MACD
        fast_ema = df['close'].ewm(span=12, adjust=False).mean()
        slow_ema = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = fast_ema - slow_ema
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']
        
        # Volatility indicators
        df['atr_14'] = (pd.DataFrame({
            'high-low': df['high'] - df['low'],
            'high-prev_close': abs(df['high'] - df['close'].shift()),
            'low-prev_close': abs(df['low'] - df['close'].shift())
        })).max(axis=1).rolling(14).mean()
        
        # Price rate of change
        for window in [5, 10, 20]:
            df[f'roc_{window}'] = df['close'].pct_change(window) * 100
        
        # Volume features
        df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
        df['volume_ma_10'] = df['volume'].rolling(window=10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma_5']
        
        # Drop NA values from feature calculation
        df = df.dropna()
        
        return df
    
    def prepare_data_for_training(self, 
                               data: pd.DataFrame, 
                               target_column: str = 'target',
                               prediction_horizon: int = 5,
                               threshold: float = 0.0) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for training by generating features and target variable.
        
        Args:
            data: DataFrame containing OHLCV price data
            target_column: Column name for the target variable
            prediction_horizon: Number of periods ahead to predict
            threshold: Required price movement threshold for signal (0.0 = any movement)
            
        Returns:
            Tuple of (X, y) where X is the feature DataFrame and y is the target Series
        """
        # Generate features
        df = self.generate_features(data)
        
        # Create target variable - future price movement direction
        future_returns = df['close'].pct_change(prediction_horizon).shift(-prediction_horizon)
        
        # Classify based on movement threshold
        if threshold > 0:
            df[target_column] = (future_returns > threshold).astype(int)
        else:
            df[target_column] = (future_returns > 0).astype(int)
        
        # Drop NA values created by target creation
        df = df.dropna()
        
        # Extract features and target
        features = df.drop(['open', 'high', 'low', 'close', 'volume', target_column], axis=1)
        target = df[target_column]
        
        return features, target
    
    def train(self,
           data: pd.DataFrame,
           target_column: str = 'target',
           prediction_horizon: int = 5,
           threshold: float = 0.0,
           test_size: float = 0.2,
           use_time_series_split: bool = True,
           n_splits: int = 5) -> Dict[str, Any]:
        """
        Train the ML model to predict trading signals.
        
        Args:
            data: DataFrame containing OHLCV price data
            target_column: Column name for the target variable
            prediction_horizon: Number of periods ahead to predict
            threshold: Required price movement threshold for signal
            test_size: Proportion of data to use for testing
            use_time_series_split: Whether to use time series cross-validation
            n_splits: Number of splits for time series cross-validation
            
        Returns:
            Dictionary with training results and metrics
        """
        # Prepare data
        X, y = self.prepare_data_for_training(data, target_column, prediction_horizon, threshold)
        
        # Create feature selector if needed
        if self.feature_selection != 'none':
            self.feature_selector = self._create_feature_selector()
            X = self.feature_selector.fit_transform(X, y)
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        # Create model if not already created
        if self.model is None:
            self.model = self._create_model()
        
        # Train model
        results = {}
        
        if use_time_series_split:
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=n_splits)
            cv_scores = []
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                self.model.fit(X_train, y_train)
                y_pred = self.model.predict(X_test)
                
                cv_scores.append({
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, zero_division=0),
                    'recall': recall_score(y_test, y_pred, zero_division=0),
                    'f1': f1_score(y_test, y_pred, zero_division=0)
                })
            
            # Calculate average metrics
            for metric in ['accuracy', 'precision', 'recall', 'f1']:
                results[metric] = np.mean([score[metric] for score in cv_scores])
                results[f'{metric}_std'] = np.std([score[metric] for score in cv_scores])
                
            # Train final model on all data
            self.model.fit(X, y)
            
        else:
            # Simple train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, shuffle=False
            )
            
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            results['accuracy'] = accuracy_score(y_test, y_pred)
            results['precision'] = precision_score(y_test, y_pred, zero_division=0)
            results['recall'] = recall_score(y_test, y_pred, zero_division=0)
            results['f1'] = f1_score(y_test, y_pred, zero_division=0)
        
        # Feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            results['feature_importance'] = dict(zip(
                X.columns if isinstance(X, pd.DataFrame) else [f'feature_{i}' for i in range(X.shape[1])],
                self.model.feature_importances_
            ))
        
        self.is_trained = True
        return results
    
    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signal predictions using the trained model.
        
        Args:
            data: DataFrame containing OHLCV price data
            
        Returns:
            Series with prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model is not trained. Call train() first.")
        
        # Generate features
        df = self.generate_features(data)
        
        # Extract features only
        features = df.drop(['open', 'high', 'low', 'close', 'volume'], axis=1, errors='ignore')
        
        # Apply feature selection if used
        if self.feature_selector is not None:
            features = self.feature_selector.transform(features)
        
        # Scale features
        features = self.scaler.transform(features)
        
        # Get predictions
        if hasattr(self.model, 'predict_proba'):
            # For models that provide probability estimates
            probas = self.model.predict_proba(features)
            predictions = pd.Series(probas[:, 1], index=df.index)
        else:
            # For models that only provide binary predictions
            predictions = pd.Series(self.model.predict(features), index=df.index)
        
        return predictions
    
    def enhance_signals(self, 
                      data: pd.DataFrame, 
                      signals: pd.Series,
                      threshold: float = 0.6) -> pd.Series:
        """
        Enhance existing trading signals using ML predictions.
        
        Args:
            data: DataFrame containing OHLCV price data
            signals: Series with original trading signals (1 for buy, -1 for sell, 0 for hold)
            threshold: Probability threshold to consider ML signal
            
        Returns:
            Series with enhanced signals
        """
        if not self.is_trained:
            raise ValueError("Model is not trained. Call train() first.")
        
        # Get ML predictions
        predictions = self.predict(data)
        
        # Enhance signals based on ML predictions
        enhanced_signals = signals.copy()
        
        # Confirm bullish signals if ML prediction is high enough
        enhanced_signals[(signals == 1) & (predictions < threshold)] = 0
        
        # Add new signals based on high ML confidence
        enhanced_signals[(signals == 0) & (predictions >= threshold)] = 1
        
        return enhanced_signals
    
    def save_model(self, filename: Optional[str] = None) -> str:
        """
        Save the trained model to disk.
        
        Args:
            filename: Name to save the model as (without extension)
            
        Returns:
            Path to the saved model
        """
        if not self.is_trained:
            raise ValueError("Model is not trained. Call train() first.")
        
        if filename is None:
            filename = f"{self.model_type}_model"
        
        model_path = os.path.join(self.output_dir, f"{filename}.joblib")
        
        # Save model and associated components
        model_data = {
            'model': self.model,
            'feature_selector': self.feature_selector,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'feature_selection': self.feature_selection,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, model_path)
        return model_path
    
    def load_model(self, model_path: str) -> None:
        """
        Load a previously trained model from disk.
        
        Args:
            model_path: Path to the saved model
        """
        model_data = joblib.load(model_path)
        
        self.model = model_data['model']
        self.feature_selector = model_data['feature_selector']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        self.feature_selection = model_data['feature_selection']
        self.is_trained = model_data['is_trained']
