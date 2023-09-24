import numpy as np
import pandas as pd
import seaborn
import xgboost as xgb
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from typing import Set, Any


def calc_feature_importance(X: pd.DataFrame, y: pd.Series, top_n=30) -> (pd.DataFrame, Set):
    """
    Calculate feature importance scores
    Parameters:
        X: The features
        y: The target
        top_n: The number of features to return
    Returns:
        DataFrame: The feature importance scores
        Set: The top features
    """
    # Create an XGBoost model
    model = xgb.XGBRegressor()

    # Fit the model
    model.fit(X, y)

    # Get feature importance scores
    importance_scores = model.feature_importances_

    # Sort feature importance scores
    sorted_indices = importance_scores.argsort()[::-1]
    sorted_scores = importance_scores[sorted_indices]
    feature_names = X.columns[sorted_indices]

    # Get the top 'n' feature importance scores and names
    top_scores = sorted_scores[:top_n]
    top_features = feature_names[:top_n]
    df = pd.DataFrame({'y': top_features, 'corr': top_scores})

    return df, set(top_features)


def plot_correlations(scores: Any, features: Any, title: str):
    """
    Plot feature importance
    Parameters:
        scores: The scores
        features: The features
        title: The title
    """
    # Plot feature importance
    plt.figure(figsize=(10, 10))
    plt.barh(range(len(scores)), scores, align='center')
    plt.yticks(range(len(features)), features)
    plt.xlabel('Score')
    plt.ylabel('Column')
    plt.title(title)
    plt.show()