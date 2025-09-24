# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 21:43:21 2025

@author: joono
"""

import pandas as pd
import numpy as np
import glob
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import plotly.express as px
import plotly.io as pio



# --- 1. Load data ---
def load_punch_data(folder):
    all_data = []
    for file in glob.glob(os.path.join(folder, "*.csv")):
        df = pd.read_csv(file)
        label = os.path.basename(file).split(".")[0].rsplit("_", 1)[0]
        session = os.path.basename(file).split(".")[0]  # use filename as session ID
        all_data.append((df, label, session))
    return all_data

def leave_one_session_out(data_list):
    sessions = sorted(set(s for _, _, s in data_list))
    for session in sessions:
        train_data = [(df, label) for df, label, s in data_list if s != session]
        test_data = [(df, label) for df, label, s in data_list if s == session]

        X_train, y_train = build_dataset(train_data)
        X_test, y_test = build_dataset(test_data)

        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        print(f"\nSession: {session}")
        print(classification_report(y_test, y_pred))

# --- 2. Feature extraction ---
def extract_features(df):
    features = {}
    for col in df.columns:
        if col == 'time':
            continue
        series = df[col].values
        time = df['time'].values
        velocity = np.gradient(series, time)
        peak_idx = np.argmax(np.abs(velocity))
        features[f"{col}_mean"] = np.mean(series)
        features[f"{col}_std"] = np.std(series)
        features[f"{col}_min"] = np.min(series)
        features[f"{col}_max"] = np.max(series)
        features[f"{col}_range"] = np.max(series) - np.min(series)
        features[f"{col}_peak_velocity"] = np.max(np.abs(velocity))
        features[f"{col}_time_to_peak"] = time[peak_idx] - time[0]
        features[f"{col}_velocity_std"] = np.std(velocity)
        features[f"{col}_velocity_range"] = np.max(velocity) - np.min(velocity)
    joint_pairs = [('right_elbow', 'right_shoulder'), ('left_wrist', 'left_elbow')]
    for j1, j2 in joint_pairs:
        if j1 in df.columns and j2 in df.columns:
            corr = np.corrcoef(df[j1], df[j2])[0, 1]
            features[f"{j1}_{j2}_corr"] = corr

    return features


# --- 3. Build dataset ---
def build_dataset(data_list):
    X, y = [], []
    for item in data_list:
        df, label = item[:2]  # ignore session if present
        feats = extract_features(df)
        X.append(feats)
        y.append(label)
    return pd.DataFrame(X), np.array(y)


# --- 3.1 Feature Distributions by Class ---
def plot_feature_distributions(X, y, top_features):
    df = X.copy()
    df['label'] = y
    for feat in top_features:
        plt.figure(figsize=(8, 4))
        sns.violinplot(x='label', y=feat, data=df)
        plt.title(f'Distribution of {feat} by Punch Type')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# -- 3.
def interactive_feature_plot(X, y, feature):
    df = X.copy()
    df['label'] = y
    fig = px.box(df, x='label', y=feature, color='label',
                 title=f'Distribution of {feature} by Punch Type')
    pio.renderers.default = 'browser'
    fig.show()

# --- 3.2 
def plot_embedding(X, y, method='pca'):
    if method == 'pca':
        reducer = PCA(n_components=2)
        title = 'PCA Embedding of Punches'
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=10)
        title = 't-SNE Embedding of Punches'
    elif method == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=42)
        title = 'UMAP Embedding of Punches'
    else:
        raise ValueError("Method must be 'pca', 'tsne', or 'umap'")

    X_embedded = reducer.fit_transform(X)
    df_embed = pd.DataFrame(X_embedded, columns=['Dim1', 'Dim2'])
    df_embed['label'] = y

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_embed, x='Dim1', y='Dim2', hue='label', palette='tab10', s=60)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def cross_validate_model(X, y, n_splits=5):
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=skf, scoring='accuracy')
    print(f"Cross-Validation Accuracy Scores: {scores}")
    print(f"Mean Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

# --- 4. Train/test split ---
def train_random_forest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Classification Report:\n", classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_, normalize='true')
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=clf.classes_, yticklabels=clf.classes_)
    plt.title("Normalized Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.figure(figsize=(10, 6))
    plt.show()

    # Feature importance
    importances = pd.Series(clf.feature_importances_, index=X.columns)
    importances.sort_values(ascending=False).head(15).plot(kind='barh')

    plt.title("Top 15 Feature Importances")
    plt.figure(figsize=(10, 6))

    plt.show()
    
    top_feats = importances.sort_values(ascending=False).head(5).index.tolist()
    plot_feature_distributions(X, y, top_feats)
    
    plot_embedding(X, y, method='pca')
    plot_embedding(X, y, method='tsne')
    plot_embedding(X, y, method='umap')
    
    interactive_feature_plot(X, y, 'R_wrist_min')


    return clf

# --- Run the pipeline ---
if __name__ == "__main__":
    folder = r"C:\Users\joono\OneDrive\Desktop\1. Post Graduate Diploma in Computer and Information Science\AI in Biomechanics\Assignment\Assignment 2\Data Collection Script\Data\\"
    data_list = load_punch_data(folder)
    X, y = build_dataset(data_list)
    classes, counts = np.unique(y, return_counts=True)
    print(pd.DataFrame({'class': classes, 'count': counts}))
    model = train_random_forest(X, y)
    cross_validate_model(X, y)
    leave_one_session_out(data_list)



