# %%
"""
EDA Interactive Template
========================

This template provides a structured approach to Exploratory Data Analysis
using Cursor's Interactive Window. Execute each cell with Shift+Enter.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("✅ All imports successful!")
print(f"🐍 Python environment ready for EDA")

# %%
# =============================================================================
# 1. DATA LOADING & FIRST LOOK
# =============================================================================

# Load house-prices dataset
DATA_PATH = "data/intro_ml/house-prices/train.csv"
df = pd.read_csv(DATA_PATH)

print(f"📊 Dataset shape: {df.shape}")
print(f"🎯 Target classes: {np.unique(df['target'])}")
print("\n📋 First 5 rows:")
df.head()

# %%
# =============================================================================
# 2. DATASET OVERVIEW
# =============================================================================

print("📈 Dataset Info:")
print(f"• Rows: {df.shape[0]:,}")
print(f"• Columns: {df.shape[1]:,}")
print(f"• Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\n🔍 Column Types:")
print(df.dtypes.value_counts())

print("\n❌ Missing Values:")
missing = df.isnull().sum()
if missing.sum() > 0:
    print(missing[missing > 0].sort_values(ascending=False))
else:
    print("No missing values found!")

# %%
# =============================================================================
# 3. DESCRIPTIVE STATISTICS
# =============================================================================

print("📊 Descriptive Statistics:")
print(df.describe())

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"\n🔄 Duplicate rows: {duplicates}")

# Check target distribution
if 'target' in df.columns:
    print("\n🎯 Target Distribution:")
    print(df['target'].value_counts().sort_index())

# %%
# =============================================================================
# 4. VISUALIZATIONS - DISTRIBUTIONS
# =============================================================================

# Select numeric columns for visualization
numeric_cols = df.select_dtypes(include=[np.number]).columns
numeric_cols = [col for col in numeric_cols if col != 'target']

# Distribution plots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.ravel()

for i, col in enumerate(numeric_cols[:4]):
    df[col].hist(bins=30, ax=axes[i], alpha=0.7, color='skyblue')
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# %%
# =============================================================================
# 5. CORRELATION ANALYSIS
# =============================================================================

# Correlation matrix
corr_matrix = df[numeric_cols].corr()

# Plot correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, 
            annot=True, 
            cmap='coolwarm', 
            center=0,
            square=True,
            fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()

# Find highly correlated features
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.8:
            high_corr_pairs.append((
                corr_matrix.columns[i], 
                corr_matrix.columns[j], 
                corr_matrix.iloc[i, j]
            ))

if high_corr_pairs:
    print("🔥 Highly correlated features (|r| > 0.8):")
    for feat1, feat2, corr in high_corr_pairs:
        print(f"• {feat1} ↔ {feat2}: {corr:.3f}")
else:
    print("✅ No highly correlated features found")

# %%
# =============================================================================
# 6. TARGET ANALYSIS (if applicable)
# =============================================================================

if 'target' in df.columns:
    # Target distribution
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    df['target'].value_counts().plot(kind='bar', color='lightcoral')
    plt.title('Target Distribution')
    plt.xlabel('Target Class')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    
    plt.subplot(1, 2, 2)
    df['target'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['lightcoral', 'lightblue', 'lightgreen'])
    plt.title('Target Distribution (Percentage)')
    plt.ylabel('')
    
    plt.tight_layout()
    plt.show()
    
    # Feature importance via target correlation
    target_corr = df[numeric_cols].corrwith(df['target']).abs().sort_values(ascending=False)
    print("\n🎯 Features most correlated with target:")
    print(target_corr.head(10))

# %%
# =============================================================================
# 7. OUTLIER DETECTION
# =============================================================================

# Box plots for outlier detection
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.ravel()

for i, col in enumerate(numeric_cols[:4]):
    df.boxplot(column=col, ax=axes[i])
    axes[i].set_title(f'Box Plot: {col}')

plt.tight_layout()
plt.show()

# Statistical outlier detection (IQR method)
print("📊 Outlier Detection (IQR Method):")
outlier_counts = {}
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outlier_counts[col] = len(outliers)

for col, count in sorted(outlier_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"• {col}: {count} outliers")

# %%
# =============================================================================
# 8. FEATURE RELATIONSHIPS
# =============================================================================

# Pairplot for key features
key_features = numeric_cols[:4]  # Take first 4 features
if 'target' in df.columns:
    sns.pairplot(df[key_features + ['target']], 
                 hue='target', 
                 diag_kind='hist',
                 corner=True)
    plt.suptitle('Feature Relationships by Target', y=1.02)
else:
    sns.pairplot(df[key_features], 
                 diag_kind='hist',
                 corner=True)
    plt.suptitle('Feature Relationships', y=1.02)

plt.tight_layout()
plt.show()

# %%
# =============================================================================
# 9. TODO - FEATURE ENGINEERING
# =============================================================================

print("🔧 Feature Engineering Section")
print("Add your custom transformations here:")
print("• Log transformations")
print("• Polynomial features")
print("• Binning/discretization")
print("• Feature interactions")
print("• Scaling/normalization")

# Example: Create a simple engineered feature
if len(numeric_cols) >= 2:
    df['feature_ratio'] = df[numeric_cols[0]] / (df[numeric_cols[1]] + 1e-6)
    print(f"\n✅ Created feature_ratio: {numeric_cols[0]}/{numeric_cols[1]}")
    
    # Quick visualization of the new feature
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    df['feature_ratio'].hist(bins=30, alpha=0.7, color='lightgreen')
    plt.title('Distribution of feature_ratio')
    plt.xlabel('feature_ratio')
    plt.ylabel('Frequency')
    
    if 'target' in df.columns:
        plt.subplot(1, 2, 2)
        for target_class in df['target'].unique():
            subset = df[df['target'] == target_class]
            plt.hist(subset['feature_ratio'], bins=20, alpha=0.6, label=f'Class {target_class}')
        plt.title('feature_ratio by Target Class')
        plt.xlabel('feature_ratio')
        plt.ylabel('Frequency')
        plt.legend()
    
    plt.tight_layout()
    plt.show()

# %%
# =============================================================================
# 10. QUICK MODEL BASELINE
# =============================================================================

if 'target' in df.columns:
    print("🤖 Quick Model Baseline")
    
    # Prepare features
    X = df[numeric_cols + ['feature_ratio']].fillna(0)
    y = df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    rf = RandomForestClassifier(random_state=42, n_estimators=100)
    rf.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = rf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"🎯 Random Forest Accuracy: {accuracy:.3f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n📊 Top 5 Most Important Features:")
    for idx, row in feature_importance.head().iterrows():
        print(f"• {row['feature']}: {row['importance']:.3f}")

# %%
# =============================================================================
# 11. SUMMARY & NEXT STEPS
# =============================================================================

print("📝 EDA Summary:")
print("=" * 50)
print(f"✅ Dataset: {df.shape[0]:,} rows, {df.shape[1]:,} columns")
print(f"✅ Missing values: {df.isnull().sum().sum()}")
print(f"✅ Duplicate rows: {df.duplicated().sum()}")
print(f"✅ Numeric features: {len(numeric_cols)}")

if 'target' in df.columns:
    print(f"✅ Target classes: {len(df['target'].unique())}")
    print(f"✅ Baseline accuracy: {accuracy:.3f}")

print("\n🎯 Next Steps:")
print("• Deep dive into interesting patterns")
print("• Advanced feature engineering")
print("• Try different models")
print("• Hyperparameter tuning")
print("• Cross-validation")

print("\n🚀 Ready for ML Marathon!") 