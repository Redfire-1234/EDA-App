import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import io
import base64
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import joblib

# Global state
state = {
    "df": None,
    "model": None,
    "X_train": None,
    "X_test": None,
    "y_train": None,
    "y_test": None
}

# -----------------------------
# 1️⃣ Data Upload
# -----------------------------
def upload_data(file):
    if file is None:
        return "No file uploaded", None
    
    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file.name)
        elif file.name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file.name)
        elif file.name.endswith(".json"):
            df = pd.read_json(file.name)
        else:
            return "Unsupported file type", None
        
        state["df"] = df
        info_text = f"""✅ {file.name} uploaded successfully!

**Dataset Info:**
- Shape: {df.shape[0]} rows × {df.shape[1]} columns
- Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
- Columns: {', '.join(df.columns.tolist())}
- Numeric Columns: {len(df.select_dtypes(include=np.number).columns)}
- Categorical Columns: {len(df.select_dtypes(include='object').columns)}
- Missing Values: {df.isnull().sum().sum()}
"""
        return info_text, df.head(20)
    except Exception as e:
        return f"❌ Error loading file: {e}", None

# -----------------------------
# 2️⃣ Data Cleaning - Enhanced
# -----------------------------

# 1. HANDLING MISSING DATA - Enhanced
def handle_missing_data(action):
    if state["df"] is None:
        return "⚠️ Please upload data first", None
    
    df = state["df"].copy()
    before_nulls = df.isnull().sum().sum()
    
    if before_nulls == 0:
        return "✅ No missing values found", df
    
    if action == "Drop Rows":
        df = df.dropna()
    elif action == "Fill with Mean":
        df = df.fillna(df.mean(numeric_only=True))
    elif action == "Fill with Median":
        df = df.fillna(df.median(numeric_only=True))
    elif action == "Fill with Mode":
        for col in df.columns:
            if not df[col].mode().empty:
                df[col] = df[col].fillna(df[col].mode()[0])
    elif action == "Forward Fill":
        df = df.fillna(method='ffill')
    elif action == "Backward Fill":
        df = df.fillna(method='bfill')
    
    state["df"] = df
    after_nulls = df.isnull().sum().sum()
    return f"✅ Missing data handled! Before: {before_nulls} → After: {after_nulls} nulls", df.head(20)

# 2. HANDLING DUPLICATES
def remove_duplicates():
    if state["df"] is None:
        return "⚠️ Please upload data first", None
    
    df = state["df"].copy()
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    state["df"] = df
    return f"✅ Removed {before - after} duplicate rows (Before: {before} → After: {after})", df.head(20)

# 3. HANDLING OUTLIERS - Enhanced with multiple methods
def remove_outliers_iqr(column):
    if state["df"] is None or not column:
        return "⚠️ Please upload data and select a column", None
    
    df = state["df"].copy()
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    before = len(df)
    df = df[~((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR)))]
    after = len(df)
    state["df"] = df
    return f"✅ IQR Method: Removed {before - after} outlier rows from '{column}'", df.head(20)

def remove_outliers_zscore(column, threshold):
    if state["df"] is None or not column:
        return "⚠️ Please upload data and select a column", None
    
    df = state["df"].copy()
    z_scores = np.abs(stats.zscore(df[column].dropna()))
    before = len(df)
    df = df[(z_scores < threshold) | df[column].isna()]
    after = len(df)
    state["df"] = df
    return f"✅ Z-Score Method: Removed {before - after} outliers (threshold={threshold})", df.head(20)

def cap_floor_outliers(column):
    if state["df"] is None or not column:
        return "⚠️ Please upload data and select a column", None
    
    df = state["df"].copy()
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    before_min = df[column].min()
    before_max = df[column].max()
    
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    state["df"] = df
    
    return f"✅ Capped/Floored outliers in '{column}'\nBefore: [{before_min:.2f}, {before_max:.2f}]\nAfter: [{df[column].min():.2f}, {df[column].max():.2f}]", df.head(20)

# 4. CORRECTING INCONSISTENT DATA
def standardize_text_data():
    if state["df"] is None:
        return "⚠️ Please upload data first", None
    
    df = state["df"].copy()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    
    for col in cat_cols:
        df[col] = df[col].str.strip().str.lower().str.replace(r'\s+', ' ', regex=True)
    
    state["df"] = df
    return f"✅ Standardized {len(cat_cols)} text columns (trim, lowercase, normalize spaces)", df.head(20)

def fix_inconsistent_categories(column, mapping_text):
    """
    mapping_text format: old_value1:new_value1, old_value2:new_value2
    Example: male:Male, m:Male, female:Female, f:Female
    """
    if state["df"] is None or not column or not mapping_text:
        return "⚠️ Please provide column and mapping", None
    
    df = state["df"].copy()
    
    try:
        # Parse mapping
        mapping = {}
        for pair in mapping_text.split(','):
            old, new = pair.strip().split(':')
            mapping[old.strip()] = new.strip()
        
        df[column] = df[column].replace(mapping)
        state["df"] = df
        return f"✅ Applied mapping to '{column}': {mapping}", df.head(20)
    except Exception as e:
        return f"❌ Error: {e}\nFormat: old1:new1, old2:new2", None

# 5. HANDLING NOISY DATA
def smooth_data_binning(column, n_bins):
    if state["df"] is None or not column:
        return "⚠️ Please select a column", None
    
    df = state["df"].copy()
    df[f"{column}_binned"] = pd.cut(df[column], bins=n_bins, labels=False)
    state["df"] = df
    return f"✅ Created binned version of '{column}' with {n_bins} bins", df.head(20)

def smooth_data_moving_average(column, window):
    if state["df"] is None or not column:
        return "⚠️ Please select a column", None
    
    df = state["df"].copy()
    df[f"{column}_ma{window}"] = df[column].rolling(window=window, min_periods=1).mean()
    state["df"] = df
    return f"✅ Created moving average of '{column}' with window={window}", df.head(20)

# 6. SCALING - Enhanced
def scale_data(method):
    if state["df"] is None:
        return "⚠️ Please upload data first", None
    
    df = state["df"].copy()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    if not numeric_cols:
        return "❌ No numeric columns to scale", df
    
    if method == "Standardization (Z-score)":
        df[numeric_cols] = StandardScaler().fit_transform(df[numeric_cols])
    elif method == "Min-Max Normalization":
        df[numeric_cols] = MinMaxScaler().fit_transform(df[numeric_cols])
    elif method == "Robust Scaling":
        from sklearn.preprocessing import RobustScaler
        df[numeric_cols] = RobustScaler().fit_transform(df[numeric_cols])
    
    state["df"] = df
    return f"✅ {method} applied to {len(numeric_cols)} numeric columns", df.head(20)

# 7. REMOVING IRRELEVANT COLUMNS
def drop_columns(columns_text):
    """columns_text: comma-separated column names"""
    if state["df"] is None:
        return "⚠️ Please upload data first", None
    
    if not columns_text:
        return "⚠️ Please specify columns to drop", None
    
    df = state["df"].copy()
    cols_to_drop = [c.strip() for c in columns_text.split(',')]
    
    # Filter only existing columns
    existing_cols = [c for c in cols_to_drop if c in df.columns]
    
    if not existing_cols:
        return f"❌ None of the specified columns exist: {cols_to_drop}", df.head(20)
    
    df = df.drop(columns=existing_cols)
    state["df"] = df
    return f"✅ Dropped {len(existing_cols)} columns: {existing_cols}", df.head(20)

# 8. VALIDATION AND VERIFICATION
def validate_data():
    if state["df"] is None:
        return "⚠️ Please upload data first"
    
    df = state["df"]
    
    report = f"""📊 **Data Validation Report**

**Basic Info:**
- Rows: {df.shape[0]}
- Columns: {df.shape[1]}
- Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB

**Data Types:**
{df.dtypes.value_counts().to_string()}

**Missing Values:**
{df.isnull().sum()[df.isnull().sum() > 0].to_string() if df.isnull().sum().sum() > 0 else 'None'}

**Duplicates:** {df.duplicated().sum()}

**Numeric Columns Summary:**
{df.describe().to_string()}

**Potential Issues:**
"""
    
    issues = []
    
    # Check for high missing values
    high_missing = df.isnull().sum() / len(df) > 0.5
    if high_missing.any():
        issues.append(f"- Columns with >50% missing: {high_missing[high_missing].index.tolist()}")
    
    # Check for constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_cols:
        issues.append(f"- Constant columns (consider dropping): {constant_cols}")
    
    # Check for high cardinality categorical columns
    cat_cols = df.select_dtypes(include='object').columns
    high_card = [col for col in cat_cols if df[col].nunique() > 50]
    if high_card:
        issues.append(f"- High cardinality categorical columns: {high_card}")
    
    if issues:
        report += "\n".join(issues)
    else:
        report += "✅ No major issues detected"
    
    return report

# -----------------------------
# 3️⃣ EDA - COMPREHENSIVE
# -----------------------------
def get_statistics():
    if state["df"] is None:
        return "⚠️ Please upload data first"
    
    df = state["df"]
    numeric_df = df.select_dtypes(include=np.number)
    
    # Basic descriptive stats
    desc_stats = df.describe(include='all').to_html()
    
    # Mode for all columns
    mode_stats = df.mode().iloc[0].to_frame(name='Mode').T.to_html()
    
    # Skewness and Kurtosis for numeric columns
    if len(numeric_df.columns) > 0:
        skew_kurt = pd.DataFrame({
            'Skewness': numeric_df.skew(),
            'Kurtosis': numeric_df.kurtosis()
        }).to_html()
    else:
        skew_kurt = "No numeric columns"
    
    # Correlation matrix
    if len(numeric_df.columns) > 1:
        corr = numeric_df.corr().to_html()
        
        # Pearson, Spearman correlations
        pearson_corr = numeric_df.corr(method='pearson').to_html()
        spearman_corr = numeric_df.corr(method='spearman').to_html()
    else:
        corr = pearson_corr = spearman_corr = "Not enough numeric columns"
    
    return f"""### 📊 Descriptive Statistics
{desc_stats}

### 📈 Mode (Most Frequent Values)
{mode_stats}

### 📐 Skewness & Kurtosis
{skew_kurt}

### 🔗 Pearson Correlation Matrix
{pearson_corr}

### 🔗 Spearman Correlation Matrix
{spearman_corr}
"""

def plot_histogram(column):
    if state["df"] is None or not column:
        return None
    
    df = state["df"]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df[column], kde=True, ax=ax)
    ax.set_title(f"Histogram of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    return fig

def plot_density(column):
    if state["df"] is None or not column:
        return None
    
    df = state["df"]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(df[column], fill=True, ax=ax)
    ax.set_title(f"Density Plot of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Density")
    plt.tight_layout()
    return fig

def plot_bar_chart(column):
    if state["df"] is None or not column:
        return None
    
    df = state["df"]
    fig, ax = plt.subplots(figsize=(10, 6))
    value_counts = df[column].value_counts()
    value_counts.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
    ax.set_title(f"Bar Chart of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def plot_pie_chart(column):
    if state["df"] is None or not column:
        return None
    
    df = state["df"]
    fig, ax = plt.subplots(figsize=(10, 8))
    value_counts = df[column].value_counts()
    ax.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=90)
    ax.set_title(f"Pie Chart of {column}")
    plt.tight_layout()
    return fig

def plot_boxplot(column):
    if state["df"] is None or not column:
        return None
    
    df = state["df"]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x=df[column], ax=ax)
    ax.set_title(f"Boxplot of {column}")
    plt.tight_layout()
    return fig

def plot_correlation_heatmap():
    if state["df"] is None:
        return None
    
    df = state["df"]
    numeric_df = df.select_dtypes(include=np.number)
    
    if numeric_df.shape[1] < 2:
        return None
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax, fmt=".2f", 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Heatmap (Pearson)")
    plt.tight_layout()
    return fig

def plot_pairplot():
    if state["df"] is None:
        return None
    
    df = state["df"]
    numeric_df = df.select_dtypes(include=np.number)
    
    if numeric_df.shape[1] < 2:
        return None
    
    # Limit to first 5 numeric columns for performance
    cols_to_plot = numeric_df.columns[:5].tolist()
    
    fig = sns.pairplot(df[cols_to_plot], diag_kind='kde', plot_kws={'alpha': 0.6})
    fig.fig.suptitle("Pairplot (First 5 Numeric Columns)", y=1.01)
    return fig.fig

def plot_missing_heatmap():
    if state["df"] is None:
        return None
    
    df = state["df"]
    
    if df.isnull().sum().sum() == 0:
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.text(0.5, 0.5, 'No Missing Values', ha='center', va='center', fontsize=16)
        ax.axis('off')
        return fig
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis', ax=ax)
    ax.set_title("Missing Values Heatmap (Yellow = Missing)")
    plt.tight_layout()
    return fig

def plot_missing_bar():
    if state["df"] is None:
        return None
    
    df = state["df"]
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    
    if len(missing) == 0:
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.text(0.5, 0.5, '✅ No Missing Values', ha='center', va='center', fontsize=16, color='green')
        ax.axis('off')
        return fig
    
    fig, ax = plt.subplots(figsize=(10, 6))
    missing.plot(kind='bar', ax=ax, color='coral', edgecolor='black')
    ax.set_title("Missing Values by Column")
    ax.set_xlabel("Column")
    ax.set_ylabel("Number of Missing Values")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def plot_scatter(x_col, y_col):
    if state["df"] is None or not x_col or not y_col:
        return None
    
    df = state["df"]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=df[x_col], y=df[y_col], ax=ax)
    ax.set_title(f"Scatter Plot: {x_col} vs {y_col}")
    plt.tight_layout()
    return fig

def plot_line(x_col, y_col):
    if state["df"] is None or not x_col or not y_col:
        return None
    
    df = state["df"]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df[x_col], df[y_col], marker='o', linestyle='-')
    ax.set_title(f"Line Plot: {x_col} vs {y_col}")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    plt.tight_layout()
    return fig

def plot_boxplot_bivariate(cat_col, num_col):
    if state["df"] is None or not cat_col or not num_col:
        return None
    
    df = state["df"]
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x=df[cat_col], y=df[num_col], ax=ax)
    ax.set_title(f"Boxplot: {num_col} by {cat_col}")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def plot_violin(cat_col, num_col):
    if state["df"] is None or not cat_col or not num_col:
        return None
    
    df = state["df"]
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.violinplot(x=df[cat_col], y=df[num_col], ax=ax)
    ax.set_title(f"Violin Plot: {num_col} by {cat_col}")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def plot_stacked_bar(cat_col1, cat_col2):
    if state["df"] is None or not cat_col1 or not cat_col2:
        return None
    
    df = state["df"]
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create crosstab
    ct = pd.crosstab(df[cat_col1], df[cat_col2])
    ct.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
    
    ax.set_title(f"Stacked Bar Chart: {cat_col1} vs {cat_col2}")
    ax.set_xlabel(cat_col1)
    ax.set_ylabel("Count")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title=cat_col2, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    return fig

def plot_categorical_heatmap(cat_col1, cat_col2):
    if state["df"] is None or not cat_col1 or not cat_col2:
        return None
    
    df = state["df"]
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create crosstab
    ct = pd.crosstab(df[cat_col1], df[cat_col2])
    sns.heatmap(ct, annot=True, fmt='d', cmap='YlOrRd', ax=ax)
    
    ax.set_title(f"Heatmap: {cat_col1} vs {cat_col2}")
    plt.tight_layout()
    return fig

# -----------------------------
# 4️⃣ Feature Engineering - COMPREHENSIVE
# -----------------------------

# 1. FEATURE CREATION
def combine_features(col1, col2, operation, new_name):
    if state["df"] is None or not col1 or not col2 or not new_name:
        return "⚠️ Please provide all inputs", None
    
    df = state["df"].copy()
    
    try:
        if operation == "Add (+)":
            df[new_name] = df[col1] + df[col2]
        elif operation == "Subtract (-)":
            df[new_name] = df[col1] - df[col2]
        elif operation == "Multiply (*)":
            df[new_name] = df[col1] * df[col2]
        elif operation == "Divide (/)":
            df[new_name] = df[col1] / df[col2]
        
        state["df"] = df
        return f"✅ Created feature '{new_name}' = {col1} {operation} {col2}", df.head(20)
    except Exception as e:
        return f"❌ Error: {e}", None

def extract_datetime_features(column):
    if state["df"] is None or not column:
        return "⚠️ Please select a column", None
    
    df = state["df"].copy()
    
    try:
        # Convert to datetime if not already
        if df[column].dtype != 'datetime64[ns]':
            df[column] = pd.to_datetime(df[column])
        
        # Extract features
        df[f"{column}_year"] = df[column].dt.year
        df[f"{column}_month"] = df[column].dt.month
        df[f"{column}_day"] = df[column].dt.day
        df[f"{column}_dayofweek"] = df[column].dt.dayofweek
        df[f"{column}_hour"] = df[column].dt.hour
        df[f"{column}_minute"] = df[column].dt.minute
        df[f"{column}_quarter"] = df[column].dt.quarter
        df[f"{column}_week"] = df[column].dt.isocalendar().week
        
        state["df"] = df
        return f"✅ Extracted 8 datetime features from '{column}'", df.head(20)
    except Exception as e:
        return f"❌ Error: {e}\nMake sure the column contains date/time data", None

def create_ratio_feature(numerator, denominator, new_name):
    if state["df"] is None or not numerator or not denominator or not new_name:
        return "⚠️ Please provide all inputs", None
    
    df = state["df"].copy()
    
    try:
        df[new_name] = df[numerator] / df[denominator]
        df[new_name] = df[new_name].replace([np.inf, -np.inf], np.nan)
        state["df"] = df
        return f"✅ Created ratio feature '{new_name}' = {numerator} / {denominator}", df.head(20)
    except Exception as e:
        return f"❌ Error: {e}", None

# 2. POLYNOMIAL FEATURES
def create_polynomial_features(columns_text, degree):
    if state["df"] is None or not columns_text:
        return "⚠️ Please provide columns", None
    
    df = state["df"].copy()
    columns = [c.strip() for c in columns_text.split(',')]
    
    try:
        from sklearn.preprocessing import PolynomialFeatures
        
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        poly_features = poly.fit_transform(df[columns])
        
        feature_names = poly.get_feature_names_out(columns)
        poly_df = pd.DataFrame(poly_features, columns=feature_names, index=df.index)
        
        # Add only new features (not original ones)
        new_features = [col for col in poly_df.columns if col not in columns]
        for col in new_features:
            df[col] = poly_df[col]
        
        state["df"] = df
        return f"✅ Created {len(new_features)} polynomial features (degree={degree})", df.head(20)
    except Exception as e:
        return f"❌ Error: {e}", None

# 3. ENCODING - Enhanced with Target Encoding
def encode_categorical(column, method):
    if state["df"] is None or not column:
        return "⚠️ Please upload data and select a column", None
    
    df = state["df"].copy()
    
    try:
        if method == "One-Hot":
            df = pd.get_dummies(df, columns=[column], drop_first=True)
        elif method == "Label":
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column].astype(str))
        elif method == "Frequency":
            freq = df[column].value_counts()
            df[column] = df[column].map(freq)
        
        state["df"] = df
        return f"✅ {method} encoding applied to '{column}'", df.head(20)
    except Exception as e:
        return f"❌ Error: {e}", None

def target_encode(cat_column, target_column):
    if state["df"] is None or not cat_column or not target_column:
        return "⚠️ Please provide both columns", None
    
    df = state["df"].copy()
    
    try:
        # Calculate mean target value for each category
        target_means = df.groupby(cat_column)[target_column].mean()
        df[f"{cat_column}_target_encoded"] = df[cat_column].map(target_means)
        
        state["df"] = df
        return f"✅ Target encoding applied: '{cat_column}' based on '{target_column}'", df.head(20)
    except Exception as e:
        return f"❌ Error: {e}", None

# 4. CYCLICAL ENCODING for Date/Time
def cyclical_encode_datetime(column, component):
    if state["df"] is None or not column:
        return "⚠️ Please select a column", None
    
    df = state["df"].copy()
    
    try:
        if df[column].dtype != 'datetime64[ns]':
            df[column] = pd.to_datetime(df[column])
        
        if component == "Month":
            df[f"{column}_month_sin"] = np.sin(2 * np.pi * df[column].dt.month / 12)
            df[f"{column}_month_cos"] = np.cos(2 * np.pi * df[column].dt.month / 12)
        elif component == "Day":
            df[f"{column}_day_sin"] = np.sin(2 * np.pi * df[column].dt.day / 31)
            df[f"{column}_day_cos"] = np.cos(2 * np.pi * df[column].dt.day / 31)
        elif component == "Hour":
            df[f"{column}_hour_sin"] = np.sin(2 * np.pi * df[column].dt.hour / 24)
            df[f"{column}_hour_cos"] = np.cos(2 * np.pi * df[column].dt.hour / 24)
        elif component == "DayOfWeek":
            df[f"{column}_dow_sin"] = np.sin(2 * np.pi * df[column].dt.dayofweek / 7)
            df[f"{column}_dow_cos"] = np.cos(2 * np.pi * df[column].dt.dayofweek / 7)
        
        state["df"] = df
        return f"✅ Cyclical encoding applied to '{column}' - {component}", df.head(20)
    except Exception as e:
        return f"❌ Error: {e}", None

# 5. TEXT FEATURES
def extract_text_features(column):
    if state["df"] is None or not column:
        return "⚠️ Please select a column", None
    
    df = state["df"].copy()
    
    try:
        df[f"{column}_length"] = df[column].astype(str).apply(len)
        df[f"{column}_word_count"] = df[column].astype(str).apply(lambda x: len(x.split()))
        df[f"{column}_char_count"] = df[column].astype(str).apply(lambda x: len(x.replace(" ", "")))
        df[f"{column}_avg_word_length"] = df[column].astype(str).apply(
            lambda x: np.mean([len(word) for word in x.split()]) if len(x.split()) > 0 else 0
        )
        
        state["df"] = df
        return f"✅ Extracted 4 text features from '{column}'", df.head(20)
    except Exception as e:
        return f"❌ Error: {e}", None

def apply_tfidf(column, max_features):
    if state["df"] is None or not column:
        return "⚠️ Please select a column", None
    
    df = state["df"].copy()
    
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        vectorizer = TfidfVectorizer(max_features=max_features)
        tfidf_matrix = vectorizer.fit_transform(df[column].astype(str))
        
        feature_names = vectorizer.get_feature_names_out()
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"tfidf_{name}" for name in feature_names], index=df.index)
        
        df = pd.concat([df, tfidf_df], axis=1)
        state["df"] = df
        
        return f"✅ Created {max_features} TF-IDF features from '{column}'", df.head(20)
    except Exception as e:
        return f"❌ Error: {e}", None

# 6. FEATURE SELECTION
def select_by_correlation(threshold, target_col):
    if state["df"] is None:
        return "⚠️ Please upload data first", None
    
    df = state["df"].copy()
    
    try:
        if not target_col:
            # Remove highly correlated features
            numeric_df = df.select_dtypes(include=np.number)
            corr_matrix = numeric_df.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
            df = df.drop(columns=to_drop)
            message = f"✅ Removed {len(to_drop)} highly correlated features (>{threshold}): {to_drop}"
        else:
            # Select features correlated with target
            numeric_df = df.select_dtypes(include=np.number)
            if target_col in numeric_df.columns:
                correlations = numeric_df.corr()[target_col].abs().sort_values(ascending=False)
                selected = correlations[correlations > threshold].index.tolist()
                selected = [col for col in selected if col != target_col]
                df = df[[target_col] + selected]
                message = f"✅ Selected {len(selected)} features correlated with '{target_col}' (>{threshold})"
            else:
                return f"❌ Target column '{target_col}' not found or not numeric", None
        
        state["df"] = df
        return message, df.head(20)
    except Exception as e:
        return f"❌ Error: {e}", None

def apply_pca_reduction(n_components):
    if state["df"] is None:
        return "⚠️ Please upload data first", None
    
    df = state["df"].copy()
    numeric_df = df.select_dtypes(include=np.number)
    
    if numeric_df.shape[1] < 2:
        return "❌ Need at least 2 numeric columns for PCA", None
    
    try:
        pca = PCA(n_components=n_components)
        pca_features = pca.fit_transform(numeric_df)
        
        pca_df = pd.DataFrame(
            pca_features, 
            columns=[f"PC{i+1}" for i in range(n_components)],
            index=df.index
        )
        
        # Keep non-numeric columns and add PCA components
        non_numeric = df.select_dtypes(exclude=np.number)
        df = pd.concat([non_numeric, pca_df], axis=1)
        
        state["df"] = df
        variance_explained = pca.explained_variance_ratio_
        
        return f"✅ Applied PCA: {numeric_df.shape[1]} features → {n_components} components\nVariance explained: {variance_explained.sum():.2%}", df.head(20)
    except Exception as e:
        return f"❌ Error: {e}", None

def feature_importance_selection(target_col, n_features):
    if state["df"] is None or not target_col:
        return "⚠️ Please provide target column", None
    
    df = state["df"].copy()
    
    try:
        X = df.drop(columns=[target_col]).select_dtypes(include=np.number)
        y = df[target_col]
        
        # Use Random Forest for feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42) if y.dtype == 'object' or y.nunique() < 10 else RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
        top_features = importances.head(n_features).index.tolist()
        
        df = df[[target_col] + top_features]
        state["df"] = df
        
        return f"✅ Selected top {n_features} features by importance\nTop features: {top_features}", df.head(20)
    except Exception as e:
        return f"❌ Error: {e}", None

# 7. INTERACTION FEATURES (already exists, keeping it)
def create_interaction_feature(col1, col2):
    if state["df"] is None or not col1 or not col2:
        return "⚠️ Please select two columns", None
    
    df = state["df"].copy()
    df[f"{col1}_x_{col2}"] = df[col1] * df[col2]
    state["df"] = df
    return f"✅ Created interaction feature: {col1}_x_{col2}", df.head(20)

# 8. AGGREGATION FEATURES
def create_aggregation_features(group_col, agg_col, agg_funcs_text):
    if state["df"] is None or not group_col or not agg_col:
        return "⚠️ Please provide all inputs", None
    
    df = state["df"].copy()
    agg_funcs = [f.strip() for f in agg_funcs_text.split(',')]
    
    try:
        agg_dict = {}
        for func in agg_funcs:
            if func.lower() == "mean":
                agg_dict[f"{agg_col}_mean_by_{group_col}"] = df.groupby(group_col)[agg_col].transform('mean')
            elif func.lower() == "sum":
                agg_dict[f"{agg_col}_sum_by_{group_col}"] = df.groupby(group_col)[agg_col].transform('sum')
            elif func.lower() == "count":
                agg_dict[f"{agg_col}_count_by_{group_col}"] = df.groupby(group_col)[agg_col].transform('count')
            elif func.lower() == "max":
                agg_dict[f"{agg_col}_max_by_{group_col}"] = df.groupby(group_col)[agg_col].transform('max')
            elif func.lower() == "min":
                agg_dict[f"{agg_col}_min_by_{group_col}"] = df.groupby(group_col)[agg_col].transform('min')
            elif func.lower() == "std":
                agg_dict[f"{agg_col}_std_by_{group_col}"] = df.groupby(group_col)[agg_col].transform('std')
        
        for name, values in agg_dict.items():
            df[name] = values
        
        state["df"] = df
        return f"✅ Created {len(agg_dict)} aggregation features grouping by '{group_col}'", df.head(20)
    except Exception as e:
        return f"❌ Error: {e}", None

# TRANSFORMATIONS (already exist)
def apply_log_transform(column):
    if state["df"] is None or not column:
        return "⚠️ Please select a column", None
    
    df = state["df"].copy()
    df[f"{column}_log"] = np.log1p(df[column])
    state["df"] = df
    return f"✅ Log transform applied to {column}", df.head(20)

def plot_3d_scatter(x_col, y_col, z_col):
    if state["df"] is None or not x_col or not y_col or not z_col:
        return None
    
    df = state["df"]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(df[x_col], df[y_col], df[z_col], c='blue', marker='o', alpha=0.6)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)
    ax.set_title(f"3D Scatter Plot: {x_col}, {y_col}, {z_col}")
    
    plt.tight_layout()
    return fig

def group_and_aggregate(group_col, agg_col, agg_func):
    if state["df"] is None or not group_col or not agg_col:
        return "⚠️ Please select columns", None
    
    df = state["df"]
    
    try:
        if agg_func == "Mean":
            result = df.groupby(group_col)[agg_col].mean().reset_index()
        elif agg_func == "Sum":
            result = df.groupby(group_col)[agg_col].sum().reset_index()
        elif agg_func == "Count":
            result = df.groupby(group_col)[agg_col].count().reset_index()
        elif agg_func == "Min":
            result = df.groupby(group_col)[agg_col].min().reset_index()
        elif agg_func == "Max":
            result = df.groupby(group_col)[agg_col].max().reset_index()
        elif agg_func == "Median":
            result = df.groupby(group_col)[agg_col].median().reset_index()
        elif agg_func == "Std":
            result = df.groupby(group_col)[agg_col].std().reset_index()
        
        return f"✅ Grouped by '{group_col}' and computed {agg_func} of '{agg_col}'", result
    except Exception as e:
        return f"❌ Error: {e}", None

def create_pivot_table(index_col, column_col, value_col, agg_func):
    if state["df"] is None or not index_col or not column_col or not value_col:
        return "⚠️ Please select all columns", None
    
    df = state["df"]
    
    try:
        agg_map = {
            "Mean": "mean",
            "Sum": "sum",
            "Count": "count",
            "Min": "min",
            "Max": "max"
        }
        
        pivot = pd.pivot_table(df, values=value_col, index=index_col, 
                               columns=column_col, aggfunc=agg_map[agg_func], 
                               fill_value=0)
        
        return f"✅ Pivot table created: {index_col} × {column_col}, aggregating {value_col}", pivot
    except Exception as e:
        return f"❌ Error: {e}", None
    if state["df"] is None or not column:
        return "⚠️ Please select a column", None
    
    df = state["df"].copy()
    df[f"{column}_sqrt"] = np.sqrt(df[column].abs())
    state["df"] = df
    return f"✅ Square root transform applied to {column}", df.head(20)

# -----------------------------
# 5️⃣ Model Building
# -----------------------------
# -----------------------------
# 5️⃣ Model Building - COMPREHENSIVE
# -----------------------------

# 1. PREPARE DATA
def split_data(target_col, test_size, val_size):
    if state["df"] is None or not target_col:
        return "⚠️ Please upload data and select target column"
    
    df = state["df"]
    
    try:
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Handle non-numeric columns
        numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
        X = X[numeric_cols]
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size/100, random_state=42
        )
        
        # Second split: train vs val
        if val_size > 0:
            val_size_adjusted = val_size / (100 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size_adjusted, random_state=42
            )
            state["X_val"] = X_val
            state["y_val"] = y_val
        else:
            X_train, y_train = X_temp, y_temp
            state["X_val"] = None
            state["y_val"] = None
        
        state["X_train"] = X_train
        state["X_test"] = X_test
        state["y_train"] = y_train
        state["y_test"] = y_test
        
        result = f"""✅ Data split successfully!

**Dataset Distribution:**
- Training set: {len(X_train)} samples ({len(X_train)/len(df)*100:.1f}%)
- Validation set: {len(X_val) if state["X_val"] is not None else 0} samples ({len(X_val)/len(df)*100:.1f}% if state["X_val"] is not None else 0)
- Test set: {len(X_test)} samples ({len(X_test)/len(df)*100:.1f}%)
- Features: {len(numeric_cols)}
- Target: {target_col}
"""
        return result
    except Exception as e:
        return f"❌ Error: {e}"

# 2. TRAIN MODEL with Cross-Validation
def train_model_with_cv(model_type, task_type, use_cv, cv_folds):
    if state["X_train"] is None:
        return "⚠️ Please split data first"
    
    X_train = state["X_train"].copy()
    X_test = state["X_test"].copy()
    y_train = state["y_train"]
    y_test = state["y_test"]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = None
    
    try:
        # Select model
        if task_type == "Regression":
            if model_type == "Linear Regression":
                model = LinearRegression()
            elif model_type == "Random Forest Regressor":
                model = RandomForestRegressor(random_state=42, n_estimators=100)
            elif model_type == "XGBoost Regressor":
                from xgboost import XGBRegressor
                model = XGBRegressor(random_state=42, n_estimators=100)
        else:  # Classification
            if model_type == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)
            elif model_type == "Decision Tree":
                model = DecisionTreeClassifier(random_state=42)
            elif model_type == "Random Forest":
                model = RandomForestClassifier(random_state=42, n_estimators=100)
            elif model_type == "SVM":
                model = SVC(probability=True)
            elif model_type == "KNN":
                model = KNeighborsClassifier()
            elif model_type == "XGBoost Classifier":
                from xgboost import XGBClassifier
                model = XGBClassifier(random_state=42, n_estimators=100)
            elif model_type == "Neural Network":
                from sklearn.neural_network import MLPClassifier
                model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        
        # Cross-validation
        cv_results = ""
        if use_cv:
            from sklearn.model_selection import cross_val_score
            if task_type == "Regression":
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds, scoring='r2')
                cv_results = f"\n**{cv_folds}-Fold Cross-Validation:**\n- Mean R² Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})\n- Scores: {cv_scores}\n"
            else:
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds, scoring='accuracy')
                cv_results = f"\n**{cv_folds}-Fold Cross-Validation:**\n- Mean Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})\n- Scores: {cv_scores}\n"
        
        # Train model
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Evaluate
        if task_type == "Regression":
            from sklearn.metrics import mean_absolute_error
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            state["model"] = model
            state["scaler"] = scaler
            return f"""✅ Model trained successfully!

**Model:** {model_type}
**Task:** {task_type}
{cv_results}
**Test Set Performance:**
- MSE: {mse:.4f}
- RMSE: {rmse:.4f}
- MAE: {mae:.4f}
- R² Score: {r2:.4f}

**Interpretation:**
- R² = {r2:.2%} of variance explained
- Average error: {mae:.4f}
"""
        else:  # Classification
            from sklearn.metrics import confusion_matrix, roc_auc_score
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # ROC-AUC (for binary classification)
            roc_auc = ""
            if len(np.unique(y_test)) == 2:
                try:
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                    auc = roc_auc_score(y_test, y_pred_proba)
                    roc_auc = f"\n- ROC-AUC: {auc:.4f}"
                except:
                    pass
            
            state["model"] = model
            state["scaler"] = scaler
            state["confusion_matrix"] = cm
            state["y_test"] = y_test
            state["y_pred"] = y_pred
            
            return f"""✅ Model trained successfully!

**Model:** {model_type}
**Task:** {task_type}
{cv_results}
**Test Set Performance:**
- Accuracy: {acc:.4f} ({acc:.2%}){roc_auc}

**Classification Report:**
{report}

**Confusion Matrix:**
{cm}
"""
    except Exception as e:
        return f"❌ Error training model: {e}"

# 3. HYPERPARAMETER TUNING
def hyperparameter_tuning(model_type, task_type, search_type, n_iter):
    if state["X_train"] is None:
        return "⚠️ Please split data first"
    
    X_train = state["X_train"].copy()
    y_train = state["y_train"]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    try:
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
        
        # Define model and parameter grid
        if task_type == "Regression":
            if model_type == "Random Forest Regressor":
                model = RandomForestRegressor(random_state=42)
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 20, None],
                    'min_samples_split': [2, 5, 10]
                }
            elif model_type == "XGBoost Regressor":
                from xgboost import XGBRegressor
                model = XGBRegressor(random_state=42)
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.3]
                }
            else:
                return "⚠️ Hyperparameter tuning not available for this model"
        else:  # Classification
            if model_type == "Random Forest":
                model = RandomForestClassifier(random_state=42)
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 20, None],
                    'min_samples_split': [2, 5, 10]
                }
            elif model_type == "XGBoost Classifier":
                from xgboost import XGBClassifier
                model = XGBClassifier(random_state=42)
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.3]
                }
            elif model_type == "SVM":
                model = SVC(probability=True)
                param_grid = {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto']
                }
            else:
                return "⚠️ Hyperparameter tuning not available for this model"
        
        # Perform search
        if search_type == "Grid Search":
            search = GridSearchCV(model, param_grid, cv=5, scoring='r2' if task_type == "Regression" else 'accuracy', n_jobs=-1)
        else:  # Random Search
            search = RandomizedSearchCV(model, param_grid, n_iter=n_iter, cv=5, scoring='r2' if task_type == "Regression" else 'accuracy', random_state=42, n_jobs=-1)
        
        search.fit(X_train_scaled, y_train)
        
        state["model"] = search.best_estimator_
        state["scaler"] = scaler
        
        return f"""✅ Hyperparameter tuning completed!

**Search Type:** {search_type}
**Best Parameters:** {search.best_params_}
**Best Score:** {search.best_score_:.4f}

**Top 5 Parameter Combinations:**
{pd.DataFrame(search.cv_results_).nlargest(5, 'mean_test_score')[['params', 'mean_test_score', 'std_test_score']].to_string()}
"""
    except Exception as e:
        return f"❌ Error: {e}"

# 4. UNSUPERVISED LEARNING
def train_clustering(n_clusters, algorithm):
    if state["df"] is None:
        return "⚠️ Please upload data first", None, None
    
    df = state["df"].copy()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    X = df[numeric_cols]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    try:
        if algorithm == "KMeans":
            model = KMeans(n_clusters=n_clusters, random_state=42)
        elif algorithm == "DBSCAN":
            from sklearn.cluster import DBSCAN
            model = DBSCAN(eps=0.5, min_samples=5)
        elif algorithm == "Agglomerative":
            from sklearn.cluster import AgglomerativeClustering
            model = AgglomerativeClustering(n_clusters=n_clusters)
        
        clusters = model.fit_predict(X_scaled)
        
        df["Cluster"] = clusters
        state["df"] = df
        state["model"] = model
        
        # Calculate metrics
        from sklearn.metrics import silhouette_score, davies_bouldin_score
        silhouette = silhouette_score(X_scaled, clusters)
        davies_bouldin = davies_bouldin_score(X_scaled, clusters)
        
        # Cluster distribution
        cluster_counts = pd.Series(clusters).value_counts().sort_index()
        
        result = f"""✅ {algorithm} clustering completed!

**Clusters:** {n_clusters if algorithm != "DBSCAN" else len(set(clusters)) - (1 if -1 in clusters else 0)}
**Silhouette Score:** {silhouette:.4f} (Higher is better, range: -1 to 1)
**Davies-Bouldin Index:** {davies_bouldin:.4f} (Lower is better)

**Cluster Distribution:**
"""
        for i, count in cluster_counts.items():
            result += f"\nCluster {i}: {count} samples ({count/len(clusters)*100:.1f}%)"
        
        # Create visualization
        if X.shape[1] >= 2:
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis', alpha=0.6)
            ax.set_title(f'{algorithm} Clustering Results')
            ax.set_xlabel('Feature 1 (scaled)')
            ax.set_ylabel('Feature 2 (scaled)')
            plt.colorbar(scatter, ax=ax, label='Cluster')
            plt.tight_layout()
        else:
            fig = None
        
        return result, df.head(20), fig
    except Exception as e:
        return f"❌ Error: {e}", None, None

def apply_dimensionality_reduction(method, n_components):
    if state["df"] is None:
        return "⚠️ Please upload data first", None, None
    
    df = state["df"].copy()
    numeric_df = df.select_dtypes(include=np.number)
    
    if numeric_df.shape[1] < 2:
        return "❌ Need at least 2 numeric columns", None, None
    
    try:
        if method == "PCA":
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=n_components)
        elif method == "t-SNE":
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=min(n_components, 3), random_state=42)
        
        reduced_features = reducer.fit_transform(numeric_df)
        
        reduced_df = pd.DataFrame(
            reduced_features,
            columns=[f"{method}{i+1}" for i in range(reduced_features.shape[1])],
            index=df.index
        )
        
        # Keep non-numeric columns
        non_numeric = df.select_dtypes(exclude=np.number)
        df = pd.concat([non_numeric, reduced_df], axis=1)
        
        state["df"] = df
        
        # Calculate variance explained (for PCA)
        var_explained = ""
        if method == "PCA":
            variance = reducer.explained_variance_ratio_
            var_explained = f"\n**Variance Explained:**\n"
            for i, var in enumerate(variance):
                var_explained += f"- {method}{i+1}: {var:.2%}\n"
            var_explained += f"- Total: {variance.sum():.2%}"
        
        # Visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        if reduced_features.shape[1] >= 2:
            ax.scatter(reduced_features[:, 0], reduced_features[:, 1], alpha=0.6)
            ax.set_xlabel(f'{method}1')
            ax.set_ylabel(f'{method}2')
            ax.set_title(f'{method} Visualization')
        plt.tight_layout()
        
        return f"""✅ {method} applied successfully!

**Original dimensions:** {numeric_df.shape[1]}
**Reduced dimensions:** {n_components}
{var_explained}
""", df.head(20), fig
    except Exception as e:
        return f"❌ Error: {e}", None, None

# 5. MODEL EVALUATION VISUALIZATIONS
def plot_confusion_matrix():
    if "confusion_matrix" not in state or state["confusion_matrix"] is None:
        return None
    
    cm = state["confusion_matrix"]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    plt.tight_layout()
    return fig

def plot_roc_curve():
    if state["model"] is None or state["X_test"] is None:
        return None
    
    try:
        from sklearn.metrics import roc_curve, auc
        
        X_test_scaled = state["scaler"].transform(state["X_test"])
        y_test = state["y_test"]
        
        # Binary classification only
        if len(np.unique(y_test)) != 2:
            return None
        
        y_pred_proba = state["model"].predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc="lower right")
        plt.tight_layout()
        return fig
    except:
        return None

def plot_feature_importance():
    if state["model"] is None or not hasattr(state["model"], 'feature_importances_'):
        return None
    
    try:
        importances = state["model"].feature_importances_
        feature_names = state["X_train"].columns
        
        indices = np.argsort(importances)[::-1][:20]  # Top 20
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(indices)), importances[indices])
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Importance')
        ax.set_title('Top 20 Feature Importances')
        ax.invert_yaxis()
        plt.tight_layout()
        return fig
    except:
        return None

# 6. MODEL DEPLOYMENT
def save_model():
    if state["model"] is None:
        return "⚠️ Train a model first"
    
    try:
        # Save model and scaler
        joblib.dump(state["model"], "trained_model.pkl")
        if "scaler" in state and state["scaler"] is not None:
            joblib.dump(state["scaler"], "scaler.pkl")
        return "✅ Model saved as 'trained_model.pkl'\n✅ Scaler saved as 'scaler.pkl'"
    except Exception as e:
        return f"❌ Error: {e}"

def load_model(model_file, scaler_file):
    try:
        if model_file is not None:
            model = joblib.load(model_file.name)
            state["model"] = model
            msg = "✅ Model loaded successfully!"
            
            if scaler_file is not None:
                scaler = joblib.load(scaler_file.name)
                state["scaler"] = scaler
                msg += "\n✅ Scaler loaded successfully!"
            
            return msg
        return "⚠️ Please upload a model file"
    except Exception as e:
        return f"❌ Error: {e}"

def split_data(target_col, test_size):
    if state["X_train"] is None:
        return "⚠️ Please split data first"
    
    X_train = state["X_train"].copy()
    X_test = state["X_test"].copy()
    y_train = state["y_train"]
    y_test = state["y_test"]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = None
    
    try:
        if task_type == "Regression":
            if model_type == "Linear Regression":
                model = LinearRegression()
            elif model_type == "Random Forest Regressor":
                model = RandomForestRegressor(random_state=42, n_estimators=100)
            
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            state["model"] = model
            return f"""✅ Model trained successfully!

**Regression Metrics:**
- MSE: {mse:.4f}
- RMSE: {rmse:.4f}
- R² Score: {r2:.4f}

**Interpretation:**
- R² = {r2:.2%} of variance explained
"""
        
        else:  # Classification
            if model_type == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)
            elif model_type == "Decision Tree":
                model = DecisionTreeClassifier(random_state=42)
            elif model_type == "Random Forest":
                model = RandomForestClassifier(random_state=42, n_estimators=100)
            elif model_type == "SVM":
                model = SVC()
            elif model_type == "KNN":
                model = KNeighborsClassifier()
            
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            
            state["model"] = model
            return f"""✅ Model trained successfully!

**Classification Metrics:**
Accuracy: {acc:.4f} ({acc:.2%})

**Classification Report:**
{report}
"""
    
    except Exception as e:
        return f"❌ Error training model: {e}"

def train_clustering(n_clusters):
    if state["df"] is None:
        return "⚠️ Please upload data first", None
    
    df = state["df"].copy()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    X = df[numeric_cols]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = model.fit_predict(X_scaled)
    
    df["Cluster"] = clusters
    state["df"] = df
    state["model"] = model
    
    # Calculate cluster statistics
    cluster_counts = pd.Series(clusters).value_counts().sort_index()
    
    result = f"✅ KMeans clustering completed with {n_clusters} clusters\n\n**Cluster Distribution:**\n"
    for i, count in cluster_counts.items():
        result += f"Cluster {i}: {count} samples ({count/len(clusters)*100:.1f}%)\n"
    
    return result, df.head(20)

def save_model():
    if state["model"] is None:
        return "⚠️ Train a model first"
    
    joblib.dump(state["model"], "trained_model.pkl")
    return "✅ Model saved as 'trained_model.pkl'"

def download_data():
    if state["df"] is None:
        return None
    
    return state["df"]

# -----------------------------
# Gradio Interface
# -----------------------------
with gr.Blocks(title="End-to-End ML Pipeline", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Complete End-to-End ML Pipeline")
    gr.Markdown("Comprehensive data processing, cleaning, analysis, and machine learning platform")
    
    with gr.Tabs():
        # ==================== TAB 1: DATA UPLOAD ====================
        with gr.Tab("📂 Data Upload"):
            gr.Markdown("### Upload your dataset (CSV, Excel, or JSON)")
            with gr.Row():
                file_input = gr.File(label="Upload Dataset")
            upload_btn = gr.Button("📤 Upload Data", variant="primary", size="lg")
            upload_output = gr.Textbox(label="Upload Status", lines=10)
            data_preview = gr.Dataframe(label="Data Preview", height=400)
            
            upload_btn.click(upload_data, inputs=[file_input], outputs=[upload_output, data_preview])
        
        # ==================== TAB 2: DATA CLEANING ====================
        with gr.Tab("🧹 Data Cleaning"):
            
            # 1. Missing Data
            with gr.Accordion("1️⃣ Handling Missing Data", open=True):
                gr.Markdown("Fill or remove missing values using various strategies")
                with gr.Row():
                    missing_action = gr.Dropdown(
                        ["Drop Rows", "Fill with Mean", "Fill with Median", "Fill with Mode", 
                         "Forward Fill", "Backward Fill"],
                        label="Missing Data Strategy",
                        value="Fill with Mean"
                    )
                    missing_btn = gr.Button("Apply", variant="primary")
                missing_output = gr.Textbox(label="Status")
                missing_data = gr.Dataframe(label="Updated Data", height=300)
                missing_btn.click(handle_missing_data, inputs=[missing_action], 
                                outputs=[missing_output, missing_data])
            
            # 2. Duplicates
            with gr.Accordion("2️⃣ Handling Duplicate Data"):
                gr.Markdown("Remove duplicate rows from dataset")
                dup_btn = gr.Button("🗑️ Remove Duplicates", variant="primary")
                dup_output = gr.Textbox(label="Status")
                dup_data = gr.Dataframe(label="Updated Data", height=300)
                dup_btn.click(remove_duplicates, outputs=[dup_output, dup_data])
            
            # 3. Outliers
            with gr.Accordion("3️⃣ Handling Outliers"):
                gr.Markdown("Detect and handle outliers using statistical methods")
                
                gr.Markdown("**Method 1: IQR (Interquartile Range)**")
                with gr.Row():
                    outlier_col_iqr = gr.Textbox(label="Column Name", placeholder="Enter numeric column")
                    outlier_btn_iqr = gr.Button("Remove Outliers (IQR)")
                outlier_output_iqr = gr.Textbox(label="Status")
                outlier_data_iqr = gr.Dataframe(label="Updated Data", height=300)
                outlier_btn_iqr.click(remove_outliers_iqr, inputs=[outlier_col_iqr], 
                                     outputs=[outlier_output_iqr, outlier_data_iqr])
                
                gr.Markdown("**Method 2: Z-Score**")
                with gr.Row():
                    outlier_col_z = gr.Textbox(label="Column Name", placeholder="Enter numeric column")
                    z_threshold = gr.Slider(1, 5, value=3, step=0.5, label="Z-Score Threshold")
                    outlier_btn_z = gr.Button("Remove Outliers (Z-Score)")
                outlier_output_z = gr.Textbox(label="Status")
                outlier_data_z = gr.Dataframe(label="Updated Data", height=300)
                outlier_btn_z.click(remove_outliers_zscore, inputs=[outlier_col_z, z_threshold], 
                                   outputs=[outlier_output_z, outlier_data_z])
                
                gr.Markdown("**Method 3: Cap/Floor (Winsorization)**")
                with gr.Row():
                    outlier_col_cap = gr.Textbox(label="Column Name", placeholder="Enter numeric column")
                    outlier_btn_cap = gr.Button("Cap/Floor Outliers")
                outlier_output_cap = gr.Textbox(label="Status")
                outlier_data_cap = gr.Dataframe(label="Updated Data", height=300)
                outlier_btn_cap.click(cap_floor_outliers, inputs=[outlier_col_cap], 
                                     outputs=[outlier_output_cap, outlier_data_cap])
            
            # 4. Inconsistent Data
            with gr.Accordion("4️⃣ Correcting Inconsistent Data"):
                gr.Markdown("Standardize text formatting and fix inconsistent categories")
                
                gr.Markdown("**Auto-standardize all text columns**")
                std_btn = gr.Button("🔧 Standardize Text Data", variant="primary")
                std_output = gr.Textbox(label="Status")
                std_data = gr.Dataframe(label="Updated Data", height=300)
                std_btn.click(standardize_text_data, outputs=[std_output, std_data])
                
                gr.Markdown("**Fix specific category mappings**")
                gr.Markdown("Format: `old1:new1, old2:new2` (Example: male:Male, m:Male, female:Female)")
                with gr.Row():
                    fix_col = gr.Textbox(label="Column Name")
                    fix_mapping = gr.Textbox(label="Mapping", placeholder="m:Male, f:Female")
                    fix_btn = gr.Button("Apply Mapping")
                fix_output = gr.Textbox(label="Status")
                fix_data = gr.Dataframe(label="Updated Data", height=300)
                fix_btn.click(fix_inconsistent_categories, inputs=[fix_col, fix_mapping], 
                            outputs=[fix_output, fix_data])
            
            # 5. Noisy Data
            with gr.Accordion("5️⃣ Handling Noisy Data"):
                gr.Markdown("Smooth noisy data using binning or moving averages")
                
                gr.Markdown("**Binning**")
                with gr.Row():
                    bin_col = gr.Textbox(label="Column Name", placeholder="Enter numeric column")
                    n_bins = gr.Slider(2, 20, value=5, step=1, label="Number of Bins")
                    bin_btn = gr.Button("Apply Binning")
                bin_output = gr.Textbox(label="Status")
                bin_data = gr.Dataframe(label="Updated Data", height=300)
                bin_btn.click(smooth_data_binning, inputs=[bin_col, n_bins], 
                            outputs=[bin_output, bin_data])
                
                gr.Markdown("**Moving Average**")
                with gr.Row():
                    ma_col = gr.Textbox(label="Column Name", placeholder="Enter numeric column")
                    ma_window = gr.Slider(2, 50, value=5, step=1, label="Window Size")
                    ma_btn = gr.Button("Apply Moving Average")
                ma_output = gr.Textbox(label="Status")
                ma_data = gr.Dataframe(label="Updated Data", height=300)
                ma_btn.click(smooth_data_moving_average, inputs=[ma_col, ma_window], 
                           outputs=[ma_output, ma_data])
            
            # 6. Scaling/Normalization
            with gr.Accordion("6️⃣ Data Transformation (Scaling/Normalization)"):
                gr.Markdown("Scale numeric features for consistency")
                with gr.Row():
                    scale_method = gr.Dropdown(
                        ["Standardization (Z-score)", "Min-Max Normalization", "Robust Scaling"], 
                        label="Scaling Method"
                    )
                    scale_btn = gr.Button("Apply Scaling", variant="primary")
                scale_output = gr.Textbox(label="Status")
                scale_data_out = gr.Dataframe(label="Updated Data", height=300)
                scale_btn.click(scale_data, inputs=[scale_method], outputs=[scale_output, scale_data_out])
            
            # 7. Remove Irrelevant Columns
            with gr.Accordion("7️⃣ Removing Irrelevant or Redundant Features"):
                gr.Markdown("Drop columns that are not useful for analysis")
                gr.Markdown("Enter column names separated by commas (e.g., `id, name, timestamp`)")
                with gr.Row():
                    drop_cols_text = gr.Textbox(label="Columns to Drop", placeholder="col1, col2, col3")
                    drop_btn = gr.Button("🗑️ Drop Columns", variant="stop")
                drop_output = gr.Textbox(label="Status")
                drop_data = gr.Dataframe(label="Updated Data", height=300)
                drop_btn.click(drop_columns, inputs=[drop_cols_text], outputs=[drop_output, drop_data])
            
            # 8. Validation
            with gr.Accordion("8️⃣ Validation and Verification"):
                gr.Markdown("Check data integrity and identify potential issues")
                validate_btn = gr.Button("📊 Validate Data", variant="primary", size="lg")
                validate_output = gr.Markdown(label="Validation Report")
                validate_btn.click(validate_data, outputs=[validate_output])
        
        # ==================== TAB 3: EDA ====================
        with gr.Tab("📈 Exploratory Data Analysis"):
            
            with gr.Accordion("1️⃣ Descriptive Statistics", open=True):
                gr.Markdown("Comprehensive statistical summary including skewness, kurtosis, and correlations")
                stats_btn = gr.Button("📊 Generate Complete Statistics", variant="primary", size="lg")
                stats_output = gr.Markdown(label="Statistics")
                stats_btn.click(get_statistics, outputs=[stats_output])
            
            with gr.Accordion("2️⃣ Univariate Analysis (Single Variable)"):
                gr.Markdown("### Numerical Variables")
                
                gr.Markdown("**Histogram with KDE**")
                with gr.Row():
                    hist_col = gr.Textbox(label="Column", placeholder="Enter numeric column")
                    hist_btn = gr.Button("Plot Histogram")
                hist_plot = gr.Plot(label="Histogram")
                hist_btn.click(plot_histogram, inputs=[hist_col], outputs=[hist_plot])
                
                gr.Markdown("**Density Plot**")
                with gr.Row():
                    density_col = gr.Textbox(label="Column", placeholder="Enter numeric column")
                    density_btn = gr.Button("Plot Density")
                density_plot = gr.Plot(label="Density Plot")
                density_btn.click(plot_density, inputs=[density_col], outputs=[density_plot])
                
                gr.Markdown("**Boxplot (Outlier Detection)**")
                with gr.Row():
                    box_col = gr.Textbox(label="Column", placeholder="Enter numeric column")
                    box_btn = gr.Button("Plot Boxplot")
                box_plot = gr.Plot(label="Boxplot")
                box_btn.click(plot_boxplot, inputs=[box_col], outputs=[box_plot])
                
                gr.Markdown("### Categorical Variables")
                
                gr.Markdown("**Bar Chart**")
                with gr.Row():
                    bar_col = gr.Textbox(label="Column", placeholder="Enter categorical column")
                    bar_btn = gr.Button("Plot Bar Chart")
                bar_plot = gr.Plot(label="Bar Chart")
                bar_btn.click(plot_bar_chart, inputs=[bar_col], outputs=[bar_plot])
                
                gr.Markdown("**Pie Chart**")
                with gr.Row():
                    pie_col = gr.Textbox(label="Column", placeholder="Enter categorical column")
                    pie_btn = gr.Button("Plot Pie Chart")
                pie_plot = gr.Plot(label="Pie Chart")
                pie_btn.click(plot_pie_chart, inputs=[pie_col], outputs=[pie_plot])
            
            with gr.Accordion("3️⃣ Bivariate Analysis (Two Variables)"):
                gr.Markdown("### Numerical vs Numerical")
                
                gr.Markdown("**Scatter Plot**")
                with gr.Row():
                    scatter_x = gr.Textbox(label="X Column", placeholder="Enter numeric column")
                    scatter_y = gr.Textbox(label="Y Column", placeholder="Enter numeric column")
                    scatter_btn = gr.Button("Plot Scatter")
                scatter_plot = gr.Plot(label="Scatter Plot")
                scatter_btn.click(plot_scatter, inputs=[scatter_x, scatter_y], outputs=[scatter_plot])
                
                gr.Markdown("**Line Plot (Time Series/Trends)**")
                with gr.Row():
                    line_x = gr.Textbox(label="X Column", placeholder="Enter column")
                    line_y = gr.Textbox(label="Y Column", placeholder="Enter numeric column")
                    line_btn = gr.Button("Plot Line")
                line_plot = gr.Plot(label="Line Plot")
                line_btn.click(plot_line, inputs=[line_x, line_y], outputs=[line_plot])
                
                gr.Markdown("### Numerical vs Categorical")
                
                gr.Markdown("**Boxplot by Category**")
                with gr.Row():
                    box_cat = gr.Textbox(label="Categorical Column", placeholder="Enter categorical column")
                    box_num = gr.Textbox(label="Numerical Column", placeholder="Enter numeric column")
                    box_bi_btn = gr.Button("Plot Boxplot")
                box_bi_plot = gr.Plot(label="Boxplot by Category")
                box_bi_btn.click(plot_boxplot_bivariate, inputs=[box_cat, box_num], outputs=[box_bi_plot])
                
                gr.Markdown("**Violin Plot**")
                with gr.Row():
                    violin_cat = gr.Textbox(label="Categorical Column", placeholder="Enter categorical column")
                    violin_num = gr.Textbox(label="Numerical Column", placeholder="Enter numeric column")
                    violin_btn = gr.Button("Plot Violin")
                violin_plot = gr.Plot(label="Violin Plot")
                violin_btn.click(plot_violin, inputs=[violin_cat, violin_num], outputs=[violin_plot])
                
                gr.Markdown("### Categorical vs Categorical")
                
                gr.Markdown("**Stacked Bar Chart**")
                with gr.Row():
                    stack_cat1 = gr.Textbox(label="Categorical Column 1", placeholder="Enter column")
                    stack_cat2 = gr.Textbox(label="Categorical Column 2", placeholder="Enter column")
                    stack_btn = gr.Button("Plot Stacked Bar")
                stack_plot = gr.Plot(label="Stacked Bar Chart")
                stack_btn.click(plot_stacked_bar, inputs=[stack_cat1, stack_cat2], outputs=[stack_plot])
                
                gr.Markdown("**Categorical Heatmap (Frequency Table)**")
                with gr.Row():
                    heat_cat1 = gr.Textbox(label="Categorical Column 1", placeholder="Enter column")
                    heat_cat2 = gr.Textbox(label="Categorical Column 2", placeholder="Enter column")
                    heat_cat_btn = gr.Button("Plot Heatmap")
                heat_cat_plot = gr.Plot(label="Categorical Heatmap")
                heat_cat_btn.click(plot_categorical_heatmap, inputs=[heat_cat1, heat_cat2], outputs=[heat_cat_plot])
            
            with gr.Accordion("4️⃣ Multivariate Analysis (Multiple Variables)"):
                gr.Markdown("**Correlation Heatmap**")
                heatmap_btn = gr.Button("🔥 Generate Correlation Heatmap", variant="primary")
                heatmap_plot = gr.Plot(label="Correlation Heatmap")
                heatmap_btn.click(plot_correlation_heatmap, outputs=[heatmap_plot])
                
                gr.Markdown("**Pairplot (All Numeric Pairs)**")
                gr.Markdown("*Note: Limited to first 5 numeric columns for performance*")
                pairplot_btn = gr.Button("Generate Pairplot")
                pairplot_plot = gr.Plot(label="Pairplot")
                pairplot_btn.click(plot_pairplot, outputs=[pairplot_plot])
                
                gr.Markdown("**3D Scatter Plot**")
                with gr.Row():
                    scatter_3d_x = gr.Textbox(label="X Column", placeholder="Enter numeric column")
                    scatter_3d_y = gr.Textbox(label="Y Column", placeholder="Enter numeric column")
                    scatter_3d_z = gr.Textbox(label="Z Column", placeholder="Enter numeric column")
                    scatter_3d_btn = gr.Button("Plot 3D Scatter")
                scatter_3d_plot = gr.Plot(label="3D Scatter Plot")
                scatter_3d_btn.click(plot_3d_scatter, inputs=[scatter_3d_x, scatter_3d_y, scatter_3d_z], 
                                    outputs=[scatter_3d_plot])
            
            with gr.Accordion("5️⃣ Missing Data Analysis"):
                gr.Markdown("Visualize missing values in your dataset")
                
                gr.Markdown("**Missing Values Heatmap**")
                miss_heat_btn = gr.Button("Generate Missing Values Heatmap")
                miss_heat_plot = gr.Plot(label="Missing Values Heatmap")
                miss_heat_btn.click(plot_missing_heatmap, outputs=[miss_heat_plot])
                
                gr.Markdown("**Missing Values Bar Chart**")
                miss_bar_btn = gr.Button("Generate Missing Values Bar Chart")
                miss_bar_plot = gr.Plot(label="Missing Values Bar Chart")
                miss_bar_btn.click(plot_missing_bar, outputs=[miss_bar_plot])
            
            with gr.Accordion("6️⃣ Data Grouping & Aggregation"):
                gr.Markdown("### Group By and Aggregate")
                gr.Markdown("Group data by a category and compute statistics")
                with gr.Row():
                    group_col = gr.Textbox(label="Group By Column", placeholder="Enter categorical column")
                    agg_col = gr.Textbox(label="Aggregate Column", placeholder="Enter numeric column")
                    agg_func = gr.Dropdown(["Mean", "Sum", "Count", "Min", "Max", "Median", "Std"], 
                                          label="Aggregation Function", value="Mean")
                    group_btn = gr.Button("Group & Aggregate")
                group_output = gr.Textbox(label="Status")
                group_data = gr.Dataframe(label="Grouped Data", height=300)
                group_btn.click(group_and_aggregate, inputs=[group_col, agg_col, agg_func], 
                              outputs=[group_output, group_data])
                
                gr.Markdown("### Pivot Table (Cross-Tabulation)")
                gr.Markdown("Create a pivot table for multi-dimensional analysis")
                with gr.Row():
                    pivot_index = gr.Textbox(label="Index (Rows)", placeholder="Enter column")
                    pivot_column = gr.Textbox(label="Columns", placeholder="Enter column")
                    pivot_value = gr.Textbox(label="Values", placeholder="Enter numeric column")
                    pivot_agg = gr.Dropdown(["Mean", "Sum", "Count", "Min", "Max"], 
                                           label="Aggregation", value="Mean")
                    pivot_btn = gr.Button("Create Pivot Table")
                pivot_output = gr.Textbox(label="Status")
                pivot_data = gr.Dataframe(label="Pivot Table", height=300)
                pivot_btn.click(create_pivot_table, inputs=[pivot_index, pivot_column, pivot_value, pivot_agg], 
                              outputs=[pivot_output, pivot_data])
        
        # ==================== TAB 4: FEATURE ENGINEERING ====================
        with gr.Tab("⚙️ Feature Engineering"):
            
            with gr.Accordion("1️⃣ Feature Creation", open=True):
                gr.Markdown("### Combine Existing Features")
                gr.Markdown("Create new features by combining two existing features")
                with gr.Row():
                    comb_col1 = gr.Textbox(label="Column 1", placeholder="Enter column name")
                    comb_col2 = gr.Textbox(label="Column 2", placeholder="Enter column name")
                    comb_op = gr.Dropdown(["Add (+)", "Subtract (-)", "Multiply (*)", "Divide (/)"], 
                                         label="Operation", value="Add (+)")
                    comb_name = gr.Textbox(label="New Feature Name", placeholder="new_feature")
                    comb_btn = gr.Button("Create Feature")
                comb_output = gr.Textbox(label="Status")
                comb_data = gr.Dataframe(label="Updated Data", height=300)
                comb_btn.click(combine_features, inputs=[comb_col1, comb_col2, comb_op, comb_name], 
                             outputs=[comb_output, comb_data])
                
                gr.Markdown("### Create Ratio Features")
                gr.Markdown("Create ratios like `Debt_to_Income = Debt / Income`")
                with gr.Row():
                    ratio_num = gr.Textbox(label="Numerator", placeholder="Column name")
                    ratio_den = gr.Textbox(label="Denominator", placeholder="Column name")
                    ratio_name = gr.Textbox(label="New Feature Name", placeholder="ratio_feature")
                    ratio_btn = gr.Button("Create Ratio")
                ratio_output = gr.Textbox(label="Status")
                ratio_data = gr.Dataframe(label="Updated Data", height=300)
                ratio_btn.click(create_ratio_feature, inputs=[ratio_num, ratio_den, ratio_name], 
                              outputs=[ratio_output, ratio_data])
            
            with gr.Accordion("2️⃣ Feature Transformation"):
                gr.Markdown("### Polynomial Features")
                gr.Markdown("Create polynomial and interaction terms for non-linear relationships")
                with gr.Row():
                    poly_cols = gr.Textbox(label="Columns (comma-separated)", placeholder="col1, col2, col3")
                    poly_degree = gr.Slider(2, 4, value=2, step=1, label="Polynomial Degree")
                    poly_btn = gr.Button("Create Polynomial Features")
                poly_output = gr.Textbox(label="Status")
                poly_data = gr.Dataframe(label="Updated Data", height=300)
                poly_btn.click(create_polynomial_features, inputs=[poly_cols, poly_degree], 
                             outputs=[poly_output, poly_data])
                
                gr.Markdown("### Log Transformation (Reduce Skewness)")
                with gr.Row():
                    log_col = gr.Textbox(label="Column Name", placeholder="Enter numeric column")
                    log_btn = gr.Button("Apply Log Transform")
                log_output = gr.Textbox(label="Status")
                log_data = gr.Dataframe(label="Updated Data", height=300)
                log_btn.click(apply_log_transform, inputs=[log_col], outputs=[log_output, log_data])
                
                gr.Markdown("### Square Root Transformation")
                with gr.Row():
                    sqrt_col = gr.Textbox(label="Column Name", placeholder="Enter numeric column")
                    sqrt_btn = gr.Button("Apply Sqrt Transform")
                sqrt_output = gr.Textbox(label="Status")
                sqrt_data = gr.Dataframe(label="Updated Data", height=300)
                sqrt_btn.click(apply_sqrt_transform, inputs=[sqrt_col], outputs=[sqrt_output, sqrt_data])
            
            with gr.Accordion("3️⃣ Encoding Categorical Features"):
                gr.Markdown("### Standard Encoding Methods")
                with gr.Row():
                    encode_col = gr.Textbox(label="Column Name", placeholder="Enter categorical column")
                    encode_method = gr.Dropdown(["One-Hot", "Label", "Frequency"], 
                                               label="Encoding Method", value="One-Hot")
                    encode_btn = gr.Button("Apply Encoding")
                encode_output = gr.Textbox(label="Status")
                encode_data = gr.Dataframe(label="Updated Data", height=300)
                encode_btn.click(encode_categorical, inputs=[encode_col, encode_method], 
                               outputs=[encode_output, encode_data])
                
                gr.Markdown("### Target Encoding (Supervised)")
                gr.Markdown("Replace categories with mean target value - useful for high-cardinality features")
                with gr.Row():
                    target_cat_col = gr.Textbox(label="Categorical Column", placeholder="Enter column")
                    target_target_col = gr.Textbox(label="Target Column", placeholder="Enter target column")
                    target_encode_btn = gr.Button("Apply Target Encoding")
                target_encode_output = gr.Textbox(label="Status")
                target_encode_data = gr.Dataframe(label="Updated Data", height=300)
                target_encode_btn.click(target_encode, inputs=[target_cat_col, target_target_col], 
                                      outputs=[target_encode_output, target_encode_data])
            
            with gr.Accordion("4️⃣ Handling Date/Time Features"):
                gr.Markdown("### Extract Date/Time Components")
                gr.Markdown("Extract year, month, day, hour, weekday, quarter, week")
                with gr.Row():
                    dt_col = gr.Textbox(label="DateTime Column", placeholder="Enter datetime column")
                    dt_extract_btn = gr.Button("Extract DateTime Features")
                dt_extract_output = gr.Textbox(label="Status")
                dt_extract_data = gr.Dataframe(label="Updated Data", height=300)
                dt_extract_btn.click(extract_datetime_features, inputs=[dt_col], 
                                   outputs=[dt_extract_output, dt_extract_data])
                
                gr.Markdown("### Cyclical Encoding for Periodic Features")
                gr.Markdown("Encode periodic features like month, hour using sin/cos transformations")
                with gr.Row():
                    cyc_col = gr.Textbox(label="DateTime Column", placeholder="Enter datetime column")
                    cyc_component = gr.Dropdown(["Month", "Day", "Hour", "DayOfWeek"], 
                                               label="Component to Encode", value="Month")
                    cyc_btn = gr.Button("Apply Cyclical Encoding")
                cyc_output = gr.Textbox(label="Status")
                cyc_data = gr.Dataframe(label="Updated Data", height=300)
                cyc_btn.click(cyclical_encode_datetime, inputs=[cyc_col, cyc_component], 
                            outputs=[cyc_output, cyc_data])
            
            with gr.Accordion("5️⃣ Handling Text Features"):
                gr.Markdown("### Extract Basic Text Statistics")
                gr.Markdown("Length, word count, character count, average word length")
                with gr.Row():
                    text_col = gr.Textbox(label="Text Column", placeholder="Enter text column")
                    text_extract_btn = gr.Button("Extract Text Features")
                text_extract_output = gr.Textbox(label="Status")
                text_extract_data = gr.Dataframe(label="Updated Data", height=300)
                text_extract_btn.click(extract_text_features, inputs=[text_col], 
                                     outputs=[text_extract_output, text_extract_data])
                
                gr.Markdown("### TF-IDF Vectorization")
                gr.Markdown("Convert text to numerical features using Term Frequency-Inverse Document Frequency")
                with gr.Row():
                    tfidf_col = gr.Textbox(label="Text Column", placeholder="Enter text column")
                    tfidf_features = gr.Slider(5, 100, value=20, step=5, label="Max Features")
                    tfidf_btn = gr.Button("Apply TF-IDF")
                tfidf_output = gr.Textbox(label="Status")
                tfidf_data = gr.Dataframe(label="Updated Data", height=300)
                tfidf_btn.click(apply_tfidf, inputs=[tfidf_col, tfidf_features], 
                              outputs=[tfidf_output, tfidf_data])
            
            with gr.Accordion("6️⃣ Feature Selection & Dimensionality Reduction"):
                gr.Markdown("### Correlation-Based Selection")
                gr.Markdown("Remove highly correlated features or select features correlated with target")
                with gr.Row():
                    corr_threshold = gr.Slider(0.5, 1.0, value=0.9, step=0.05, label="Correlation Threshold")
                    corr_target = gr.Textbox(label="Target Column (optional)", placeholder="Leave empty to remove correlated features")
                    corr_btn = gr.Button("Select Features")
                corr_output = gr.Textbox(label="Status")
                corr_data = gr.Dataframe(label="Updated Data", height=300)
                corr_btn.click(select_by_correlation, inputs=[corr_threshold, corr_target], 
                             outputs=[corr_output, corr_data])
                
                gr.Markdown("### PCA (Principal Component Analysis)")
                gr.Markdown("Reduce dimensions while keeping variance")
                with gr.Row():
                    pca_components = gr.Slider(2, 20, value=5, step=1, label="Number of Components")
                    pca_btn = gr.Button("Apply PCA")
                pca_output = gr.Textbox(label="Status")
                pca_data = gr.Dataframe(label="Updated Data", height=300)
                pca_btn.click(apply_pca_reduction, inputs=[pca_components], 
                            outputs=[pca_output, pca_data])
                
                gr.Markdown("### Feature Importance (Tree-Based)")
                gr.Markdown("Select top features based on Random Forest importance")
                with gr.Row():
                    imp_target = gr.Textbox(label="Target Column", placeholder="Enter target column")
                    imp_n_features = gr.Slider(5, 50, value=10, step=5, label="Number of Top Features")
                    imp_btn = gr.Button("Select by Importance")
                imp_output = gr.Textbox(label="Status")
                imp_data = gr.Dataframe(label="Updated Data", height=300)
                imp_btn.click(feature_importance_selection, inputs=[imp_target, imp_n_features], 
                            outputs=[imp_output, imp_data])
            
            with gr.Accordion("7️⃣ Interaction Features"):
                gr.Markdown("Capture relationships by multiplying features")
                with gr.Row():
                    inter_col1 = gr.Textbox(label="Column 1", placeholder="Enter numeric column")
                    inter_col2 = gr.Textbox(label="Column 2", placeholder="Enter numeric column")
                    inter_btn = gr.Button("Create Interaction")
                inter_output = gr.Textbox(label="Status")
                inter_data = gr.Dataframe(label="Updated Data", height=300)
                inter_btn.click(create_interaction_feature, inputs=[inter_col1, inter_col2], 
                              outputs=[inter_output, inter_data])
            
            with gr.Accordion("8️⃣ Aggregation Features (Grouped Data)"):
                gr.Markdown("Compute summary statistics per group")
                gr.Markdown("Example: Average_purchase_per_customer, Total_sales_per_city")
                with gr.Row():
                    agg_group_col = gr.Textbox(label="Group By Column", placeholder="Enter categorical column")
                    agg_agg_col = gr.Textbox(label="Aggregate Column", placeholder="Enter numeric column")
                    agg_funcs = gr.Textbox(label="Functions (comma-separated)", 
                                          placeholder="mean, sum, count, max, min, std",
                                          value="mean, sum, count")
                    agg_btn = gr.Button("Create Aggregation Features")
                agg_output = gr.Textbox(label="Status")
                agg_data = gr.Dataframe(label="Updated Data", height=300)
                agg_btn.click(create_aggregation_features, inputs=[agg_group_col, agg_agg_col, agg_funcs], 
                            outputs=[agg_output, agg_data])
        
        # ==================== TAB 5: MODEL BUILDING ====================
        with gr.Tab("🤖 Model Building"):
            
            gr.Markdown("# Comprehensive Model Building Pipeline")
            
            # SUPERVISED LEARNING
            gr.Markdown("## 📊 Supervised Learning")
            
            with gr.Accordion("1️⃣ Prepare Data - Train/Test Split", open=True):
                gr.Markdown("Split dataset into training and testing sets")
                with gr.Row():
                    target_col = gr.Textbox(label="Target Column", placeholder="Enter target column name")
                    test_size = gr.Slider(10, 50, value=20, step=5, label="Test Size (%)")
                    split_btn = gr.Button("✂️ Split Data", variant="primary")
                split_output = gr.Textbox(label="Split Status", lines=8)
                split_btn.click(split_data, inputs=[target_col, test_size], outputs=[split_output])
            
            with gr.Accordion("2️⃣ Select & Train Model"):
                gr.Markdown("Choose model type and train with optional cross-validation")
                with gr.Row():
                    model_type = gr.Dropdown(
                        ["Linear Regression", "Random Forest Regressor", "XGBoost Regressor",
                         "Logistic Regression", "Decision Tree", "Random Forest", "SVM", "KNN", 
                         "XGBoost Classifier", "Neural Network"],
                        label="Model Type",
                        value="Random Forest"
                    )
                    task_type = gr.Radio(["Regression", "Classification"], label="Task Type", value="Classification")
                
                with gr.Row():
                    use_cv = gr.Checkbox(label="Use Cross-Validation", value=False)
                    cv_folds = gr.Slider(3, 10, value=5, step=1, label="CV Folds")
                
                train_btn = gr.Button("🚀 Train Model", variant="primary", size="lg")
                train_output = gr.Textbox(label="Training Results", lines=20)
                train_btn.click(train_model_with_cv, inputs=[model_type, task_type, use_cv, cv_folds], 
                              outputs=[train_output])
            
            with gr.Accordion("3️⃣ Hyperparameter Tuning"):
                gr.Markdown("Optimize model hyperparameters using Grid Search or Random Search")
                with gr.Row():
                    tune_model = gr.Dropdown(
                        ["Random Forest", "Random Forest Regressor", "XGBoost Classifier", 
                         "XGBoost Regressor", "SVM"],
                        label="Model to Tune",
                        value="Random Forest"
                    )
                    tune_task = gr.Radio(["Regression", "Classification"], label="Task Type", value="Classification")
                
                with gr.Row():
                    search_type = gr.Dropdown(["Grid Search", "Random Search"], label="Search Type", value="Grid Search")
                    n_iter = gr.Slider(10, 100, value=20, step=10, label="Random Search Iterations")
                
                tune_btn = gr.Button("🔧 Tune Hyperparameters", variant="secondary")
                tune_output = gr.Textbox(label="Tuning Results", lines=15)
                tune_btn.click(hyperparameter_tuning, inputs=[tune_model, tune_task, search_type, n_iter], 
                             outputs=[tune_output])
            
            with gr.Accordion("4️⃣ Model Evaluation & Visualization"):
                gr.Markdown("Visualize model performance")
                
                gr.Markdown("**Confusion Matrix** (Classification)")
                cm_btn = gr.Button("Generate Confusion Matrix")
                cm_plot = gr.Plot(label="Confusion Matrix")
                cm_btn.click(plot_confusion_matrix, outputs=[cm_plot])
                
                gr.Markdown("**ROC Curve** (Binary Classification)")
                roc_btn = gr.Button("Generate ROC Curve")
                roc_plot = gr.Plot(label="ROC Curve")
                roc_btn.click(plot_roc_curve, outputs=[roc_plot])
                
                gr.Markdown("**Feature Importance** (Tree-based models)")
                fi_btn = gr.Button("Generate Feature Importance")
                fi_plot = gr.Plot(label="Feature Importance")
                fi_btn.click(plot_feature_importance, outputs=[fi_plot])
            
            # UNSUPERVISED LEARNING
            gr.Markdown("## 🔍 Unsupervised Learning")
            
            with gr.Accordion("5️⃣ Clustering"):
                gr.Markdown("Group similar data points into clusters")
                with gr.Row():
                    n_clusters = gr.Slider(2, 15, value=3, step=1, label="Number of Clusters")
                    cluster_algo = gr.Dropdown(["KMeans", "DBSCAN", "Agglomerative"], 
                                              label="Clustering Algorithm", value="KMeans")
                    cluster_btn = gr.Button("🔍 Run Clustering", variant="primary")
                
                cluster_output = gr.Textbox(label="Clustering Results", lines=12)
                cluster_data = gr.Dataframe(label="Data with Clusters", height=300)
                cluster_plot = gr.Plot(label="Cluster Visualization")
                
                cluster_btn.click(train_clustering, inputs=[n_clusters, cluster_algo], 
                                outputs=[cluster_output, cluster_data, cluster_plot])
            
            with gr.Accordion("6️⃣ Dimensionality Reduction"):
                gr.Markdown("Reduce feature dimensions while preserving information")
                with gr.Row():
                    dim_method = gr.Dropdown(["PCA", "t-SNE"], label="Method", value="PCA")
                    dim_components = gr.Slider(2, 10, value=2, step=1, label="Number of Components")
                    dim_btn = gr.Button("📉 Apply Dimensionality Reduction", variant="primary")
                
                dim_output = gr.Textbox(label="Results", lines=10)
                dim_data = gr.Dataframe(label="Reduced Data", height=300)
                dim_plot = gr.Plot(label="Visualization")
                
                dim_btn.click(apply_dimensionality_reduction, inputs=[dim_method, dim_components], 
                            outputs=[dim_output, dim_data, dim_plot])
            
            # MODEL DEPLOYMENT
            gr.Markdown("## 💾 Model Deployment")
            
            with gr.Accordion("7️⃣ Save Model"):
                gr.Markdown("Export trained model for later use")
                save_btn = gr.Button("💾 Save Model", variant="secondary", size="lg")
                save_output = gr.Textbox(label="Save Status")
                save_btn.click(save_model, outputs=[save_output])
            
            with gr.Accordion("8️⃣ Load Model"):
                gr.Markdown("Import previously saved model")
                with gr.Row():
                    load_model_file = gr.File(label="Model File (.pkl)")
                    load_scaler_file = gr.File(label="Scaler File (.pkl) - Optional")
                load_btn = gr.Button("📂 Load Model")
                load_output = gr.Textbox(label="Load Status")
                load_btn.click(load_model, inputs=[load_model_file, load_scaler_file], outputs=[load_output])
        
        # ==================== TAB 6: DOWNLOAD ====================
        with gr.Tab("📄 Download Results"):
            gr.Markdown("### Download Processed Dataset")
            gr.Markdown("Export your cleaned and processed data as CSV")
            download_btn = gr.Button("📥 Prepare Download", variant="primary", size="lg")
            download_file = gr.File(label="Download CSV")
            download_btn.click(download_data, outputs=[download_file])
    
    # Footer
    gr.Markdown("---")
    gr.Markdown("💡 **Tip:** Process your data step-by-step from left to right through the tabs")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
