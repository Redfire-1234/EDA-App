import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import warnings
from scipy import stats

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, RobustScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, classification_report, mean_squared_error, 
                            r2_score, confusion_matrix, roc_auc_score, roc_curve, 
                            auc, mean_absolute_error, silhouette_score, davies_bouldin_score)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.manifold import TSNE
import joblib

warnings.filterwarnings('ignore')
sns.set_style("darkgrid")

# Global state
state = {
    "df": None,
    "model": None,
    "scaler": None,
    "X_train": None,
    "X_test": None,
    "y_train": None,
    "y_test": None,
    "X_val": None,
    "y_val": None,
    "confusion_matrix": None,
    "y_pred": None
}

# ============================================================
# 1️⃣ DATA UPLOAD
# ============================================================
def upload_data(file):
    if file is None:
        return "⚠️ No file uploaded", None
    
    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file.name)
        elif file.name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file.name)
        elif file.name.endswith(".json"):
            df = pd.read_json(file.name)
        else:
            return "❌ Unsupported file type. Please upload CSV, Excel, or JSON", None
        
        state["df"] = df
        info_text = f"""✅ **{file.name}** uploaded successfully!

**Dataset Information:**
- **Shape:** {df.shape[0]} rows × {df.shape[1]} columns
- **Memory Usage:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
- **Columns:** {', '.join(df.columns.tolist())}
- **Numeric Columns:** {len(df.select_dtypes(include=np.number).columns)}
- **Categorical Columns:** {len(df.select_dtypes(include='object').columns)}
- **Missing Values:** {df.isnull().sum().sum()}
"""
        return info_text, df.head(20)
    except Exception as e:
        return f"❌ Error loading file: {str(e)}", None

# ============================================================
# 2️⃣ DATA CLEANING
# ============================================================

def handle_missing_data(action):
    if state["df"] is None:
        return "⚠️ Please upload data first", None
    
    df = state["df"].copy()
    before_nulls = df.isnull().sum().sum()
    
    if before_nulls == 0:
        return "✅ No missing values found in the dataset", df.head(20)
    
    try:
        if action == "Drop Rows":
            df = df.dropna()
        elif action == "Fill with Mean":
            numeric_cols = df.select_dtypes(include=np.number).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif action == "Fill with Median":
            numeric_cols = df.select_dtypes(include=np.number).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif action == "Fill with Mode":
            for col in df.columns:
                if not df[col].mode().empty:
                    df[col] = df[col].fillna(df[col].mode()[0])
        elif action == "Forward Fill":
            df = df.ffill()
        elif action == "Backward Fill":
            df = df.bfill()
        
        state["df"] = df
        after_nulls = df.isnull().sum().sum()
        return f"✅ Missing data handled! Before: {before_nulls} → After: {after_nulls} missing values", df.head(20)
    except Exception as e:
        return f"❌ Error: {str(e)}", None

def remove_duplicates():
    if state["df"] is None:
        return "⚠️ Please upload data first", None
    
    df = state["df"].copy()
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    state["df"] = df
    removed = before - after
    return f"✅ Removed {removed} duplicate rows (Before: {before} → After: {after} rows)", df.head(20)

def remove_outliers_iqr(column):
    if state["df"] is None or not column:
        return "⚠️ Please upload data and select a column", None
    
    df = state["df"].copy()
    
    if column not in df.columns:
        return f"❌ Column '{column}' not found in dataset", None
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        return f"❌ Column '{column}' is not numeric. IQR method only works with numeric data.", None
    
    try:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        before = len(df)
        df = df[~((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR)))]
        after = len(df)
        state["df"] = df
        return f"✅ IQR Method: Removed {before - after} outlier rows from '{column}'", df.head(20)
    except Exception as e:
        return f"❌ Error: {str(e)}", None

def remove_outliers_zscore(column, threshold):
    if state["df"] is None or not column:
        return "⚠️ Please upload data and select a column", None
    
    df = state["df"].copy()
    
    if column not in df.columns:
        return f"❌ Column '{column}' not found in dataset", None
    
    try:
        z_scores = np.abs(stats.zscore(df[column].dropna()))
        before = len(df)
        mask = z_scores < threshold
        valid_indices = df[column].dropna().index[mask]
        df = df.loc[valid_indices]
        after = len(df)
        state["df"] = df
        return f"✅ Z-Score Method: Removed {before - after} outliers (threshold={threshold})", df.head(20)
    except Exception as e:
        return f"❌ Error: {str(e)}", None

def cap_floor_outliers(column):
    if state["df"] is None or not column:
        return "⚠️ Please upload data and select a column", None
    
    df = state["df"].copy()
    
    if column not in df.columns:
        return f"❌ Column '{column}' not found in dataset", None
    
    try:
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
    except Exception as e:
        return f"❌ Error: {str(e)}", None

def standardize_text_data():
    if state["df"] is None:
        return "⚠️ Please upload data first", None
    
    df = state["df"].copy()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    
    if not cat_cols:
        return "⚠️ No text columns found in dataset", df.head(20)
    
    try:
        for col in cat_cols:
            df[col] = df[col].astype(str).str.strip().str.lower().str.replace(r'\s+', ' ', regex=True)
        
        state["df"] = df
        return f"✅ Standardized {len(cat_cols)} text columns (trimmed, lowercase, normalized spaces)", df.head(20)
    except Exception as e:
        return f"❌ Error: {str(e)}", None

def scale_data(method):
    if state["df"] is None:
        return "⚠️ Please upload data first", None
    
    df = state["df"].copy()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    if not numeric_cols:
        return "❌ No numeric columns to scale", df.head(20)
    
    try:
        if method == "Standardization (Z-score)":
            df[numeric_cols] = StandardScaler().fit_transform(df[numeric_cols])
        elif method == "Min-Max Normalization":
            df[numeric_cols] = MinMaxScaler().fit_transform(df[numeric_cols])
        elif method == "Robust Scaling":
            df[numeric_cols] = RobustScaler().fit_transform(df[numeric_cols])
        
        state["df"] = df
        return f"✅ {method} applied to {len(numeric_cols)} numeric columns", df.head(20)
    except Exception as e:
        return f"❌ Error: {str(e)}", None

def drop_columns(columns_text):
    if state["df"] is None:
        return "⚠️ Please upload data first", None
    
    if not columns_text:
        return "⚠️ Please specify columns to drop", None
    
    df = state["df"].copy()
    cols_to_drop = [c.strip() for c in columns_text.split(',')]
    existing_cols = [c for c in cols_to_drop if c in df.columns]
    
    if not existing_cols:
        return f"❌ None of the specified columns exist: {cols_to_drop}", df.head(20)
    
    try:
        df = df.drop(columns=existing_cols)
        state["df"] = df
        return f"✅ Dropped {len(existing_cols)} columns: {existing_cols}", df.head(20)
    except Exception as e:
        return f"❌ Error: {str(e)}", None

def validate_data():
    if state["df"] is None:
        return "⚠️ Please upload data first"
    
    df = state["df"]
    
    missing_info = df.isnull().sum()[df.isnull().sum() > 0].to_string() if df.isnull().sum().sum() > 0 else '✅ None'
    
    report = f"""## 📊 Data Validation Report

### Basic Information
- **Rows:** {df.shape[0]}
- **Columns:** {df.shape[1]}
- **Memory:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB

### Data Types
```
{df.dtypes.value_counts().to_string()}
```

### Missing Values
```
{missing_info}
```

### Duplicates
**{df.duplicated().sum()}** duplicate rows found

### Numeric Columns Summary
```
{df.describe().to_string()}
```

### Potential Issues
"""
    
    issues = []
    
    high_missing = df.isnull().sum() / len(df) > 0.5
    if high_missing.any():
        issues.append(f"⚠️ Columns with >50% missing: {high_missing[high_missing].index.tolist()}")
    
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_cols:
        issues.append(f"⚠️ Constant columns (consider dropping): {constant_cols}")
    
    cat_cols = df.select_dtypes(include='object').columns
    high_card = [col for col in cat_cols if df[col].nunique() > 50]
    if high_card:
        issues.append(f"⚠️ High cardinality categorical columns: {high_card}")
    
    if issues:
        report += "\n".join(issues)
    else:
        report += "✅ No major issues detected"
    
    return report

# ============================================================
# 3️⃣ EXPLORATORY DATA ANALYSIS
# ============================================================

def get_statistics():
    if state["df"] is None:
        return "⚠️ Please upload data first"
    
    df = state["df"]
    numeric_df = df.select_dtypes(include=np.number)
    
    desc_stats = df.describe(include='all').to_html()
    mode_stats = df.mode().iloc[0].to_frame(name='Mode').T.to_html() if len(df.mode()) > 0 else "No mode available"
    
    if len(numeric_df.columns) > 0:
        skew_kurt = pd.DataFrame({
            'Skewness': numeric_df.skew(),
            'Kurtosis': numeric_df.kurtosis()
        }).to_html()
        pearson_corr = numeric_df.corr(method='pearson').to_html()
        spearman_corr = numeric_df.corr(method='spearman').to_html()
    else:
        skew_kurt = pearson_corr = spearman_corr = "❌ No numeric columns available"
    
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
    if state["df"] is None or not column or column not in state["df"].columns:
        return None
    
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(state["df"][column], kde=True, ax=ax, color='steelblue')
        ax.set_title(f"Histogram of {column}", fontsize=14, fontweight='bold')
        ax.set_xlabel(column, fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        plt.tight_layout()
        return fig
    except Exception as e:
        return None

def plot_density(column):
    if state["df"] is None or not column or column not in state["df"].columns:
        return None
    
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.kdeplot(state["df"][column], fill=True, ax=ax, color='coral')
        ax.set_title(f"Density Plot of {column}", fontsize=14, fontweight='bold')
        ax.set_xlabel(column, fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        plt.tight_layout()
        return fig
    except Exception as e:
        return None

def plot_boxplot(column):
    if state["df"] is None or not column or column not in state["df"].columns:
        return None
    
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x=state["df"][column], ax=ax, color='lightgreen')
        ax.set_title(f"Boxplot of {column}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    except Exception as e:
        return None

def plot_bar_chart(column):
    if state["df"] is None or not column or column not in state["df"].columns:
        return None
    
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        value_counts = state["df"][column].value_counts()
        value_counts.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
        ax.set_title(f"Bar Chart of {column}", fontsize=14, fontweight='bold')
        ax.set_xlabel(column, fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        return fig
    except Exception as e:
        return None

def plot_correlation_heatmap():
    if state["df"] is None:
        return None
    
    numeric_df = state["df"].select_dtypes(include=np.number)
    if numeric_df.shape[1] < 2:
        return None
    
    try:
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax, fmt=".2f", 
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        ax.set_title("Correlation Heatmap (Pearson)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    except Exception as e:
        return None

def plot_scatter(x_col, y_col):
    if state["df"] is None or not x_col or not y_col:
        return None
    
    if x_col not in state["df"].columns or y_col not in state["df"].columns:
        return None
    
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=state["df"][x_col], y=state["df"][y_col], ax=ax, alpha=0.6)
        ax.set_title(f"Scatter Plot: {x_col} vs {y_col}", fontsize=14, fontweight='bold')
        ax.set_xlabel(x_col, fontsize=12)
        ax.set_ylabel(y_col, fontsize=12)
        plt.tight_layout()
        return fig
    except Exception as e:
        return None

# ============================================================
# 4️⃣ FEATURE ENGINEERING
# ============================================================

def encode_categorical(column, method):
    if state["df"] is None or not column or column not in state["df"].columns:
        return "⚠️ Please upload data and select a valid column", None
    
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
        return f"❌ Error: {str(e)}", None

def apply_log_transform(column):
    if state["df"] is None or not column or column not in state["df"].columns:
        return "⚠️ Please select a valid column", None
    
    df = state["df"].copy()
    
    try:
        df[f"{column}_log"] = np.log1p(df[column].abs())
        state["df"] = df
        return f"✅ Log transform applied to '{column}' (created '{column}_log')", df.head(20)
    except Exception as e:
        return f"❌ Error: {str(e)}", None

def apply_sqrt_transform(column):
    if state["df"] is None or not column or column not in state["df"].columns:
        return "⚠️ Please select a valid column", None
    
    df = state["df"].copy()
    
    try:
        df[f"{column}_sqrt"] = np.sqrt(df[column].abs())
        state["df"] = df
        return f"✅ Square root transform applied to '{column}' (created '{column}_sqrt')", df.head(20)
    except Exception as e:
        return f"❌ Error: {str(e)}", None

def create_polynomial_features(columns_text, degree):
    if state["df"] is None or not columns_text:
        return "⚠️ Please provide columns", None
    
    df = state["df"].copy()
    columns = [c.strip() for c in columns_text.split(',')]
    
    # Validate columns exist
    invalid_cols = [c for c in columns if c not in df.columns]
    if invalid_cols:
        return f"❌ Columns not found: {invalid_cols}", None
    
    try:
        from sklearn.preprocessing import PolynomialFeatures
        
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        poly_features = poly.fit_transform(df[columns])
        
        feature_names = poly.get_feature_names_out(columns)
        poly_df = pd.DataFrame(poly_features, columns=feature_names, index=df.index)
        
        # Add only new features
        new_features = [col for col in poly_df.columns if col not in columns]
        for col in new_features:
            df[col] = poly_df[col]
        
        state["df"] = df
        return f"✅ Created {len(new_features)} polynomial features (degree={degree})", df.head(20)
    except Exception as e:
        return f"❌ Error: {str(e)}", None

# ============================================================
# 5️⃣ MODEL BUILDING
# ============================================================

def split_data(target_col, test_size, val_size):
    if state["df"] is None or not target_col:
        return "⚠️ Please upload data and select target column"
    
    df = state["df"]
    
    if target_col not in df.columns:
        return f"❌ Target column '{target_col}' not found in dataset"
    
    try:
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols:
            return "❌ No numeric features found for training. Please ensure you have numeric columns."
        
        X = X[numeric_cols]
        
        # Handle missing values in target
        if y.isnull().any():
            return f"❌ Target column '{target_col}' contains missing values. Please clean the data first."
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size/100, random_state=42
        )
        
        if val_size > 0:
            val_size_adjusted = val_size / (100 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size_adjusted, random_state=42
            )
            state["X_val"] = X_val
            state["y_val"] = y_val
            val_info = f"- **Validation set:** {len(X_val)} samples ({len(X_val)/len(df)*100:.1f}%)\n"
        else:
            X_train, y_train = X_temp, y_temp
            state["X_val"] = None
            state["y_val"] = None
            val_info = ""
        
        state["X_train"] = X_train
        state["X_test"] = X_test
        state["y_train"] = y_train
        state["y_test"] = y_test
        
        result = f"""✅ **Data split successfully!**

### Dataset Distribution
- **Training set:** {len(X_train)} samples ({len(X_train)/len(df)*100:.1f}%)
{val_info}- **Test set:** {len(X_test)} samples ({len(X_test)/len(df)*100:.1f}%)
- **Features:** {len(numeric_cols)}
- **Target:** {target_col}

### Feature Names
{', '.join(numeric_cols[:10])}{'...' if len(numeric_cols) > 10 else ''}
"""
        return result
    except Exception as e:
        return f"❌ Error: {str(e)}"

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
    
    try:
        # Select model
        if task_type == "Regression":
            if model_type == "Linear Regression":
                model = LinearRegression()
            elif model_type == "Random Forest Regressor":
                model = RandomForestRegressor(random_state=42, n_estimators=100)
            else:
                return f"⚠️ Model '{model_type}' not supported for regression"
        else:  # Classification
            if model_type == "Logistic Regression":
                model = LogisticRegression(max_iter=1000, random_state=42)
            elif model_type == "Decision Tree":
                model = DecisionTreeClassifier(random_state=42)
            elif model_type == "Random Forest":
                model = RandomForestClassifier(random_state=42, n_estimators=100)
            elif model_type == "SVM":
                model = SVC(probability=True, random_state=42)
            elif model_type == "KNN":
                model = KNeighborsClassifier()
            else:
                return f"⚠️ Model '{model_type}' not supported for classification"
        
        # Cross-validation
        cv_results = ""
        if use_cv:
            scoring = 'r2' if task_type == "Regression" else 'accuracy'
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds, scoring=scoring)
            metric_name = "R²" if task_type == "Regression" else "Accuracy"
            cv_results = f"""
### Cross-Validation Results ({cv_folds}-Fold)
- **Mean {metric_name}:** {cv_scores.mean():.4f} (±{cv_scores.std():.4f})
- **Individual Scores:** {[f'{s:.4f}' for s in cv_scores]}
"""
        
        # Train model
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        if task_type == "Regression":
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            state["model"] = model
            state["scaler"] = scaler
            
            return f"""✅ **Model trained successfully!**

### Model Information
- **Model Type:** {model_type}
- **Task:** {task_type}
{cv_results}

### Test Set Performance
- **MSE:** {mse:.4f}
- **RMSE:** {rmse:.4f}
- **MAE:** {mae:.4f}
- **R² Score:** {r2:.4f}

### Interpretation
- The model explains **{r2:.2%}** of the variance in the target variable
- Average prediction error: **{mae:.4f}**
- {'Good performance ✅' if r2 > 0.7 else 'Consider improving the model ⚠️'}
"""
        else:  # Classification
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            
            roc_auc = ""
            if len(np.unique(y_test)) == 2:
                try:
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                    auc_score = roc_auc_score(y_test, y_pred_proba)
                    roc_auc = f"\n- **ROC-AUC:** {auc_score:.4f}"
                except:
                    pass
            
            state["model"] = model
            state["scaler"] = scaler
            state["confusion_matrix"] = cm
            state["y_test"] = y_test
            state["y_pred"] = y_pred
            
            return f"""✅ **Model trained successfully!**

### Model Information
- **Model Type:** {model_type}
- **Task:** {task_type}
{cv_results}

### Test Set Performance
- **Accuracy:** {acc:.4f} ({acc:.2%}){roc_auc}

### Classification Report
```
{report}
```

### Confusion Matrix
```
{cm}
```

### Interpretation
{'Excellent performance ✅' if acc > 0.9 else 'Good performance ✅' if acc > 0.8 else 'Consider improving the model ⚠️'}
"""
    except Exception as e:
        return f"❌ Error training model: {str(e)}"

def train_clustering(n_clusters, algorithm):
    if state["df"] is None:
        return "⚠️ Please upload data first", None, None
    
    df = state["df"].copy()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    if not numeric_cols:
        return "❌ No numeric columns found for clustering", None, None
    
    X = df[numeric_cols]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    try:
        if algorithm == "KMeans":
            model = KMeans(n_clusters=n_clusters, random_state=42)
        elif algorithm == "DBSCAN":
            model = DBSCAN(eps=0.5, min_samples=5)
        elif algorithm == "Agglomerative":
            model = AgglomerativeClustering(n_clusters=n_clusters)
        
        clusters = model.fit_predict(X_scaled)
        
        df["Cluster"] = clusters
        state["df"] = df
        state["model"] = model
        
        # Calculate metrics
        silhouette = silhouette_score(X_scaled, clusters)
        davies_bouldin = davies_bouldin_score(X_scaled, clusters)
        
        # Cluster distribution
        cluster_counts = pd.Series(clusters).value_counts().sort_index()
        
        result = f"""✅ **{algorithm} clustering completed!**

### Cluster Information
- **Number of Clusters:** {n_clusters if algorithm != "DBSCAN" else len(set(clusters)) - (1 if -1 in clusters else 0)}
- **Silhouette Score:** {silhouette:.4f} (Higher is better, range: -1 to 1)
- **Davies-Bouldin Index:** {davies_bouldin:.4f} (Lower is better)

### Cluster Distribution
"""
        for i, count in cluster_counts.items():
            result += f"\n- Cluster {i}: {count} samples ({count/len(clusters)*100:.1f}%)"
        
        # Create visualization
        if X.shape[1] >= 2:
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis', alpha=0.6)
            ax.set_title(f'{algorithm} Clustering Results', fontsize=14, fontweight='bold')
            ax.set_xlabel('Feature 1 (scaled)', fontsize=12)
            ax.set_ylabel('Feature 2 (scaled)', fontsize=12)
            plt.colorbar(scatter, ax=ax, label='Cluster')
            plt.tight_layout()
        else:
            fig = None
        
        return result, df.head(20), fig
    except Exception as e:
        return f"❌ Error: {str(e)}", None, None

def plot_confusion_matrix():
    if "confusion_matrix" not in state or state["confusion_matrix"] is None:
        return None
    
    try:
        cm = state["confusion_matrix"]
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar_kws={'label': 'Count'})
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        plt.tight_layout()
        return fig
    except Exception as e:
        return None

def plot_roc_curve():
    if state["model"] is None or state["X_test"] is None:
        return None
    
    try:
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
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
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
        ax.barh(range(len(indices)), importances[indices], color='steelblue', edgecolor='black')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title('Top 20 Feature Importances', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        return fig
    except:
        return None

def save_model():
    if state["model"] is None:
        return "⚠️ Train a model first"
    
    try:
        joblib.dump(state["model"], "trained_model.pkl")
        if "scaler" in state and state["scaler"] is not None:
            joblib.dump(state["scaler"], "scaler.pkl")
        return "✅ Model saved as 'trained_model.pkl'\n✅ Scaler saved as 'scaler.pkl'"
    except Exception as e:
        return f"❌ Error: {str(e)}"

def download_data():
    if state["df"] is None:
        return None
    return state["df"]

# ============================================================
# GRADIO INTERFACE
# ============================================================

with gr.Blocks(title="End-to-End ML Pipeline", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Complete End-to-End ML Pipeline")
    gr.Markdown("**Comprehensive data processing, cleaning, analysis, and machine learning platform**")
    
    with gr.Tabs():
        # TAB 1: DATA UPLOAD
        with gr.Tab("📂 Data Upload"):
            gr.Markdown("### Upload your dataset (CSV, Excel, or JSON)")
            file_input = gr.File(label="Upload Dataset")
            upload_btn = gr.Button("📤 Upload Data", variant="primary", size="lg")
            upload_output = gr.Textbox(label="Upload Status", lines=10)
            data_preview = gr.Dataframe(label="Data Preview", height=400)
            
            upload_btn.click(upload_data, inputs=[file_input], outputs=[upload_output, data_preview])
        
        # TAB 2: DATA CLEANING
        with gr.Tab("🧹 Data Cleaning"):
            with gr.Accordion("1️⃣ Handle Missing Data", open=True):
                gr.Markdown("Fill or remove missing values using various strategies")
                missing_action = gr.Dropdown(
                    ["Drop Rows", "Fill with Mean", "Fill with Median", "Fill with Mode", 
                     "Forward Fill", "Backward Fill"],
                    label="Missing Data Strategy",
                    value="Fill with Mean"
                )
                missing_btn = gr.Button("Apply Strategy", variant="primary")
                missing_output = gr.Textbox(label="Status")
                missing_data = gr.Dataframe(label="Updated Data", height=300)
                missing_btn.click(handle_missing_data, inputs=[missing_action], 
                                outputs=[missing_output, missing_data])
            
            with gr.Accordion("2️⃣ Remove Duplicates"):
                gr.Markdown("Remove duplicate rows from the dataset")
                dup_btn = gr.Button("🗑️ Remove Duplicates", variant="primary")
                dup_output = gr.Textbox(label="Status")
                dup_data = gr.Dataframe(label="Updated Data", height=300)
                dup_btn.click(remove_duplicates, outputs=[dup_output, dup_data])
            
            with gr.Accordion("3️⃣ Handle Outliers"):
                gr.Markdown("**IQR Method (Interquartile Range)**")
                outlier_col_iqr = gr.Textbox(label="Column Name", placeholder="Enter numeric column")
                outlier_btn_iqr = gr.Button("Remove Outliers (IQR)")
                outlier_output_iqr = gr.Textbox(label="Status")
                outlier_data_iqr = gr.Dataframe(label="Updated Data", height=300)
                outlier_btn_iqr.click(remove_outliers_iqr, inputs=[outlier_col_iqr], 
                                     outputs=[outlier_output_iqr, outlier_data_iqr])
                
                gr.Markdown("**Z-Score Method**")
                with gr.Row():
                    outlier_col_z = gr.Textbox(label="Column Name")
                    z_threshold = gr.Slider(1, 5, value=3, step=0.5, label="Threshold")
                outlier_btn_z = gr.Button("Remove Outliers (Z-Score)")
                outlier_output_z = gr.Textbox(label="Status")
                outlier_data_z = gr.Dataframe(label="Updated Data", height=300)
                outlier_btn_z.click(remove_outliers_zscore, inputs=[outlier_col_z, z_threshold], 
                                   outputs=[outlier_output_z, outlier_data_z])
                
                gr.Markdown("**Cap/Floor Method (Winsorization)**")
                outlier_col_cap = gr.Textbox(label="Column Name")
                outlier_btn_cap = gr.Button("Cap/Floor Outliers")
                outlier_output_cap = gr.Textbox(label="Status")
                outlier_data_cap = gr.Dataframe(label="Updated Data", height=300)
                outlier_btn_cap.click(cap_floor_outliers, inputs=[outlier_col_cap], 
                                     outputs=[outlier_output_cap, outlier_data_cap])
            
            with gr.Accordion("4️⃣ Standardize Text"):
                gr.Markdown("Auto-standardize all text columns (trim, lowercase, normalize spaces)")
                std_btn = gr.Button("🔧 Standardize Text Data", variant="primary")
                std_output = gr.Textbox(label="Status")
                std_data = gr.Dataframe(label="Updated Data", height=300)
                std_btn.click(standardize_text_data, outputs=[std_output, std_data])
            
            with gr.Accordion("5️⃣ Scale Data"):
                gr.Markdown("Scale numeric features for model compatibility")
                scale_method = gr.Dropdown(
                    ["Standardization (Z-score)", "Min-Max Normalization", "Robust Scaling"], 
                    label="Scaling Method",
                    value="Standardization (Z-score)"
                )
                scale_btn = gr.Button("Apply Scaling", variant="primary")
                scale_output = gr.Textbox(label="Status")
                scale_data_out = gr.Dataframe(label="Updated Data", height=300)
                scale_btn.click(scale_data, inputs=[scale_method], outputs=[scale_output, scale_data_out])
            
            with gr.Accordion("6️⃣ Drop Columns"):
                gr.Markdown("Remove irrelevant columns (comma-separated: col1, col2)")
                drop_cols_text = gr.Textbox(label="Columns to Drop", placeholder="col1, col2, col3")
                drop_btn = gr.Button("🗑️ Drop Columns", variant="stop")
                drop_output = gr.Textbox(label="Status")
                drop_data = gr.Dataframe(label="Updated Data", height=300)
                drop_btn.click(drop_columns, inputs=[drop_cols_text], outputs=[drop_output, drop_data])
            
            with gr.Accordion("7️⃣ Validate Data"):
                gr.Markdown("Check data integrity and identify issues")
                validate_btn = gr.Button("📊 Validate Data", variant="primary", size="lg")
                validate_output = gr.Markdown(label="Validation Report")
                validate_btn.click(validate_data, outputs=[validate_output])
        
        # TAB 3: EDA
        with gr.Tab("📈 Exploratory Data Analysis"):
            with gr.Accordion("Statistics", open=True):
                gr.Markdown("Generate comprehensive statistical summary")
                stats_btn = gr.Button("📊 Generate Statistics", variant="primary", size="lg")
                stats_output = gr.Markdown(label="Statistics")
                stats_btn.click(get_statistics, outputs=[stats_output])
            
            with gr.Accordion("Univariate Analysis"):
                gr.Markdown("**Histogram**")
                hist_col = gr.Textbox(label="Column Name")
                hist_btn = gr.Button("Plot Histogram")
                hist_plot = gr.Plot(label="Histogram")
                hist_btn.click(plot_histogram, inputs=[hist_col], outputs=[hist_plot])
                
                gr.Markdown("**Density Plot**")
                density_col = gr.Textbox(label="Column Name")
                density_btn = gr.Button("Plot Density")
                density_plot = gr.Plot(label="Density Plot")
                density_btn.click(plot_density, inputs=[density_col], outputs=[density_plot])
                
                gr.Markdown("**Boxplot**")
                box_col = gr.Textbox(label="Column Name")
                box_btn = gr.Button("Plot Boxplot")
                box_plot = gr.Plot(label="Boxplot")
                box_btn.click(plot_boxplot, inputs=[box_col], outputs=[box_plot])
                
                gr.Markdown("**Bar Chart**")
                bar_col = gr.Textbox(label="Column Name")
                bar_btn = gr.Button("Plot Bar Chart")
                bar_plot = gr.Plot(label="Bar Chart")
                bar_btn.click(plot_bar_chart, inputs=[bar_col], outputs=[bar_plot])
            
            with gr.Accordion("Bivariate Analysis"):
                gr.Markdown("**Scatter Plot**")
                with gr.Row():
                    scatter_x = gr.Textbox(label="X Column")
                    scatter_y = gr.Textbox(label="Y Column")
                scatter_btn = gr.Button("Plot Scatter")
                scatter_plot = gr.Plot(label="Scatter Plot")
                scatter_btn.click(plot_scatter, inputs=[scatter_x, scatter_y], outputs=[scatter_plot])
            
            with gr.Accordion("Multivariate Analysis"):
                gr.Markdown("**Correlation Heatmap**")
                heatmap_btn = gr.Button("Generate Heatmap", variant="primary")
                heatmap_plot = gr.Plot(label="Correlation Heatmap")
                heatmap_btn.click(plot_correlation_heatmap, outputs=[heatmap_plot])
        
        # TAB 4: FEATURE ENGINEERING
        with gr.Tab("⚙️ Feature Engineering"):
            with gr.Accordion("Encoding", open=True):
                gr.Markdown("### Categorical Encoding")
                with gr.Row():
                    encode_col = gr.Textbox(label="Column Name")
                    encode_method = gr.Dropdown(["One-Hot", "Label", "Frequency"], 
                                               label="Method", value="One-Hot")
                encode_btn = gr.Button("Apply Encoding", variant="primary")
                encode_output = gr.Textbox(label="Status")
                encode_data = gr.Dataframe(label="Updated Data", height=300)
                encode_btn.click(encode_categorical, inputs=[encode_col, encode_method], 
                               outputs=[encode_output, encode_data])
            
            with gr.Accordion("Transformations"):
                gr.Markdown("**Log Transformation**")
                log_col = gr.Textbox(label="Column Name")
                log_btn = gr.Button("Apply Log Transform")
                log_output = gr.Textbox(label="Status")
                log_data = gr.Dataframe(label="Updated Data", height=300)
                log_btn.click(apply_log_transform, inputs=[log_col], outputs=[log_output, log_data])
                
                gr.Markdown("**Square Root Transformation**")
                sqrt_col = gr.Textbox(label="Column Name")
                sqrt_btn = gr.Button("Apply Sqrt Transform")
                sqrt_output = gr.Textbox(label="Status")
                sqrt_data = gr.Dataframe(label="Updated Data", height=300)
                sqrt_btn.click(apply_sqrt_transform, inputs=[sqrt_col], outputs=[sqrt_output, sqrt_data])
            
            with gr.Accordion("Polynomial Features"):
                gr.Markdown("Create polynomial and interaction terms")
                with gr.Row():
                    poly_cols = gr.Textbox(label="Columns (comma-separated)", placeholder="col1, col2")
                    poly_degree = gr.Slider(2, 4, value=2, step=1, label="Degree")
                poly_btn = gr.Button("Create Polynomial Features")
                poly_output = gr.Textbox(label="Status")
                poly_data = gr.Dataframe(label="Updated Data", height=300)
                poly_btn.click(create_polynomial_features, inputs=[poly_cols, poly_degree], 
                             outputs=[poly_output, poly_data])
        
        # TAB 5: MODEL BUILDING
        with gr.Tab("🤖 Model Building"):
            gr.Markdown("## Supervised Learning Pipeline")
            
            with gr.Accordion("1️⃣ Split Data", open=True):
                gr.Markdown("Split dataset into train/validation/test sets")
                target_col = gr.Textbox(label="Target Column", placeholder="Enter target column name")
                with gr.Row():
                    test_size = gr.Slider(10, 50, value=20, step=5, label="Test Size (%)")
                    val_size = gr.Slider(0, 30, value=0, step=5, label="Validation Size (%)")
                split_btn = gr.Button("✂️ Split Data", variant="primary", size="lg")
                split_output = gr.Markdown(label="Split Status")
                split_btn.click(split_data, inputs=[target_col, test_size, val_size], outputs=[split_output])
            
            with gr.Accordion("2️⃣ Train Model"):
                gr.Markdown("Select and train a machine learning model")
                with gr.Row():
                    model_type = gr.Dropdown(
                        ["Linear Regression", "Random Forest Regressor",
                         "Logistic Regression", "Decision Tree", "Random Forest", "SVM", "KNN"],
                        label="Model Type",
                        value="Random Forest"
                    )
                    task_type = gr.Radio(["Regression", "Classification"], 
                                        label="Task Type", value="Classification")
                
                with gr.Row():
                    use_cv = gr.Checkbox(label="Use Cross-Validation", value=False)
                    cv_folds = gr.Slider(3, 10, value=5, step=1, label="CV Folds")
                
                train_btn = gr.Button("🚀 Train Model", variant="primary", size="lg")
                train_output = gr.Markdown(label="Training Results")
                train_btn.click(train_model_with_cv, inputs=[model_type, task_type, use_cv, cv_folds], 
                              outputs=[train_output])
            
            with gr.Accordion("3️⃣ Model Evaluation"):
                gr.Markdown("**Confusion Matrix** (Classification)")
                cm_btn = gr.Button("Generate Confusion Matrix")
                cm_plot = gr.Plot(label="Confusion Matrix")
                cm_btn.click(plot_confusion_matrix, outputs=[cm_plot])
                
                gr.Markdown("**ROC Curve** (Binary Classification)")
                roc_btn = gr.Button("Generate ROC Curve")
                roc_plot = gr.Plot(label="ROC Curve")
                roc_btn.click(plot_roc_curve, outputs=[roc_plot])
                
                gr.Markdown("**Feature Importance** (Tree-based)")
                fi_btn = gr.Button("Generate Feature Importance")
                fi_plot = gr.Plot(label="Feature Importance")
                fi_btn.click(plot_feature_importance, outputs=[fi_plot])
            
            with gr.Accordion("4️⃣ Clustering (Unsupervised)"):
                gr.Markdown("Group similar data points")
                with gr.Row():
                    n_clusters = gr.Slider(2, 15, value=3, step=1, label="Number of Clusters")
                    cluster_algo = gr.Dropdown(["KMeans", "DBSCAN", "Agglomerative"], 
                                              label="Algorithm", value="KMeans")
                cluster_btn = gr.Button("🔍 Run Clustering", variant="primary")
                cluster_output = gr.Markdown(label="Results")
                cluster_data = gr.Dataframe(label="Data with Clusters", height=300)
                cluster_plot = gr.Plot(label="Cluster Visualization")
                cluster_btn.click(train_clustering, inputs=[n_clusters, cluster_algo], 
                                outputs=[cluster_output, cluster_data, cluster_plot])
            
            with gr.Accordion("5️⃣ Save Model"):
                gr.Markdown("Export trained model")
                save_btn = gr.Button("💾 Save Model", variant="secondary", size="lg")
                save_output = gr.Textbox(label="Status")
                save_btn.click(save_model, outputs=[save_output])
        
        # TAB 6: DOWNLOAD
        with gr.Tab("📄 Download"):
            gr.Markdown("### Download Processed Dataset")
            download_btn = gr.Button("📥 Prepare Download", variant="primary", size="lg")
            download_file = gr.File(label="Download CSV")
            download_btn.click(download_data, outputs=[download_file])
    
    gr.Markdown("---")
    gr.Markdown("💡 **Tip:** Process data step-by-step through the tabs • 🚀 **Complete ML Pipeline**")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
