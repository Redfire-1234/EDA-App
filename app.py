import gradio as gr
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Global variables to store the dataframe and history
current_df = None
df_history = []

def load_file(file):
    global current_df, df_history
    if file is None:
        return "No file uploaded", None, gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[])
    filename = file.name.lower()
    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(file.name)
        elif filename.endswith(".xlsx"):
            df = pd.read_excel(file.name)
        elif filename.endswith(".json"):
            df = pd.read_json(file.name)
        else:
            return "Unsupported file format", None, gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[])
        
        current_df = df.copy()
        df_history = [df.copy()]  # Initialize history with original data
        info = f"Rows: {df.shape[0]} | Columns: {df.shape[1]}"
        
        # Get column lists for different purposes
        all_cols = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        text_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        return (info, df.head(20), 
          gr.update(choices=all_cols),  # missing_cols
          gr.update(choices=all_cols),  # dup_cols
          gr.update(choices=numeric_cols),  # outlier_cols
          gr.update(choices=all_cols),  # dtype_col
          gr.update(choices=text_cols),  # text_cols
          gr.update(choices=numeric_cols),  # scale_cols
          gr.update(choices=all_cols),  # uni_col
          gr.update(choices=all_cols),  # bi_col1
          gr.update(choices=all_cols),  # bi_col2
          gr.update(choices=numeric_cols),  # outlier_col_eda
          gr.update(choices=numeric_cols),  # dist_col
          gr.update(choices=all_cols))  # cat_col
    except Exception as e:
        return f"Error: {str(e)}", None, gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[])

def load_file_extended(file):
    result = load_file(file)
    # Add extra dropdown updates for Feature Engineering tab
    if result[0] != "No file uploaded" and "Error" not in result[0]:
        all_cols = current_df.columns.tolist()
        numeric_cols = current_df.select_dtypes(include=[np.number]).columns.tolist()
        return result + (gr.update(choices=all_cols), gr.update(choices=all_cols), 
                        gr.update(choices=numeric_cols), gr.update(choices=all_cols),
                        gr.update(choices=numeric_cols), gr.update(choices=numeric_cols))
    else:
        return result + (gr.update(choices=[]), gr.update(choices=[]), 
                        gr.update(choices=[]), gr.update(choices=[]),
                        gr.update(choices=[]), gr.update(choices=[]))
        
def save_to_history():
    global current_df, df_history
    if current_df is not None:
        df_history.append(current_df.copy())
        # Keep only last 10 states to avoid memory issues
        if len(df_history) > 10:
            df_history.pop(0)

def undo_last_action():
    global current_df, df_history
    if len(df_history) > 1:
        df_history.pop()  # Remove current state
        current_df = df_history[-1].copy()  # Restore previous state
        return f"✓ Undo successful. Restored to previous state (History: {len(df_history)} states)", current_df.head(20)
    else:
        return "Cannot undo. No previous state available.", current_df.head(20) if current_df is not None else None

def get_cleaning_summary():
    global current_df
    if current_df is None:
        return "No dataset loaded"
    
    summary = []
    summary.append(f"**Dataset Shape:** {current_df.shape[0]} rows × {current_df.shape[1]} columns\n")
    
    # Missing values
    missing = current_df.isnull().sum()
    if missing.sum() > 0:
        summary.append("**Missing Values:**")
        for col, count in missing[missing > 0].items():
            pct = (count / len(current_df)) * 100
            summary.append(f"  • {col}: {count} ({pct:.1f}%)")
        summary.append("")
    else:
        summary.append("**Missing Values:** None\n")
    
    # Duplicates
    dup_count = current_df.duplicated().sum()
    summary.append(f"**Duplicate Rows:** {dup_count}\n")
    
    # Data types
    summary.append("**Data Types:**")
    dtype_counts = current_df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        summary.append(f"  • {dtype}: {count} columns")
    
    return "\n".join(summary)

def handle_missing_values(strategy, columns, fill_value):
    global current_df
    if current_df is None:
        return "No dataset loaded", None
    
    if not columns:
        return "Please select at least one column", None
    
    save_to_history()
    df = current_df.copy()
    
    try:
        for col in columns:
            if df[col].isnull().sum() == 0:
                continue
                
            if strategy == "Drop rows":
                df = df.dropna(subset=[col])
            elif strategy == "Fill with mean":
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col].fillna(df[col].mean(), inplace=True)
            elif strategy == "Fill with median":
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col].fillna(df[col].median(), inplace=True)
            elif strategy == "Fill with mode":
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col].fillna(mode_val[0], inplace=True)
            elif strategy == "Fill with custom value":
                df[col].fillna(fill_value, inplace=True)
            elif strategy == "Forward fill":
                df[col].fillna(method='ffill', inplace=True)
            elif strategy == "Backward fill":
                df[col].fillna(method='bfill', inplace=True)
        
        current_df = df
        return f"✓ Missing values handled using '{strategy}' for {len(columns)} column(s)", df.head(20)
    except Exception as e:
        df_history.pop()  # Remove failed state from history
        return f"Error: {str(e)}", None

def remove_duplicates(subset_cols):
    global current_df
    if current_df is None:
        return "No dataset loaded", None
    
    save_to_history()
    df = current_df.copy()
    initial_count = len(df)
    
    try:
        if subset_cols:
            df = df.drop_duplicates(subset=subset_cols)
        else:
            df = df.drop_duplicates()
        
        removed = initial_count - len(df)
        current_df = df
        return f"✓ Removed {removed} duplicate rows", df.head(20)
    except Exception as e:
        df_history.pop()
        return f"Error: {str(e)}", None

def handle_outliers(method, columns, threshold):
    global current_df
    if current_df is None:
        return "No dataset loaded", None
    
    if not columns:
        return "Please select at least one column", None
    
    save_to_history()
    df = current_df.copy()
    
    try:
        removed_count = 0
        for col in columns:
            if method == "IQR Method":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                initial = len(df)
                df = df[(df[col] >= lower) & (df[col] <= upper)]
                removed_count += initial - len(df)
            elif method == "Z-Score Method":
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                initial = len(df)
                mask = np.abs(stats.zscore(df[col])) < threshold
                df = df[mask]
                removed_count += initial - len(df)
            elif method == "Cap outliers":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                df[col] = df[col].clip(lower, upper)
        
        current_df = df
        msg = f"✓ Outliers handled using '{method}' for {len(columns)} column(s)"
        if method != "Cap outliers":
            msg += f" ({removed_count} rows removed)"
        return msg, df.head(20)
    except Exception as e:
        df_history.pop()
        return f"Error: {str(e)}", None

def correct_data_types(column, new_type):
    global current_df
    if current_df is None:
        return "No dataset loaded", None
    
    if not column:
        return "Please select a column", None
    
    save_to_history()
    df = current_df.copy()
    
    try:
        if new_type == "int":
            df[column] = pd.to_numeric(df[column], errors='coerce').astype('Int64')
        elif new_type == "float":
            df[column] = pd.to_numeric(df[column], errors='coerce')
        elif new_type == "string":
            df[column] = df[column].astype(str)
        elif new_type == "datetime":
            df[column] = pd.to_datetime(df[column], errors='coerce')
        elif new_type == "category":
            df[column] = df[column].astype('category')
        elif new_type == "bool":
            df[column] = df[column].astype(bool)
        
        current_df = df
        return f"✓ Column '{column}' converted to {new_type}", df.head(20)
    except Exception as e:
        df_history.pop()
        return f"Error: {str(e)}", None

def standardize_text(columns, operation):
    global current_df
    if current_df is None:
        return "No dataset loaded", None
    
    if not columns:
        return "Please select at least one column", None
    
    save_to_history()
    df = current_df.copy()
    
    try:
        for col in columns:
            if operation == "Lowercase":
                df[col] = df[col].astype(str).str.lower()
            elif operation == "Uppercase":
                df[col] = df[col].astype(str).str.upper()
            elif operation == "Title Case":
                df[col] = df[col].astype(str).str.title()
            elif operation == "Strip whitespace":
                df[col] = df[col].astype(str).str.strip()
            elif operation == "Remove special characters":
                df[col] = df[col].astype(str).str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
        
        current_df = df
        return f"✓ Text standardized: {operation} for {len(columns)} column(s)", df.head(20)
    except Exception as e:
        df_history.pop()
        return f"Error: {str(e)}", None

def scale_normalize(columns, method):
    global current_df
    if current_df is None:
        return "No dataset loaded", None
    
    if not columns:
        return "Please select at least one column", None
    
    save_to_history()
    df = current_df.copy()
    
    try:
        if method == "Standard Scaler (Z-score)":
            scaler = StandardScaler()
            df[columns] = scaler.fit_transform(df[columns])
        elif method == "Min-Max Scaler (0-1)":
            scaler = MinMaxScaler()
            df[columns] = scaler.fit_transform(df[columns])
        elif method == "Robust Scaler":
            scaler = RobustScaler()
            df[columns] = scaler.fit_transform(df[columns])
        
        current_df = df
        return f"✓ Scaling applied: {method} for {len(columns)} column(s)", df.head(20)
    except Exception as e:
        df_history.pop()
        return f"Error: {str(e)}", None

def download_cleaned_data():
    global current_df
    if current_df is None:
        return None
    
    output_path = "cleaned_data.csv"
    current_df.to_csv(output_path, index=False)
    return output_path

def reset_data():
    global current_df, df_history
    current_df = None
    df_history = []
    return "Dataset reset. Please upload a new file.", None

# EDA Functions
def understand_data():
    global current_df
    if current_df is None:
        return "No dataset loaded", None
    
    info = []
    info.append(f"### Dataset Overview\n")
    info.append(f"**Shape:** {current_df.shape[0]} rows × {current_df.shape[1]} columns\n")
    info.append(f"**Memory Usage:** {current_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")
    info.append(f"\n**Column Information:**\n")
    
    for col in current_df.columns:
        dtype = current_df[col].dtype
        non_null = current_df[col].count()
        null_count = current_df[col].isnull().sum()
        unique = current_df[col].nunique()
        info.append(f"- **{col}**: {dtype} | Non-Null: {non_null} | Null: {null_count} | Unique: {unique}")
    
    return "\n".join(info), current_df.head(10)

def descriptive_stats(stat_type):
    global current_df
    if current_df is None:
        return None
    
    try:
        if stat_type == "Numeric Only":
            return current_df.describe()
        elif stat_type == "All Columns":
            return current_df.describe(include='all')
        else:  # Categorical Only
            return current_df.describe(include=['object', 'category'])
    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})

def univariate_analysis(column, plot_type):
    global current_df
    if current_df is None or not column:
        return None, "No data or column selected"
    
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if plot_type == "Histogram":
            if pd.api.types.is_numeric_dtype(current_df[column]):
                ax.hist(current_df[column].dropna(), bins=30, edgecolor='black', alpha=0.7)
                ax.set_xlabel(column)
                ax.set_ylabel("Frequency")
                ax.set_title(f"Histogram of {column}")
                ax.grid(True, alpha=0.3)
            else:
                plt.close(fig)
                return None, "Histogram only works with numeric columns"
        
        elif plot_type == "Box Plot":
            if pd.api.types.is_numeric_dtype(current_df[column]):
                ax.boxplot(current_df[column].dropna(), vert=True)
                ax.set_ylabel(column)
                ax.set_title(f"Box Plot of {column}")
                ax.grid(True, alpha=0.3)
            else:
                plt.close(fig)
                return None, "Box plot only works with numeric columns"
        
        elif plot_type == "Value Counts":
            value_counts = current_df[column].value_counts().head(20)
            ax.bar(range(len(value_counts)), value_counts.values)
            ax.set_xticks(range(len(value_counts)))
            ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
            ax.set_xlabel(column)
            ax.set_ylabel("Count")
            ax.set_title(f"Value Counts of {column}")
            ax.grid(True, alpha=0.3)
        
        elif plot_type == "Statistics":
            stats_text = f"### Statistics for {column}\n\n"
            if pd.api.types.is_numeric_dtype(current_df[column]):
                stats_text += f"**Mean:** {current_df[column].mean():.2f}\n\n"
                stats_text += f"**Median:** {current_df[column].median():.2f}\n\n"
                stats_text += f"**Std Dev:** {current_df[column].std():.2f}\n\n"
                stats_text += f"**Min:** {current_df[column].min():.2f}\n\n"
                stats_text += f"**Max:** {current_df[column].max():.2f}\n\n"
            stats_text += f"**Unique Values:** {current_df[column].nunique()}\n\n"
            stats_text += f"**Missing Values:** {current_df[column].isnull().sum()}\n\n"
            plt.close(fig)
            return None, stats_text
        
        plt.tight_layout()
        return fig, ""
    except Exception as e:
        plt.close(fig)
        return None, f"Error: {str(e)}"

def bivariate_analysis(col1, col2, plot_type):
    global current_df
    if current_df is None or not col1 or not col2:
        return None, "Please select both columns"
    
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if plot_type == "Scatter Plot":
            if pd.api.types.is_numeric_dtype(current_df[col1]) and pd.api.types.is_numeric_dtype(current_df[col2]):
                ax.scatter(current_df[col1], current_df[col2], alpha=0.5)
                ax.set_xlabel(col1)
                ax.set_ylabel(col2)
                ax.set_title(f"Scatter Plot: {col1} vs {col2}")
                ax.grid(True, alpha=0.3)
            else:
                plt.close(fig)
                return None, "Scatter plot requires numeric columns"
        
        elif plot_type == "Line Plot":
            if pd.api.types.is_numeric_dtype(current_df[col2]):
                ax.plot(current_df[col1], current_df[col2], marker='o', markersize=3)
                ax.set_xlabel(col1)
                ax.set_ylabel(col2)
                ax.set_title(f"Line Plot: {col1} vs {col2}")
                ax.grid(True, alpha=0.3)
            else:
                plt.close(fig)
                return None, "Line plot requires numeric Y-axis"
        
        elif plot_type == "Bar Plot":
            grouped = current_df.groupby(col1)[col2].mean().head(20)
            ax.bar(range(len(grouped)), grouped.values)
            ax.set_xticks(range(len(grouped)))
            ax.set_xticklabels(grouped.index, rotation=45, ha='right')
            ax.set_xlabel(col1)
            ax.set_ylabel(f"Mean of {col2}")
            ax.set_title(f"Bar Plot: {col1} vs {col2}")
            ax.grid(True, alpha=0.3)
        
        elif plot_type == "Correlation":
            if pd.api.types.is_numeric_dtype(current_df[col1]) and pd.api.types.is_numeric_dtype(current_df[col2]):
                corr = current_df[[col1, col2]].corr().iloc[0, 1]
                stats_text = f"### Correlation Analysis\n\n"
                stats_text += f"**Pearson Correlation:** {corr:.4f}\n\n"
                if abs(corr) > 0.7:
                    stats_text += "**Interpretation:** Strong correlation\n"
                elif abs(corr) > 0.4:
                    stats_text += "**Interpretation:** Moderate correlation\n"
                else:
                    stats_text += "**Interpretation:** Weak correlation\n"
                plt.close(fig)
                return None, stats_text
            else:
                plt.close(fig)
                return None, "Correlation requires numeric columns"
        
        plt.tight_layout()
        return fig, ""
    except Exception as e:
        plt.close(fig)
        return None, f"Error: {str(e)}"

def correlation_matrix(method):
    global current_df
    if current_df is None:
        return None
    
    try:
        numeric_df = current_df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return None
        
        corr = numeric_df.corr(method=method)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, ax=ax, cbar_kws={"shrink": 0.8})
        ax.set_title(f"Correlation Matrix ({method.capitalize()})")
        plt.tight_layout()
        return fig
    except Exception as e:
        return None

def missing_value_analysis():
    global current_df
    if current_df is None:
        return None, "No dataset loaded"
    
    try:
        missing = current_df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        
        if len(missing) == 0:
            return None, "### No missing values found! ✓"
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(len(missing)), missing.values)
        ax.set_xticks(range(len(missing)))
        ax.set_xticklabels(missing.index, rotation=45, ha='right')
        ax.set_xlabel("Columns")
        ax.set_ylabel("Missing Count")
        ax.set_title("Missing Values by Column")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        details = "### Missing Value Details\n\n"
        for col, count in missing.items():
            pct = (count / len(current_df)) * 100
            details += f"**{col}**: {count} ({pct:.1f}%)\n\n"
        
        return fig, details
    except Exception as e:
        return None, f"Error: {str(e)}"

def outlier_detection(column, method):
    global current_df
    if current_df is None or not column:
        return None, "No data or column selected"
    
    try:
        if not pd.api.types.is_numeric_dtype(current_df[column]):
            return None, "Please select a numeric column"
        
        data = current_df[column].dropna()
        
        if method == "Box Plot":
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.boxplot(data, vert=True)
            ax.set_ylabel(column)
            ax.set_title(f"Box Plot - Outlier Detection for {column}")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            return fig, ""
        
        elif method == "IQR Analysis":
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = data[(data < lower) | (data > upper)]
            
            stats_text = f"### IQR Outlier Analysis for {column}\n\n"
            stats_text += f"**Q1 (25%):** {Q1:.2f}\n\n"
            stats_text += f"**Q3 (75%):** {Q3:.2f}\n\n"
            stats_text += f"**IQR:** {IQR:.2f}\n\n"
            stats_text += f"**Lower Bound:** {lower:.2f}\n\n"
            stats_text += f"**Upper Bound:** {upper:.2f}\n\n"
            stats_text += f"**Number of Outliers:** {len(outliers)} ({len(outliers)/len(data)*100:.1f}%)\n\n"
            
            return None, stats_text
        
        elif method == "Z-Score Analysis":
            z_scores = np.abs(stats.zscore(data))
            outliers = data[z_scores > 3]
            
            stats_text = f"### Z-Score Outlier Analysis for {column}\n\n"
            stats_text += f"**Mean:** {data.mean():.2f}\n\n"
            stats_text += f"**Std Dev:** {data.std():.2f}\n\n"
            stats_text += f"**Outliers (|Z| > 3):** {len(outliers)} ({len(outliers)/len(data)*100:.1f}%)\n\n"
            
            return None, stats_text
        
    except Exception as e:
        return None, f"Error: {str(e)}"

def distribution_analysis(column):
    global current_df
    if current_df is None or not column:
        return None, "No data or column selected"
    
    from scipy import stats as sp_stats
    
    try:
        if not pd.api.types.is_numeric_dtype(current_df[column]):
            return None, "Please select a numeric column"
        
        data = current_df[column].dropna()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram with KDE
        ax1.hist(data, bins=30, density=True, alpha=0.7, edgecolor='black')
        data.plot(kind='kde', ax=ax1, color='red', linewidth=2)
        ax1.set_xlabel(column)
        ax1.set_ylabel("Density")
        ax1.set_title(f"Distribution of {column}")
        ax1.grid(True, alpha=0.3)
        
        # Q-Q Plot
        sp_stats.probplot(data, dist="norm", plot=ax2)
        ax2.set_title("Q-Q Plot (Normal Distribution)")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Statistics
        skewness = sp_stats.skew(data)
        kurtosis = sp_stats.kurtosis(data)
        
        stats_text = f"### Distribution Statistics for {column}\n\n"
        stats_text += f"**Mean:** {data.mean():.2f}\n\n"
        stats_text += f"**Median:** {data.median():.2f}\n\n"
        stats_text += f"**Std Dev:** {data.std():.2f}\n\n"
        stats_text += f"**Skewness:** {skewness:.2f} "
        if abs(skewness) < 0.5:
            stats_text += "(Fairly symmetric)\n\n"
        elif skewness > 0:
            stats_text += "(Right-skewed)\n\n"
        else:
            stats_text += "(Left-skewed)\n\n"
        stats_text += f"**Kurtosis:** {kurtosis:.2f}\n\n"
        
        return fig, stats_text
    except Exception as e:
        return None, f"Error: {str(e)}"

def categorical_analysis(column):
    global current_df
    if current_df is None or not column:
        return None, "No data or column selected"
    
    try:
        value_counts = current_df[column].value_counts().head(15)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar plot
        ax1.bar(range(len(value_counts)), value_counts.values)
        ax1.set_xticks(range(len(value_counts)))
        ax1.set_xticklabels(value_counts.index, rotation=45, ha='right')
        ax1.set_xlabel(column)
        ax1.set_ylabel("Count")
        ax1.set_title(f"Top Categories - {column}")
        ax1.grid(True, alpha=0.3)
        
        # Pie chart
        ax2.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%', startangle=90)
        ax2.set_title(f"Distribution - {column}")
        
        plt.tight_layout()
        
        # Statistics
        stats_text = f"### Categorical Analysis for {column}\n\n"
        stats_text += f"**Unique Values:** {current_df[column].nunique()}\n\n"
        stats_text += f"**Most Common:** {value_counts.index[0]} ({value_counts.values[0]} occurrences)\n\n"
        stats_text += f"**Mode Frequency:** {value_counts.values[0]/len(current_df)*100:.1f}%\n\n"
        
        return fig, stats_text
    except Exception as e:
        return None, f"Error: {str(e)}"

with gr.Blocks(title="EDA App with Data Cleaning", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 📊 EDA & Data Cleaning App")
    gr.Markdown("Upload CSV, Excel, or JSON files for exploration and cleaning")
    
    with gr.Tab("📁 Upload & Overview"):
        with gr.Row():
            file_input = gr.File(
                label="Upload Dataset",
                file_types=[".csv", ".xlsx", ".json"]
            )
        
        with gr.Row():
            info_output = gr.Textbox(label="Dataset Info", lines=2)
        
        table_output = gr.Dataframe(label="Data Preview", wrap=True)
    
    with gr.Tab("🧹 Data Cleaning"):
        gr.Markdown("### Data Quality Summary")
        with gr.Row():
            summary_btn = gr.Button("Generate Summary", variant="primary")
            undo_btn = gr.Button("↩️ Undo Last Action", variant="secondary", size="sm")
        
        summary_output = gr.Markdown()
        
        with gr.Row():
            undo_status = gr.Textbox(label="Undo Status", visible=False)
            undo_preview = gr.Dataframe(label="Preview After Undo", visible=False)
        
        summary_btn.click(
            fn=get_cleaning_summary,
            outputs=summary_output
        )
        
        undo_btn.click(
            fn=undo_last_action,
            outputs=[undo_status, undo_preview]
        ).then(
            lambda: (gr.update(visible=True), gr.update(visible=True)),
            outputs=[undo_status, undo_preview]
        )
        
        gr.Markdown("---")
        
        with gr.Accordion("1️⃣ Handle Missing Values", open=False):
            with gr.Row():
                missing_strategy = gr.Dropdown(
                    choices=["Drop rows", "Fill with mean", "Fill with median", 
                            "Fill with mode", "Fill with custom value", 
                            "Forward fill", "Backward fill"],
                    label="Strategy",
                    value="Drop rows"
                )
                missing_cols = gr.Dropdown(
                    choices=[],
                    label="Select Columns",
                    multiselect=True,
                    interactive=True
                )
                fill_val = gr.Textbox(label="Custom Fill Value (if applicable)", value="0")
            
            with gr.Row():
                missing_btn = gr.Button("Apply", variant="primary")
                missing_undo_btn = gr.Button("↩️ Undo", variant="secondary")
            
            missing_status = gr.Textbox(label="Status")
            missing_preview = gr.Dataframe(label="Preview")
            
            missing_btn.click(
                fn=handle_missing_values,
                inputs=[missing_strategy, missing_cols, fill_val],
                outputs=[missing_status, missing_preview]
            )
            
            missing_undo_btn.click(
                fn=undo_last_action,
                outputs=[missing_status, missing_preview]
            )
        
        with gr.Accordion("2️⃣ Remove Duplicates", open=False):
            dup_cols = gr.Dropdown(
                choices=[],
                label="Select Subset Columns (leave empty for all columns)",
                multiselect=True,
                interactive=True
            )
            
            with gr.Row():
                dup_btn = gr.Button("Remove Duplicates", variant="primary")
                dup_undo_btn = gr.Button("↩️ Undo", variant="secondary")
            
            dup_status = gr.Textbox(label="Status")
            dup_preview = gr.Dataframe(label="Preview")
            
            dup_btn.click(
                fn=remove_duplicates,
                inputs=dup_cols,
                outputs=[dup_status, dup_preview]
            )
            
            dup_undo_btn.click(
                fn=undo_last_action,
                outputs=[dup_status, dup_preview]
            )
        
        with gr.Accordion("3️⃣ Handle Outliers", open=False):
            with gr.Row():
                outlier_method = gr.Dropdown(
                    choices=["IQR Method", "Z-Score Method", "Cap outliers"],
                    label="Method",
                    value="IQR Method"
                )
                outlier_cols = gr.Dropdown(
                    choices=[],
                    label="Select Numeric Columns",
                    multiselect=True,
                    interactive=True
                )
                z_threshold = gr.Number(label="Z-Score Threshold", value=3)
            
            with gr.Row():
                outlier_btn = gr.Button("Apply", variant="primary")
                outlier_undo_btn = gr.Button("↩️ Undo", variant="secondary")
            
            outlier_status = gr.Textbox(label="Status")
            outlier_preview = gr.Dataframe(label="Preview")
            
            outlier_btn.click(
                fn=handle_outliers,
                inputs=[outlier_method, outlier_cols, z_threshold],
                outputs=[outlier_status, outlier_preview]
            )
            
            outlier_undo_btn.click(
                fn=undo_last_action,
                outputs=[outlier_status, outlier_preview]
            )
        
        with gr.Accordion("4️⃣ Correct Data Types", open=False):
            with gr.Row():
                dtype_col = gr.Dropdown(
                    choices=[],
                    label="Select Column",
                    interactive=True
                )
                dtype_type = gr.Dropdown(
                    choices=["int", "float", "string", "datetime", "category", "bool"],
                    label="New Type",
                    value="int"
                )
            
            with gr.Row():
                dtype_btn = gr.Button("Convert", variant="primary")
                dtype_undo_btn = gr.Button("↩️ Undo", variant="secondary")
            
            dtype_status = gr.Textbox(label="Status")
            dtype_preview = gr.Dataframe(label="Preview")
            
            dtype_btn.click(
                fn=correct_data_types,
                inputs=[dtype_col, dtype_type],
                outputs=[dtype_status, dtype_preview]
            )
            
            dtype_undo_btn.click(
                fn=undo_last_action,
                outputs=[dtype_status, dtype_preview]
            )
        
        with gr.Accordion("5️⃣ Standardize Text", open=False):
            with gr.Row():
                text_cols = gr.Dropdown(
                    choices=[],
                    label="Select Text Columns",
                    multiselect=True,
                    interactive=True
                )
                text_operation = gr.Dropdown(
                    choices=["Lowercase", "Uppercase", "Title Case", 
                            "Strip whitespace", "Remove special characters"],
                    label="Operation",
                    value="Lowercase"
                )
            
            with gr.Row():
                text_btn = gr.Button("Apply", variant="primary")
                text_undo_btn = gr.Button("↩️ Undo", variant="secondary")
            
            text_status = gr.Textbox(label="Status")
            text_preview = gr.Dataframe(label="Preview")
            
            text_btn.click(
                fn=standardize_text,
                inputs=[text_cols, text_operation],
                outputs=[text_status, text_preview]
            )
            
            text_undo_btn.click(
                fn=undo_last_action,
                outputs=[text_status, text_preview]
            )
        
        with gr.Accordion("6️⃣ Scaling & Normalization", open=False):
            with gr.Row():
                scale_cols = gr.Dropdown(
                    choices=[],
                    label="Select Numeric Columns",
                    multiselect=True,
                    interactive=True
                )
                scale_method = gr.Dropdown(
                    choices=["Standard Scaler (Z-score)", "Min-Max Scaler (0-1)", "Robust Scaler"],
                    label="Scaling Method",
                    value="Standard Scaler (Z-score)"
                )
            
            with gr.Row():
                scale_btn = gr.Button("Apply", variant="primary")
                scale_undo_btn = gr.Button("↩️ Undo", variant="secondary")
            
            scale_status = gr.Textbox(label="Status")
            scale_preview = gr.Dataframe(label="Preview")
            
            scale_btn.click(
                fn=scale_normalize,
                inputs=[scale_cols, scale_method],
                outputs=[scale_status, scale_preview]
            )
            
            scale_undo_btn.click(
                fn=undo_last_action,
                outputs=[scale_status, scale_preview]
            )
        
        gr.Markdown("---")
        
        with gr.Row():
            download_btn = gr.Button("📥 Download Cleaned Data", variant="secondary", size="lg")
            reset_btn = gr.Button("🔄 Reset Dataset", variant="stop", size="lg")
        
        download_file = gr.File(label="Download")
        reset_status = gr.Textbox(label="Status")
        reset_preview = gr.Dataframe(label="Preview")
        
        download_btn.click(
            fn=download_cleaned_data,
            outputs=download_file
        )
        
        reset_btn.click(
            fn=reset_data,
            outputs=[reset_status, reset_preview]
        )
    
    with gr.Tab("📊 EDA (Exploratory Data Analysis)"):
        gr.Markdown("### Exploratory Data Analysis Tools")
        
        with gr.Accordion("1️⃣ Understanding Data", open=True):
            understand_btn = gr.Button("Show Data Overview", variant="primary")
            understand_output = gr.Markdown()
            understand_table = gr.Dataframe(label="Sample Data")
            
            understand_btn.click(
                fn=understand_data,
                outputs=[understand_output, understand_table]
            )
        
        with gr.Accordion("2️⃣ Descriptive Statistics", open=False):
            with gr.Row():
                desc_type = gr.Radio(
                    choices=["Numeric Only", "All Columns", "Categorical Only"],
                    label="Statistics Type",
                    value="Numeric Only"
                )
            desc_btn = gr.Button("Generate Statistics", variant="primary")
            desc_output = gr.Dataframe(label="Descriptive Statistics")
            
            desc_btn.click(
                fn=descriptive_stats,
                inputs=desc_type,
                outputs=desc_output
            )
        
        with gr.Accordion("3️⃣ Univariate Analysis", open=False):
            with gr.Row():
                uni_col = gr.Dropdown(choices=[], label="Select Column", interactive=True)
                uni_plot_type = gr.Dropdown(
                    choices=["Histogram", "Box Plot", "Value Counts", "Statistics"],
                    label="Analysis Type",
                    value="Histogram"
                )
            uni_btn = gr.Button("Analyze", variant="primary")
            uni_output = gr.Plot(label="Visualization")
            uni_stats = gr.Markdown(label="Statistics")
            
            uni_btn.click(
                fn=univariate_analysis,
                inputs=[uni_col, uni_plot_type],
                outputs=[uni_output, uni_stats]
            )
        
        with gr.Accordion("4️⃣ Bivariate Analysis", open=False):
            with gr.Row():
                bi_col1 = gr.Dropdown(choices=[], label="Select Column 1", interactive=True)
                bi_col2 = gr.Dropdown(choices=[], label="Select Column 2", interactive=True)
                bi_plot_type = gr.Dropdown(
                    choices=["Scatter Plot", "Line Plot", "Bar Plot", "Correlation"],
                    label="Plot Type",
                    value="Scatter Plot"
                )
            bi_btn = gr.Button("Analyze", variant="primary")
            bi_output = gr.Plot(label="Visualization")
            bi_stats = gr.Markdown(label="Statistics")
            
            bi_btn.click(
                fn=bivariate_analysis,
                inputs=[bi_col1, bi_col2, bi_plot_type],
                outputs=[bi_output, bi_stats]
            )
        
        with gr.Accordion("5️⃣ Correlation Matrix (Multivariate)", open=False):
            with gr.Row():
                corr_method = gr.Dropdown(
                    choices=["pearson", "spearman", "kendall"],
                    label="Correlation Method",
                    value="pearson"
                )
            corr_btn = gr.Button("Generate Correlation Matrix", variant="primary")
            corr_output = gr.Plot(label="Correlation Heatmap")
            
            corr_btn.click(
                fn=correlation_matrix,
                inputs=corr_method,
                outputs=corr_output
            )
        
        with gr.Accordion("6️⃣ Missing Value Analysis", open=False):
            missing_analysis_btn = gr.Button("Analyze Missing Values", variant="primary")
            missing_plot = gr.Plot(label="Missing Values Visualization")
            missing_details = gr.Markdown(label="Details")
            
            missing_analysis_btn.click(
                fn=missing_value_analysis,
                outputs=[missing_plot, missing_details]
            )
        
        with gr.Accordion("7️⃣ Outlier Detection", open=False):
            outlier_col_eda = gr.Dropdown(choices=[], label="Select Numeric Column", interactive=True)
            outlier_method_vis = gr.Dropdown(
                choices=["Box Plot", "IQR Analysis", "Z-Score Analysis"],
                label="Detection Method",
                value="Box Plot"
            )
            outlier_btn_eda = gr.Button("Detect Outliers", variant="primary")
            outlier_plot_eda = gr.Plot(label="Visualization")
            outlier_stats_eda = gr.Markdown(label="Outlier Statistics")
            
            outlier_btn_eda.click(
                fn=outlier_detection,
                inputs=[outlier_col_eda, outlier_method_vis],
                outputs=[outlier_plot_eda, outlier_stats_eda]
            )
        
        with gr.Accordion("8️⃣ Distribution Analysis", open=False):
            dist_col = gr.Dropdown(choices=[], label="Select Numeric Column", interactive=True)
            dist_btn = gr.Button("Analyze Distribution", variant="primary")
            dist_plot = gr.Plot(label="Distribution Plot")
            dist_stats = gr.Markdown(label="Distribution Statistics")
            
            dist_btn.click(
                fn=distribution_analysis,
                inputs=dist_col,
                outputs=[dist_plot, dist_stats]
            )
        
        with gr.Accordion("9️⃣ Categorical Data Analysis", open=False):
            cat_col = gr.Dropdown(choices=[], label="Select Categorical Column", interactive=True)
            cat_btn = gr.Button("Analyze", variant="primary")
            cat_plot = gr.Plot(label="Visualization")
            cat_stats = gr.Markdown(label="Statistics")
            
            cat_btn.click(
                fn=categorical_analysis,
                inputs=cat_col,
                outputs=[cat_plot, cat_stats]
            )

    with gr.Tab("⚙️ Feature Engineering"):
        gr.Markdown("### Feature Engineering Tools")
        
        with gr.Accordion("1️⃣ Feature Creation", open=False):
            gr.Markdown("#### Combine or Transform Features")
            
            with gr.Row():
                fc_operation = gr.Dropdown(
                    choices=["Combine (Add)", "Combine (Multiply)", "Combine (Divide)", 
                            "Mathematical Transform", "DateTime Features"],
                    label="Operation Type",
                    value="Combine (Add)"
                )
            
            with gr.Row():
                fc_col1 = gr.Dropdown(choices=[], label="Select Column 1", interactive=True)
                fc_col2 = gr.Dropdown(choices=[], label="Select Column 2 (if combining)", interactive=True)
                fc_new_name = gr.Textbox(label="New Feature Name", placeholder="new_feature")
            
            with gr.Row():
                fc_math_op = gr.Dropdown(
                    choices=["Square", "Cube", "Square Root", "Absolute"],
                    label="Math Operation (if selected)",
                    value="Square"
                )
            
            with gr.Row():
                fc_btn = gr.Button("Create Feature", variant="primary")
                fc_undo_btn = gr.Button("↩️ Undo", variant="secondary")
            
            fc_status = gr.Textbox(label="Status")
            fc_preview = gr.Dataframe(label="Preview")
            
            def create_feature(operation, col1, col2, new_name, math_op):
                global current_df
                if current_df is None:
                    return "No dataset loaded", None
                
                if not col1 or not new_name:
                    return "Please select column and provide feature name", None
                
                save_to_history()
                df = current_df.copy()
                
                try:
                    if operation == "Combine (Add)":
                        if not col2:
                            return "Please select Column 2 for combination", None
                        df[new_name] = df[col1] + df[col2]
                    elif operation == "Combine (Multiply)":
                        if not col2:
                            return "Please select Column 2 for combination", None
                        df[new_name] = df[col1] * df[col2]
                    elif operation == "Combine (Divide)":
                        if not col2:
                            return "Please select Column 2 for combination", None
                        df[new_name] = df[col1] / df[col2].replace(0, np.nan)
                    elif operation == "Mathematical Transform":
                        if math_op == "Square":
                            df[new_name] = df[col1] ** 2
                        elif math_op == "Cube":
                            df[new_name] = df[col1] ** 3
                        elif math_op == "Square Root":
                            df[new_name] = np.sqrt(df[col1].abs())
                        elif math_op == "Absolute":
                            df[new_name] = df[col1].abs()
                    elif operation == "DateTime Features":
                        df[col1] = pd.to_datetime(df[col1], errors='coerce')
                        df[f"{col1}_year"] = df[col1].dt.year
                        df[f"{col1}_month"] = df[col1].dt.month
                        df[f"{col1}_day"] = df[col1].dt.day
                        df[f"{col1}_dayofweek"] = df[col1].dt.dayofweek
                        current_df = df
                        return "✓ DateTime features created (year, month, day, dayofweek)", df.head(20)
                    
                    current_df = df
                    return f"✓ Feature '{new_name}' created successfully", df.head(20)
                except Exception as e:
                    df_history.pop()
                    return f"Error: {str(e)}", None
            
            fc_btn.click(
                fn=create_feature,
                inputs=[fc_operation, fc_col1, fc_col2, fc_new_name, fc_math_op],
                outputs=[fc_status, fc_preview]
            )
            
            fc_undo_btn.click(
                fn=undo_last_action,
                outputs=[fc_status, fc_preview]
            )
        
        with gr.Accordion("2️⃣ Feature Transformation", open=False):
            with gr.Row():
                ft_cols = gr.Dropdown(choices=[], label="Select Columns", multiselect=True, interactive=True)
                ft_method = gr.Dropdown(
                    choices=["Log Transform", "Square Transform", "Power Transform (Yeo-Johnson)"],
                    label="Transformation Method",
                    value="Log Transform"
                )
            
            with gr.Row():
                ft_btn = gr.Button("Transform", variant="primary")
                ft_undo_btn = gr.Button("↩️ Undo", variant="secondary")
            
            ft_status = gr.Textbox(label="Status")
            ft_preview = gr.Dataframe(label="Preview")
            
            def transform_features(columns, method):
                global current_df
                if current_df is None:
                    return "No dataset loaded", None
                
                if not columns:
                    return "Please select at least one column", None
                
                save_to_history()
                df = current_df.copy()
                
                try:
                    from sklearn.preprocessing import PowerTransformer
                    
                    for col in columns:
                        if not pd.api.types.is_numeric_dtype(df[col]):
                            continue
                        
                        if method == "Log Transform":
                            df[col] = np.log1p(df[col].clip(lower=0))
                        elif method == "Square Transform":
                            df[col] = df[col] ** 2
                        elif method == "Power Transform (Yeo-Johnson)":
                            pt = PowerTransformer(method='yeo-johnson')
                            df[col] = pt.fit_transform(df[[col]])
                    
                    current_df = df
                    return f"✓ Transformation applied: {method}", df.head(20)
                except Exception as e:
                    df_history.pop()
                    return f"Error: {str(e)}", None
            
            ft_btn.click(
                fn=transform_features,
                inputs=[ft_cols, ft_method],
                outputs=[ft_status, ft_preview]
            )
            
            ft_undo_btn.click(
                fn=undo_last_action,
                outputs=[ft_status, ft_preview]
            )
        
        with gr.Accordion("3️⃣ Encoding Categorical Variables", open=False):
            with gr.Row():
                enc_cols = gr.Dropdown(choices=[], label="Select Columns", multiselect=True, interactive=True)
                enc_method = gr.Dropdown(
                    choices=["Label Encoding", "One-Hot Encoding", "Ordinal Encoding"],
                    label="Encoding Method",
                    value="Label Encoding"
                )
            
            with gr.Row():
                enc_btn = gr.Button("Encode", variant="primary")
                enc_undo_btn = gr.Button("↩️ Undo", variant="secondary")
            
            enc_status = gr.Textbox(label="Status")
            enc_preview = gr.Dataframe(label="Preview")
            
            def encode_features(columns, method):
                global current_df
                if current_df is None:
                    return "No dataset loaded", None
                
                if not columns:
                    return "Please select at least one column", None
                
                save_to_history()
                df = current_df.copy()
                
                try:
                    from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
                    
                    if method == "Label Encoding":
                        for col in columns:
                            le = LabelEncoder()
                            df[col] = le.fit_transform(df[col].astype(str))
                    
                    elif method == "One-Hot Encoding":
                        df = pd.get_dummies(df, columns=columns, prefix=columns, drop_first=False)
                    
                    elif method == "Ordinal Encoding":
                        oe = OrdinalEncoder()
                        df[columns] = oe.fit_transform(df[columns].astype(str))
                    
                    current_df = df
                    return f"✓ Encoding applied: {method}", df.head(20)
                except Exception as e:
                    df_history.pop()
                    return f"Error: {str(e)}", None
            
            enc_btn.click(
                fn=encode_features,
                inputs=[enc_cols, enc_method],
                outputs=[enc_status, enc_preview]
            )
            
            enc_undo_btn.click(
                fn=undo_last_action,
                outputs=[enc_status, enc_preview]
            )
        
        with gr.Accordion("4️⃣ Binning / Discretization", open=False):
            with gr.Row():
                bin_col = gr.Dropdown(choices=[], label="Select Column", interactive=True)
                bin_method = gr.Dropdown(
                    choices=["Equal Width", "Equal Frequency"],
                    label="Binning Method",
                    value="Equal Width"
                )
                bin_count = gr.Number(label="Number of Bins", value=5)
            
            with gr.Row():
                bin_btn = gr.Button("Apply Binning", variant="primary")
                bin_undo_btn = gr.Button("↩️ Undo", variant="secondary")
            
            bin_status = gr.Textbox(label="Status")
            bin_preview = gr.Dataframe(label="Preview")
            
            def apply_binning(column, method, n_bins):
                global current_df
                if current_df is None:
                    return "No dataset loaded", None
                
                if not column:
                    return "Please select a column", None
                
                save_to_history()
                df = current_df.copy()
                
                try:
                    if not pd.api.types.is_numeric_dtype(df[column]):
                        return "Binning requires numeric columns", None
                    
                    n_bins = int(n_bins)
                    
                    if method == "Equal Width":
                        df[f"{column}_binned"] = pd.cut(df[column], bins=n_bins, labels=False)
                    elif method == "Equal Frequency":
                        df[f"{column}_binned"] = pd.qcut(df[column], q=n_bins, labels=False, duplicates='drop')
                    
                    current_df = df
                    return f"✓ Binning applied: {method} with {n_bins} bins", df.head(20)
                except Exception as e:
                    df_history.pop()
                    return f"Error: {str(e)}", None
            
            bin_btn.click(
                fn=apply_binning,
                inputs=[bin_col, bin_method, bin_count],
                outputs=[bin_status, bin_preview]
            )
            
            bin_undo_btn.click(
                fn=undo_last_action,
                outputs=[bin_status, bin_preview]
            )
        
        with gr.Accordion("5️⃣ Feature Selection", open=False):
            with gr.Row():
                fs_method = gr.Dropdown(
                    choices=["Correlation Threshold", "Variance Threshold"],
                    label="Selection Method",
                    value="Correlation Threshold"
                )
                fs_threshold = gr.Number(label="Threshold", value=0.9)
            
            fs_btn = gr.Button("Select Features", variant="primary")
            fs_status = gr.Textbox(label="Status")
            fs_info = gr.Markdown(label="Selected Features Info")
            
            def select_features(method, threshold):
                global current_df
                if current_df is None:
                    return "No dataset loaded", "No info"
                
                try:
                    df = current_df.copy()
                    numeric_df = df.select_dtypes(include=[np.number])
                    
                    if method == "Correlation Threshold":
                        corr_matrix = numeric_df.corr().abs()
                        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
                        
                        info = f"### Correlation-based Feature Selection\n\n"
                        info += f"**Threshold:** {threshold}\n\n"
                        info += f"**Features to drop (high correlation):** {len(to_drop)}\n\n"
                        if to_drop:
                            info += f"**Dropped features:** {', '.join(to_drop)}\n\n"
                        else:
                            info += "**No features exceeded the threshold**\n\n"
                        
                        return f"Found {len(to_drop)} features with correlation > {threshold}", info
                    
                    elif method == "Variance Threshold":
                        from sklearn.feature_selection import VarianceThreshold
                        selector = VarianceThreshold(threshold=threshold)
                        selector.fit(numeric_df)
                        
                        selected = numeric_df.columns[selector.get_support()].tolist()
                        removed = numeric_df.columns[~selector.get_support()].tolist()
                        
                        info = f"### Variance-based Feature Selection\n\n"
                        info += f"**Threshold:** {threshold}\n\n"
                        info += f"**Features kept:** {len(selected)}\n\n"
                        info += f"**Features removed:** {len(removed)}\n\n"
                        if removed:
                            info += f"**Removed features:** {', '.join(removed)}\n\n"
                        
                        return f"Selected {len(selected)} features with variance > {threshold}", info
                    
                except Exception as e:
                    return f"Error: {str(e)}", "Error occurred"
            
            fs_btn.click(
                fn=select_features,
                inputs=[fs_method, fs_threshold],
                outputs=[fs_status, fs_info]
            )
        
        with gr.Accordion("6️⃣ Dimensionality Reduction", open=False):
            with gr.Row():
                dr_method = gr.Dropdown(
                    choices=["PCA (Principal Component Analysis)"],
                    label="Method",
                    value="PCA (Principal Component Analysis)"
                )
                dr_components = gr.Number(label="Number of Components", value=2)
            
            with gr.Row():
                dr_btn = gr.Button("Apply Reduction", variant="primary")
                dr_undo_btn = gr.Button("↩️ Undo", variant="secondary")
            
            dr_status = gr.Textbox(label="Status")
            dr_preview = gr.Dataframe(label="Preview")
            dr_plot = gr.Plot(label="Explained Variance")
            
            def reduce_dimensions(method, n_components):
                global current_df
                if current_df is None:
                    return "No dataset loaded", None, None
                
                save_to_history()
                df = current_df.copy()
                
                try:
                    from sklearn.decomposition import PCA
                    import matplotlib.pyplot as plt
                    
                    numeric_df = df.select_dtypes(include=[np.number])
                    n_components = int(n_components)
                    
                    if method == "PCA (Principal Component Analysis)":
                        pca = PCA(n_components=n_components)
                        components = pca.fit_transform(numeric_df.fillna(0))
                        
                        # Add PCA components to dataframe
                        for i in range(n_components):
                            df[f'PC{i+1}'] = components[:, i]
                        
                        # Plot explained variance
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.bar(range(1, n_components + 1), pca.explained_variance_ratio_)
                        ax.set_xlabel('Principal Component')
                        ax.set_ylabel('Explained Variance Ratio')
                        ax.set_title('PCA Explained Variance')
                        ax.grid(True, alpha=0.3)
                        plt.tight_layout()
                        
                        current_df = df
                        total_var = pca.explained_variance_ratio_.sum()
                        return f"✓ PCA applied: {n_components} components (explains {total_var:.2%} variance)", df.head(20), fig
                    
                except Exception as e:
                    df_history.pop()
                    return f"Error: {str(e)}", None, None
            
            dr_btn.click(
                fn=reduce_dimensions,
                inputs=[dr_method, dr_components],
                outputs=[dr_status, dr_preview, dr_plot]
            )
            
            dr_undo_btn.click(
                fn=undo_last_action,
                outputs=[dr_status, dr_preview]
            )
        
        with gr.Accordion("7️⃣ Polynomial Features (Interaction)", open=False):
            with gr.Row():
                poly_cols = gr.Dropdown(choices=[], label="Select Columns", multiselect=True, interactive=True)
                poly_degree = gr.Number(label="Polynomial Degree", value=2)
            
            with gr.Row():
                poly_btn = gr.Button("Create Polynomial Features", variant="primary")
                poly_undo_btn = gr.Button("↩️ Undo", variant="secondary")
            
            poly_status = gr.Textbox(label="Status")
            poly_preview = gr.Dataframe(label="Preview")
            
            def create_polynomial(columns, degree):
                global current_df
                if current_df is None:
                    return "No dataset loaded", None
                
                if not columns:
                    return "Please select at least one column", None
                
                save_to_history()
                df = current_df.copy()
                
                try:
                    from sklearn.preprocessing import PolynomialFeatures
                    
                    degree = int(degree)
                    poly = PolynomialFeatures(degree=degree, include_bias=False)
                    poly_features = poly.fit_transform(df[columns])
                    
                    # Get feature names
                    feature_names = poly.get_feature_names_out(columns)
                    
                    # Add polynomial features
                    for i, name in enumerate(feature_names):
                        if name not in columns:  # Skip original features
                            df[name] = poly_features[:, i]
                    
                    current_df = df
                    new_features = len(feature_names) - len(columns)
                    return f"✓ Created {new_features} polynomial features (degree {degree})", df.head(20)
                except Exception as e:
                    df_history.pop()
                    return f"Error: {str(e)}", None
            
            poly_btn.click(
                fn=create_polynomial,
                inputs=[poly_cols, poly_degree],
                outputs=[poly_status, poly_preview]
            )
            
            poly_undo_btn.click(
                fn=undo_last_action,
                outputs=[poly_status, poly_preview]
            )
    # Update all dropdowns when file is uploaded
    file_input.change(
        fn=load_file_extended,
        inputs=file_input,
        outputs=[info_output, table_output, missing_cols, dup_cols, 
                outlier_cols, dtype_col, text_cols, scale_cols,
                uni_col, bi_col1, bi_col2, outlier_col_eda, dist_col, cat_col,
                fc_col1, fc_col2, ft_cols, enc_cols, bin_col, poly_cols]
    )
    

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
