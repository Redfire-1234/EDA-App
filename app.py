import gradio as gr
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy import stats

# Global variable to store the dataframe
current_df = None

def load_file(file):
    global current_df
    if file is None:
        return "No file uploaded", None
    filename = file.name.lower()
    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(file.name)
        elif filename.endswith(".xlsx"):
            df = pd.read_excel(file.name)
        elif filename.endswith(".json"):
            df = pd.read_json(file.name)
        else:
            return "Unsupported file format", None
        
        current_df = df.copy()
        info = f"Rows: {df.shape[0]} | Columns: {df.shape[1]}"
        return info, df.head(20)
    except Exception as e:
        return f"Error: {str(e)}", None

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

def handle_missing_values(strategy, columns_str, fill_value):
    global current_df
    if current_df is None:
        return "No dataset loaded", None
    
    df = current_df.copy()
    
    if columns_str.strip():
        columns = [c.strip() for c in columns_str.split(",")]
        columns = [c for c in columns if c in df.columns]
    else:
        columns = df.columns.tolist()
    
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
        return f"✓ Missing values handled using '{strategy}'", df.head(20)
    except Exception as e:
        return f"Error: {str(e)}", None

def remove_duplicates(subset_cols):
    global current_df
    if current_df is None:
        return "No dataset loaded", None
    
    df = current_df.copy()
    initial_count = len(df)
    
    if subset_cols.strip():
        cols = [c.strip() for c in subset_cols.split(",")]
        cols = [c for c in cols if c in df.columns]
        df = df.drop_duplicates(subset=cols if cols else None)
    else:
        df = df.drop_duplicates()
    
    removed = initial_count - len(df)
    current_df = df
    return f"✓ Removed {removed} duplicate rows", df.head(20)

def handle_outliers(method, columns_str, threshold):
    global current_df
    if current_df is None:
        return "No dataset loaded", None
    
    df = current_df.copy()
    
    if columns_str.strip():
        columns = [c.strip() for c in columns_str.split(",")]
        columns = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    else:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
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
                df = df[np.abs(stats.zscore(df[col])) < threshold]
                removed_count += initial - len(df)
            elif method == "Cap outliers":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                df[col] = df[col].clip(lower, upper)
        
        current_df = df
        msg = f"✓ Outliers handled using '{method}'"
        if method != "Cap outliers":
            msg += f" ({removed_count} rows removed)"
        return msg, df.head(20)
    except Exception as e:
        return f"Error: {str(e)}", None

def correct_data_types(column, new_type):
    global current_df
    if current_df is None:
        return "No dataset loaded", None
    
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
        return f"Error: {str(e)}", None

def standardize_text(columns_str, operation):
    global current_df
    if current_df is None:
        return "No dataset loaded", None
    
    df = current_df.copy()
    
    if columns_str.strip():
        columns = [c.strip() for c in columns_str.split(",")]
        columns = [c for c in columns if c in df.columns]
    else:
        columns = df.select_dtypes(include=['object']).columns.tolist()
    
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
        return f"✓ Text standardized: {operation}", df.head(20)
    except Exception as e:
        return f"Error: {str(e)}", None

def scale_normalize(columns_str, method):
    global current_df
    if current_df is None:
        return "No dataset loaded", None
    
    df = current_df.copy()
    
    if columns_str.strip():
        columns = [c.strip() for c in columns_str.split(",")]
        columns = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    else:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
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
        return f"✓ Scaling applied: {method}", df.head(20)
    except Exception as e:
        return f"Error: {str(e)}", None

def download_cleaned_data():
    global current_df
    if current_df is None:
        return None
    
    output_path = "cleaned_data.csv"
    current_df.to_csv(output_path, index=False)
    return output_path

def reset_data():
    global current_df
    current_df = None
    return "Dataset reset. Please upload a new file.", None

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
        
        file_input.change(
            fn=load_file,
            inputs=file_input,
            outputs=[info_output, table_output]
        )
    
    with gr.Tab("🧹 Data Cleaning"):
        gr.Markdown("### Data Quality Summary")
        summary_btn = gr.Button("Generate Summary", variant="primary")
        summary_output = gr.Markdown()
        
        summary_btn.click(
            fn=get_cleaning_summary,
            outputs=summary_output
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
                missing_cols = gr.Textbox(
                    label="Columns (comma-separated, leave empty for all)",
                    placeholder="e.g., age, salary"
                )
                fill_val = gr.Textbox(label="Custom Fill Value (if applicable)", value="0")
            
            missing_btn = gr.Button("Apply", variant="primary")
            missing_status = gr.Textbox(label="Status")
            missing_preview = gr.Dataframe(label="Preview")
            
            missing_btn.click(
                fn=handle_missing_values,
                inputs=[missing_strategy, missing_cols, fill_val],
                outputs=[missing_status, missing_preview]
            )
        
        with gr.Accordion("2️⃣ Remove Duplicates", open=False):
            dup_cols = gr.Textbox(
                label="Subset Columns (comma-separated, leave empty for all)",
                placeholder="e.g., id, email"
            )
            dup_btn = gr.Button("Remove Duplicates", variant="primary")
            dup_status = gr.Textbox(label="Status")
            dup_preview = gr.Dataframe(label="Preview")
            
            dup_btn.click(
                fn=remove_duplicates,
                inputs=dup_cols,
                outputs=[dup_status, dup_preview]
            )
        
        with gr.Accordion("3️⃣ Handle Outliers", open=False):
            with gr.Row():
                outlier_method = gr.Dropdown(
                    choices=["IQR Method", "Z-Score Method", "Cap outliers"],
                    label="Method",
                    value="IQR Method"
                )
                outlier_cols = gr.Textbox(
                    label="Columns (comma-separated, leave empty for all numeric)",
                    placeholder="e.g., price, age"
                )
                z_threshold = gr.Number(label="Z-Score Threshold", value=3)
            
            outlier_btn = gr.Button("Apply", variant="primary")
            outlier_status = gr.Textbox(label="Status")
            outlier_preview = gr.Dataframe(label="Preview")
            
            outlier_btn.click(
                fn=handle_outliers,
                inputs=[outlier_method, outlier_cols, z_threshold],
                outputs=[outlier_status, outlier_preview]
            )
        
        with gr.Accordion("4️⃣ Correct Data Types", open=False):
            with gr.Row():
                dtype_col = gr.Textbox(label="Column Name", placeholder="e.g., age")
                dtype_type = gr.Dropdown(
                    choices=["int", "float", "string", "datetime", "category", "bool"],
                    label="New Type",
                    value="int"
                )
            
            dtype_btn = gr.Button("Convert", variant="primary")
            dtype_status = gr.Textbox(label="Status")
            dtype_preview = gr.Dataframe(label="Preview")
            
            dtype_btn.click(
                fn=correct_data_types,
                inputs=[dtype_col, dtype_type],
                outputs=[dtype_status, dtype_preview]
            )
        
        with gr.Accordion("5️⃣ Standardize Text", open=False):
            with gr.Row():
                text_cols = gr.Textbox(
                    label="Columns (comma-separated, leave empty for all text)",
                    placeholder="e.g., name, city"
                )
                text_operation = gr.Dropdown(
                    choices=["Lowercase", "Uppercase", "Title Case", 
                            "Strip whitespace", "Remove special characters"],
                    label="Operation",
                    value="Lowercase"
                )
            
            text_btn = gr.Button("Apply", variant="primary")
            text_status = gr.Textbox(label="Status")
            text_preview = gr.Dataframe(label="Preview")
            
            text_btn.click(
                fn=standardize_text,
                inputs=[text_cols, text_operation],
                outputs=[text_status, text_preview]
            )
        
        with gr.Accordion("6️⃣ Scaling & Normalization", open=False):
            with gr.Row():
                scale_cols = gr.Textbox(
                    label="Columns (comma-separated, leave empty for all numeric)",
                    placeholder="e.g., price, salary"
                )
                scale_method = gr.Dropdown(
                    choices=["Standard Scaler (Z-score)", "Min-Max Scaler (0-1)", "Robust Scaler"],
                    label="Scaling Method",
                    value="Standard Scaler (Z-score)"
                )
            
            scale_btn = gr.Button("Apply", variant="primary")
            scale_status = gr.Textbox(label="Status")
            scale_preview = gr.Dataframe(label="Preview")
            
            scale_btn.click(
                fn=scale_normalize,
                inputs=[scale_cols, scale_method],
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

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)

