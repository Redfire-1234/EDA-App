import gradio as gr
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy import stats

# Global variables to store the dataframe and history
current_df = None
df_history = []

def load_file(file):
    global current_df, df_history
    if file is None:
        return "No file uploaded", None, gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[])
    filename = file.name.lower()
    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(file.name)
        elif filename.endswith(".xlsx"):
            df = pd.read_excel(file.name)
        elif filename.endswith(".json"):
            df = pd.read_json(file.name)
        else:
            return "Unsupported file format", None, gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[])
        
        current_df = df.copy()
        df_history = [df.copy()]  # Initialize history with original data
        info = f"Rows: {df.shape[0]} | Columns: {df.shape[1]}"
        
        # Get column lists for different purposes
        all_cols = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        text_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        return (info, df.head(20), 
                gr.update(choices=all_cols), 
                gr.update(choices=all_cols),
                gr.update(choices=numeric_cols),
                gr.update(choices=all_cols),
                gr.update(choices=text_cols),
                gr.update(choices=numeric_cols),
                gr.update(choices=all_cols))
    except Exception as e:
        return f"Error: {str(e)}", None, gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[])

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
    
    # Update all dropdowns when file is uploaded
    file_input.change(
        fn=load_file,
        inputs=file_input,
        outputs=[info_output, table_output, missing_cols, dup_cols, 
                outlier_cols, dtype_col, text_cols, scale_cols, dtype_col]
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
