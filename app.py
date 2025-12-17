import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

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
        preview = df.head().to_html()
        return f"✅ {file.name} uploaded successfully!\n\nShape: {df.shape}\n\nPreview:\n{preview}", df
    except Exception as e:
        return f"❌ Error loading file: {e}", None

# -----------------------------
# 2️⃣ Data Cleaning
# -----------------------------
def handle_missing_data(action):
    if state["df"] is None:
        return "⚠️ Please upload data first", None
    
    df = state["df"].copy()
    
    if df.isnull().sum().sum() == 0:
        return "No missing values found", df
    
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
    
    state["df"] = df
    return f"✅ Missing data handled! Remaining nulls: {df.isnull().sum().sum()}", df

def remove_duplicates():
    if state["df"] is None:
        return "⚠️ Please upload data first", None
    
    df = state["df"].copy()
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    state["df"] = df
    return f"✅ Removed {before - after} duplicate rows", df

def remove_outliers(column):
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
    return f"✅ Removed {before - after} outlier rows from {column}", df

def scale_data(method):
    if state["df"] is None:
        return "⚠️ Please upload data first", None
    
    df = state["df"].copy()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    if not numeric_cols:
        return "No numeric columns to scale", df
    
    if method == "Standardization":
        df[numeric_cols] = StandardScaler().fit_transform(df[numeric_cols])
    elif method == "Min-Max Normalization":
        df[numeric_cols] = MinMaxScaler().fit_transform(df[numeric_cols])
    
    state["df"] = df
    return f"✅ {method} applied to {len(numeric_cols)} numeric columns", df

# -----------------------------
# 3️⃣ EDA
# -----------------------------
def get_statistics():
    if state["df"] is None:
        return "⚠️ Please upload data first"
    
    df = state["df"]
    stats = df.describe().to_html()
    corr = df.corr().to_html() if len(df.select_dtypes(include=np.number).columns) > 1 else "No numeric columns for correlation"
    return f"Descriptive Statistics:\n{stats}\n\nCorrelation Matrix:\n{corr}"

def plot_histogram(column):
    if state["df"] is None or not column:
        return None
    
    df = state["df"]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df[column], kde=True, ax=ax)
    ax.set_title(f"Histogram of {column}")
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
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax, fmt=".2f")
    ax.set_title("Correlation Heatmap")
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

# -----------------------------
# 4️⃣ Feature Engineering
# -----------------------------
def encode_categorical(column, method):
    if state["df"] is None or not column:
        return "⚠️ Please upload data and select a column", None
    
    df = state["df"].copy()
    
    if method == "One-Hot":
        df = pd.get_dummies(df, columns=[column], drop_first=True)
    elif method == "Label":
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
    elif method == "Frequency":
        freq = df[column].value_counts()
        df[column] = df[column].map(freq)
    
    state["df"] = df
    return f"✅ {method} encoding applied to {column}", df

def create_interaction_feature(col1, col2):
    if state["df"] is None or not col1 or not col2:
        return "⚠️ Please select two columns", None
    
    df = state["df"].copy()
    df[f"{col1}_x_{col2}"] = df[col1] * df[col2]
    state["df"] = df
    return f"✅ Created interaction feature: {col1}_x_{col2}", df

def apply_log_transform(column):
    if state["df"] is None or not column:
        return "⚠️ Please select a column", None
    
    df = state["df"].copy()
    df[f"{column}_log"] = np.log1p(df[column])
    state["df"] = df
    return f"✅ Log transform applied to {column}", df

# -----------------------------
# 5️⃣ Model Building
# -----------------------------
def split_data(target_col, test_size):
    if state["df"] is None or not target_col:
        return "⚠️ Please upload data and select target column"
    
    df = state["df"]
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Handle non-numeric columns
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    X = X[numeric_cols]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size/100, random_state=42
    )
    
    state["X_train"] = X_train
    state["X_test"] = X_test
    state["y_train"] = y_train
    state["y_test"] = y_test
    
    return f"✅ Data split: {len(X_train)} train, {len(X_test)} test rows"

def train_model(model_type, task_type):
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
            r2 = r2_score(y_test, y_pred)
            
            state["model"] = model
            return f"✅ Model trained!\n\nMSE: {mse:.4f}\nR² Score: {r2:.4f}"
        
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
            return f"✅ Model trained!\n\nAccuracy: {acc:.4f}\n\nClassification Report:\n{report}"
    
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
    
    return f"✅ KMeans clustering completed with {n_clusters} clusters", df

def save_model():
    if state["model"] is None:
        return "⚠️ Train a model first"
    
    joblib.dump(state["model"], "trained_model.pkl")
    return "✅ Model saved as trained_model.pkl"

def download_data():
    if state["df"] is None:
        return None
    
    return state["df"]

# -----------------------------
# Gradio Interface
# -----------------------------
def get_numeric_columns():
    if state["df"] is not None:
        return state["df"].select_dtypes(include=np.number).columns.tolist()
    return []

def get_categorical_columns():
    if state["df"] is not None:
        return state["df"].select_dtypes(include="object").columns.tolist()
    return []

def get_all_columns():
    if state["df"] is not None:
        return state["df"].columns.tolist()
    return []

# Create Gradio interface with tabs
with gr.Blocks(title="End-to-End ML Pipeline", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 End-to-End ML Pipeline with Gradio")
    gr.Markdown("A comprehensive machine learning pipeline for data processing, analysis, and modeling")
    
    with gr.Tabs():
        # Tab 1: Data Upload
        with gr.Tab("📂 Data Upload"):
            with gr.Row():
                file_input = gr.File(label="Upload Dataset (CSV, Excel, JSON)")
            upload_btn = gr.Button("Upload Data", variant="primary")
            upload_output = gr.Textbox(label="Status", lines=3)
            data_preview = gr.Dataframe(label="Data Preview")
            
            upload_btn.click(upload_data, inputs=[file_input], outputs=[upload_output, data_preview])
        
        # Tab 2: Data Cleaning
        with gr.Tab("🧹 Data Cleaning"):
            gr.Markdown("### Handle Missing Data")
            with gr.Row():
                missing_action = gr.Dropdown(
                    ["Drop Rows", "Fill with Mean", "Fill with Median", "Fill with Mode"],
                    label="Missing Data Action"
                )
                missing_btn = gr.Button("Apply")
            missing_output = gr.Textbox(label="Status")
            missing_data = gr.Dataframe(label="Updated Data")
            missing_btn.click(handle_missing_data, inputs=[missing_action], outputs=[missing_output, missing_data])
            
            gr.Markdown("### Remove Duplicates")
            dup_btn = gr.Button("Remove Duplicates")
            dup_output = gr.Textbox(label="Status")
            dup_data = gr.Dataframe(label="Updated Data")
            dup_btn.click(remove_duplicates, outputs=[dup_output, dup_data])
            
            gr.Markdown("### Remove Outliers")
            with gr.Row():
                outlier_col = gr.Textbox(label="Column Name", placeholder="Enter numeric column name")
                outlier_btn = gr.Button("Remove Outliers")
            outlier_output = gr.Textbox(label="Status")
            outlier_data = gr.Dataframe(label="Updated Data")
            outlier_btn.click(remove_outliers, inputs=[outlier_col], outputs=[outlier_output, outlier_data])
            
            gr.Markdown("### Scale Data")
            with gr.Row():
                scale_method = gr.Dropdown(["Standardization", "Min-Max Normalization"], label="Scaling Method")
                scale_btn = gr.Button("Apply Scaling")
            scale_output = gr.Textbox(label="Status")
            scale_data_out = gr.Dataframe(label="Updated Data")
            scale_btn.click(scale_data, inputs=[scale_method], outputs=[scale_output, scale_data_out])
        
        # Tab 3: EDA
        with gr.Tab("📈 Exploratory Data Analysis"):
            gr.Markdown("### Descriptive Statistics")
            stats_btn = gr.Button("Show Statistics")
            stats_output = gr.HTML(label="Statistics")
            stats_btn.click(get_statistics, outputs=[stats_output])
            
            gr.Markdown("### Visualizations")
            with gr.Row():
                hist_col = gr.Textbox(label="Column for Histogram", placeholder="Enter numeric column")
                hist_btn = gr.Button("Plot Histogram")
            hist_plot = gr.Plot(label="Histogram")
            hist_btn.click(plot_histogram, inputs=[hist_col], outputs=[hist_plot])
            
            with gr.Row():
                box_col = gr.Textbox(label="Column for Boxplot", placeholder="Enter numeric column")
                box_btn = gr.Button("Plot Boxplot")
            box_plot = gr.Plot(label="Boxplot")
            box_btn.click(plot_boxplot, inputs=[box_col], outputs=[box_plot])
            
            gr.Markdown("### Correlation Heatmap")
            heatmap_btn = gr.Button("Generate Heatmap")
            heatmap_plot = gr.Plot(label="Correlation Heatmap")
            heatmap_btn.click(plot_correlation_heatmap, outputs=[heatmap_plot])
            
            gr.Markdown("### Scatter Plot")
            with gr.Row():
                scatter_x = gr.Textbox(label="X Column", placeholder="Enter column name")
                scatter_y = gr.Textbox(label="Y Column", placeholder="Enter column name")
                scatter_btn = gr.Button("Plot Scatter")
            scatter_plot = gr.Plot(label="Scatter Plot")
            scatter_btn.click(plot_scatter, inputs=[scatter_x, scatter_y], outputs=[scatter_plot])
        
        # Tab 4: Feature Engineering
        with gr.Tab("⚙️ Feature Engineering"):
            gr.Markdown("### Encode Categorical Variables")
            with gr.Row():
                encode_col = gr.Textbox(label="Column Name", placeholder="Enter categorical column")
                encode_method = gr.Dropdown(["One-Hot", "Label", "Frequency"], label="Encoding Method")
                encode_btn = gr.Button("Apply Encoding")
            encode_output = gr.Textbox(label="Status")
            encode_data = gr.Dataframe(label="Updated Data")
            encode_btn.click(encode_categorical, inputs=[encode_col, encode_method], outputs=[encode_output, encode_data])
            
            gr.Markdown("### Create Interaction Features")
            with gr.Row():
                inter_col1 = gr.Textbox(label="Column 1", placeholder="Enter numeric column")
                inter_col2 = gr.Textbox(label="Column 2", placeholder="Enter numeric column")
                inter_btn = gr.Button("Create Interaction")
            inter_output = gr.Textbox(label="Status")
            inter_data = gr.Dataframe(label="Updated Data")
            inter_btn.click(create_interaction_feature, inputs=[inter_col1, inter_col2], outputs=[inter_output, inter_data])
            
            gr.Markdown("### Log Transformation")
            with gr.Row():
                log_col = gr.Textbox(label="Column Name", placeholder="Enter numeric column")
                log_btn = gr.Button("Apply Log Transform")
            log_output = gr.Textbox(label="Status")
            log_data = gr.Dataframe(label="Updated Data")
            log_btn.click(apply_log_transform, inputs=[log_col], outputs=[log_output, log_data])
        
        # Tab 5: Model Building
        with gr.Tab("🤖 Model Building"):
            gr.Markdown("### Supervised Learning")
            with gr.Row():
                target_col = gr.Textbox(label="Target Column", placeholder="Enter target column name")
                test_size = gr.Slider(10, 50, value=20, step=5, label="Test Size (%)")
                split_btn = gr.Button("Split Data")
            split_output = gr.Textbox(label="Status")
            split_btn.click(split_data, inputs=[target_col, test_size], outputs=[split_output])
            
            with gr.Row():
                model_type = gr.Dropdown(
                    ["Linear Regression", "Random Forest Regressor", "Logistic Regression", 
                     "Decision Tree", "Random Forest", "SVM", "KNN"],
                    label="Model Type"
                )
                task_type = gr.Radio(["Regression", "Classification"], label="Task Type", value="Classification")
                train_btn = gr.Button("Train Model", variant="primary")
            train_output = gr.Textbox(label="Training Results", lines=10)
            train_btn.click(train_model, inputs=[model_type, task_type], outputs=[train_output])
            
            gr.Markdown("### Unsupervised Learning - KMeans Clustering")
            with gr.Row():
                n_clusters = gr.Slider(2, 10, value=3, step=1, label="Number of Clusters")
                cluster_btn = gr.Button("Run Clustering")
            cluster_output = gr.Textbox(label="Status")
            cluster_data = gr.Dataframe(label="Data with Clusters")
            cluster_btn.click(train_clustering, inputs=[n_clusters], outputs=[cluster_output, cluster_data])
            
            gr.Markdown("### Save Model")
            save_btn = gr.Button("Save Model")
            save_output = gr.Textbox(label="Status")
            save_btn.click(save_model, outputs=[save_output])
        
        # Tab 6: Download Results
        with gr.Tab("📄 Download Results"):
            gr.Markdown("### Download Processed Data")
            download_btn = gr.Button("Prepare Download")
            download_file = gr.File(label="Download CSV")
            download_btn.click(download_data, outputs=[download_file])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
