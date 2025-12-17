import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------- Data Loader --------------------
def load_data(file):
    if file is None:
        return None, "No file uploaded"

    name = file.name.lower()
    try:
        if name.endswith('.csv'):
            df = pd.read_csv(file)
        elif name.endswith('.json'):
            df = pd.read_json(file)
        elif name.endswith('.xlsx'):
            df = pd.read_excel(file)
        else:
            return None, "Unsupported file format"
    except Exception as e:
        return None, str(e)

    return df, "File loaded successfully"

# -------------------- EDA Functions --------------------
def basic_info(df):
    if df is None:
        return ""
    info = {
        "Rows": df.shape[0],
        "Columns": df.shape[1]
    }
    return pd.DataFrame(info, index=[0])


def missing_values(df):
    if df is None:
        return ""
    return df.isnull().sum().reset_index(name="Missing Values")


def describe_data(df):
    if df is None:
        return ""
    return df.describe(include='all')


# -------------------- Visualization --------------------
def plot_histogram(df, column):
    if df is None or column is None:
        return None

    plt.figure()
    df[column].dropna().hist()
    plt.title(f"Histogram of {column}")
    return plt.gcf()


def plot_boxplot(df, column):
    if df is None or column is None:
        return None

    plt.figure()
    df.boxplot(column=column)
    plt.title(f"Boxplot of {column}")
    return plt.gcf()


def correlation_heatmap(df):
    if df is None:
        return None

    num_df = df.select_dtypes(include=np.number)
    if num_df.empty:
        return None

    corr = num_df.corr()
    plt.figure()
    plt.imshow(corr)
    plt.colorbar()
    plt.xticks(range(len(corr)), corr.columns, rotation=90)
    plt.yticks(range(len(corr)), corr.columns)
    plt.title("Correlation Heatmap")
    return plt.gcf()

# -------------------- Gradio UI --------------------
with gr.Blocks(title="EDA App") as app:
    gr.Markdown("# 📊 Exploratory Data Analysis App")

    with gr.Sidebar():
        file_input = gr.File(label="Upload Dataset", file_types=[".csv", ".json", ".xlsx"])
        load_btn = gr.Button("Load Data")
        status = gr.Textbox(label="Status")

    df_state = gr.State()

    with gr.Tab("Overview"):
        info_out = gr.Dataframe(label="Dataset Info")
        missing_out = gr.Dataframe(label="Missing Values")

    with gr.Tab("Statistics"):
        desc_out = gr.Dataframe(label="Descriptive Statistics")

    with gr.Tab("Visualizations"):
        col_dropdown = gr.Dropdown(label="Select Column")
        hist_plot = gr.Plot()
        box_plot = gr.Plot()
        corr_plot = gr.Plot()

    # -------------------- Callbacks --------------------
    def on_load(file):
        df, msg = load_data(file)
        if df is None:
            return None, msg, None, None, None, None
        cols = df.columns.tolist()
        return df, msg, basic_info(df), missing_values(df), cols, describe_data(df)

    load_btn.click(
        on_load,
        inputs=file_input,
        outputs=[df_state, status, info_out, missing_out, col_dropdown, desc_out]
    )

    col_dropdown.change(
        lambda df, col: plot_histogram(df, col),
        inputs=[df_state, col_dropdown],
        outputs=hist_plot
    )

    col_dropdown.change(
        lambda df, col: plot_boxplot(df, col),
        inputs=[df_state, col_dropdown],
        outputs=box_plot
    )

    load_btn.click(
        lambda df: correlation_heatmap(df),
        inputs=df_state,
        outputs=corr_plot
    )

app.launch()
