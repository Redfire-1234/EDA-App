import gradio as gr
import pandas as pd

def load_file(file):
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

        info = f"Rows: {df.shape[0]} | Columns: {df.shape[1]}"
        return info, df.head()

    except Exception as e:
        return f"Error: {str(e)}", None


with gr.Blocks(title="EDA App") as app:
    gr.Markdown("## 📊 EDA App")
    gr.Markdown("Upload CSV, Excel, or JSON files")

    file_input = gr.File(
        label="Upload Dataset",
        file_types=[".csv", ".xlsx", ".json"]
    )

    info_output = gr.Textbox(label="Dataset Info")
    table_output = gr.Dataframe(label="Data Preview")

    file_input.change(
        fn=load_file,
        inputs=file_input,
        outputs=[info_output, table_output]
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)

