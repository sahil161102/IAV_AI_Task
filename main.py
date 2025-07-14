# run -> python -m uvicorn main:app --reload
# open the "localhost link"/docs to add the arguments of Rest API 

import os
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import JSONResponse, FileResponse
from asammdf import MDF
from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd
import matplotlib.pyplot as plt
import uuid
import re
import numpy as np
app = FastAPI()

# Load model once
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
model = T5ForConditionalGeneration.from_pretrained("flan-t5-visualization")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
print("v1")
def parse_user_query(user_query: str):
    try:
        input_ids = tokenizer(user_query, return_tensors="pt").input_ids
        output = model.generate(input_ids, max_length=64)
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        return decoded
    except:
        return None

def extract_signal(mdf_path, signal_name):
    try:
        mdf = MDF(mdf_path)
        signal = mdf.get(signal_name)
        df = pd.DataFrame({
            'timestamp': signal.timestamps,
            'value': signal.samples})
        return df
    except Exception as e:
        print(f"[ERROR] Signal '{signal_name}' not found: {e}")
        return None
    
def parse_output_string(s):
    matches = re.findall(r"'(\w+)'\s*:\s*'([^']+)'", s)
    return {k: v for k, v in matches}

def generate_plot_and_stats(df, attribute, method):
    data = df["value"]
    time = df["timestamp"]

    # Compute duration between samples
    if len(time) > 1:
        time_deltas = time.diff().fillna(0)
    else:
        time_deltas = [1] * len(data)  # fallback if only one point

    summary = {
        "mean": round(data.mean(), 3),
        "median": round(data.median(), 3),
        "std": round(data.std(), 3),
        "min": round(data.min(), 3),
        "max": round(data.max(), 3)
    }

    filename = f"{uuid.uuid4().hex}.png"
    filepath = os.path.join("uploads", filename)

    plt.figure(figsize=(10, 6))
    method = method.lower()

    if method == "histogram":
        # Bin edges
        bins = 50
        bin_values, bin_edges = np.histogram(data, bins=bins)

        # Map each value to its bin duration using time_deltas
        bin_indices = np.digitize(data, bin_edges[:-1], right=False)
        duration_per_bin = [0] * bins

        for i in range(len(data)):
            idx = bin_indices[i] - 1
            if 0 <= idx < bins:
                duration_per_bin[idx] += time_deltas.iloc[i]

        # Plot duration vs. bin centers
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        plt.bar(bin_centers, duration_per_bin, width=bin_edges[1]-bin_edges[0], edgecolor='black', color='skyblue')
        plt.ylabel("Duration [s]")

    elif method == "line plot":
        plt.plot(time, data, color='blue')
        plt.xlabel("Time [s]")
        plt.ylabel(attribute)

    elif method == "bar chart":
        data.plot(kind='bar', color='orange')
        plt.ylabel(attribute)

    elif method == "scatter plot":
        plt.scatter(time, data, alpha=0.6)
        plt.xlabel("Time [s]")
        plt.ylabel(attribute)

    elif method == "box plot":
        plt.boxplot(data)
        plt.ylabel(attribute)

    else:
        return None, None

    plt.title(f"{method.title()} of {attribute}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

    return summary, filepath

@app.post("/plot/")
async def plot_from_mdf(file: UploadFile = File(...), query: str = Form(...)):
    filename = f"{uuid.uuid4().hex}_{file.filename}"
    filepath = os.path.join(UPLOAD_DIR, filename)

    with open(filepath, "wb") as f:
        f.write(await file.read())

    # Parse query
    parsed = parse_user_query(query)
    parsed = parse_output_string(parsed)
    if not parsed or "Method" not in parsed or "Attribute" not in parsed:
        return JSONResponse({"error": "Could not interpret query"}, status_code=400)

    df = extract_signal(filepath, parsed["Attribute"])
    if df is None:
        return JSONResponse({"error": f"Attribute '{parsed['Attribute']}' not found"}, status_code=404)

    summary, plot_path = generate_plot_and_stats(df, parsed["Attribute"], parsed["Method"])
    if plot_path is None:
        return JSONResponse({"error": "Unsupported plot method"}, status_code=400)

    return {
        "summary_statistics": summary,
        "plot_image": f"/plot/image/{os.path.basename(plot_path)}"
    }

@app.get("/plot/image/{filename}")
async def get_plot_image(filename: str):
    filepath = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type="image/png")
    return JSONResponse({"error": "Image not found"}, status_code=404)
