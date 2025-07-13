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
    timestamps = df["timestamp"]

    summary = {
        "mean": round(data.mean(), 3),
        "median": round(data.median(), 3),
        "std": round(data.std(), 3),
        "min": round(data.min(), 3),
        "max": round(data.max(), 3)
    }

    filename = f"{uuid.uuid4().hex}.png"
    filepath = os.path.join(UPLOAD_DIR, filename)

    plt.figure(figsize=(10, 6))
    method = method.lower()

    if method == "histogram":
        plt.hist(data, bins=50, color='skyblue', edgecolor='black')
        plt.xlabel(attribute)
    elif method == "line plot":
        plt.plot(timestamps, data, color='blue')
        plt.xlabel("Time (s)")
    elif method == "bar chart":
        # Bar chart with timestamps as x-axis may be unreadable if there are many points
        # So either plot limited values or keep as index-based bar chart
        plt.bar(range(len(data)), data, color='orange')
        plt.xlabel("Sample Index")
    elif method == "scatter plot":
        plt.scatter(timestamps, data, alpha=0.6)
        plt.xlabel("Time (s)")
    elif method == "box plot":
        plt.boxplot(data)
        plt.xlabel(attribute)
    else:
        return None, None

    plt.title(f"{method.title()} of {attribute}")
    plt.ylabel(attribute)
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
