# IAV_AI_Task
# Signal Plotting REST API 🚗📊

This REST API allows you to upload an MDF4 vehicle signal file (`.mf4`) along with a natural language query (e.g., "plot a histogram of battery voltage") and returns a generated plot with summary statistics.

---

## 🛠️ Requirements

Install the dependencies using:

```bash
pip install -r requirements.txt

python -m uvicorn main:app --reload
