# IAV_AI_Task
Detailed information about the training of LLM is given in the documentation provided.
You can access the model weights via this link: https://drive.google.com/drive/folders/1RlUcDL_g-0y2cpkulMIoqvngBu8p6SYW?usp=sharing
# Signal Plotting REST API

This REST API allows you to upload an MDF4 vehicle signal file (`.mf4`) along with a natural language query (e.g., "plot a histogram of battery voltage") and returns a generated plot with summary statistics.

---

## 🛠️ Requirements

Install the dependencies using:

```bash
pip install -r requirements.txt
python -m uvicorn main:app --reload
```
Open your browser and go to: http://127.0.0.1:8000/docs
This opens the Swagger UI with a form-based interface.
Upload your .mf4 data file in the file field and type your query.
<img width="1786" height="855" alt="image" src="https://github.com/user-attachments/assets/9bda200a-cb4a-40d1-a2a8-50fbcb48d627" />
After execution, it will generate the summary statistics and the appropriate plot. Afterwards you can nevigate to the filename under uploads folder to get the plot.

<img width="1378" height="870" alt="image" src="https://github.com/user-attachments/assets/33e182a4-72cf-4664-b18d-6ed79493ed0f" />
