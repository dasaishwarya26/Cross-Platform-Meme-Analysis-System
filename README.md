# System Components

## 1. FastAPI Backend

Located in: `backend/main.py`  
Handles real-time computation of:

- `/semantic_drift`
- `/influence_matrix`
- `/temporal_cooccurrence`
- `/explain` (LLM explanation)

Core technologies:

- Python 3.11
- FastAPI + SQLAlchemy
- PostgreSQL + TimescaleDB + dblink
- Sentence-BERT (`all-MiniLM-L6-v2`)
- OpenAI ChatGPT API (`gpt-4o-mini`)

Backend server (team VM):  
```

[http://128.226.29.112:8000](http://128.226.29.112:8000)

```

API docs:  
```

[http://128.226.29.112:8000/docs](http://128.226.29.112:8000/docs)

````

---

## 2. Streamlit Dashboard

Located in: `dashboard/app.py` + `dashboard/pages/`

Implements three major tools:

- **Semantic Drift Heatmap**
- **Influence Matrix Viewer**
- **Temporal Co-occurrence Viewer**

Each page supports:

✔ Live parameter selection  
✔ Plotly visualization  
✔ “Explain” button powered by `/explain` endpoint  

Run dashboard:

```bash
uv run streamlit run dashboard/app.py
````

---

# How to Run

## **1. SSH into VM (with port forwarding)**

```bash
ssh -L 8000:localhost:8000 -L 8501:localhost:8501 <user>@128.226.29.112
```

---

## **2. Open Dashboard**

```
http://localhost:8501
```

Backend and Dashboard are continuously running in the background.
---

## **3. Start Streamlit Dashboard**

1. click on button "Compute <something: graph name>"
2. click on button "Explain <something: graph name>(Generate LLM)"
   > Carefully click on this button, we will be charged by OpenAI. 
