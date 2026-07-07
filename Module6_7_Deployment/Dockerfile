# House-price valuation API (Module 6).
#
# IMPORTANT: build from the REPOSITORY ROOT so the image can include the
# Module 4-5 inference code, the saved model, and the training schema:
#
#   docker build -f Module6_7_Deployment/Dockerfile -t valuation-api .
#   docker run -p 8000:8000 valuation-api
#
# The container serves the FastAPI app. To serve the Streamlit demo instead,
# override the command (see README).

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# libgomp1 is required by scikit-learn / xgboost runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first for better layer caching.
COPY Module6_7_Deployment/requirements.txt /app/Module6_7_Deployment/requirements.txt
RUN pip install -r /app/Module6_7_Deployment/requirements.txt

# Copy only what inference + monitoring need (keeps the image small).
COPY Module4_5_Modeling/src /app/Module4_5_Modeling/src
COPY Module4_5_Modeling/models /app/Module4_5_Modeling/models
COPY Module4_5_Modeling/outputs /app/Module4_5_Modeling/outputs
COPY CleanData/data-v2/processed /app/CleanData/data-v2/processed
COPY Module6_7_Deployment /app/Module6_7_Deployment

WORKDIR /app/Module6_7_Deployment

ENV PREDICTION_LOG_PATH=/app/Module6_7_Deployment/logs/predictions.jsonl

EXPOSE 8000

# Render / HF Spaces inject $PORT; default to 8000 for local runs.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
