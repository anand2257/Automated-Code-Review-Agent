FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose port 7860 (This is required by Hugging Face)
EXPOSE 7860

# Run the inference script for the logs, then start the web server to stay alive
CMD python inference.py && uvicorn app:app --host 0.0.0.0 --port 7860
