FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose port 7860 for Hugging Face
EXPOSE 7860

# Run inference first, then start the server from the new folder location
CMD python inference.py && uvicorn server.app:app --host 0.0.0.0 --port 7860
