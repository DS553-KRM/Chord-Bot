FROM python:3.10-slim

WORKDIR /app

# System deps (node exporter)
RUN apt-get update \
 && apt-get install -y --no-install-recommends prometheus-node-exporter ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App files
COPY . .

# Ensure entrypoint is executable
RUN chmod +x /app/entrypoint.sh

# Env for Gradio inside containers
ENV GRADIO_SERVER_NAME=0.0.0.0

# Expose container-side ports:
EXPOSE 7860 8000 9100

# Start both Node Exporter and the app
ENTRYPOINT ["/app/entrypoint.sh"]

