FROM python:3.10-slim

# Set working directory 
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose container ports:
EXPOSE 7860 8000 9100

# Run the app
CMD ["python", "app.py"]

