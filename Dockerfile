# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# 🔧 Install system dependencies (for Kaleido, Plotly, or others)
# Install system dependencies + Chromium
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libexpat1 wget gnupg ca-certificates chromium \
      chromium-driver && \
    rm -rf /var/lib/apt/lists/*

# Make sure Kaleido knows where to find Chromium
ENV BROWSER_PATH=/usr/bin/chromium

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full app source code
COPY . .

# Expose the port for FastAPI
EXPOSE 8081

# Start the FastAPI app with WebSocket support
CMD ["uvicorn", "main_websocket:app", "--host", "0.0.0.0", "--port", "8081", "--proxy-headers", "--forwarded-allow-ips", "*", "--ws", "auto", "--ws-ping-interval", "30", "--ws-ping-timeout", "180", "--log-level", "info"]
