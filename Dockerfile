FROM python:3.10-slim

WORKDIR /code

# Install system dependencies, Node.js, and git-lfs
RUN apt-get update && apt-get install -y \
    build-essential \
    nodejs \
    npm \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Initialize git-lfs
RUN git lfs install

# Clone the repository and fetch LFS files using authentication
ARG GIT_ACCESS_TOKEN
RUN git clone https://oauth2:${GIT_ACCESS_TOKEN}@github.com/resamimi/diabetes_dashboard.git /code && \
    cd /code && \
    git lfs pull

# Install frontend dependencies
WORKDIR /code/static/react/chat-interface
RUN npm install

# Build frontend
RUN npm run build

# Return to main directory
WORKDIR /code

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Preload the model and data
RUN python3 preload.py

EXPOSE 8080

# Use gunicorn to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:8080", \
"--timeout", "300", \
"--workers", "1", \
"--threads", "4", \
"--preload", \
"--max-requests", "1000", \
"--max-requests-jitter", "50", \
"flask_app:app"]