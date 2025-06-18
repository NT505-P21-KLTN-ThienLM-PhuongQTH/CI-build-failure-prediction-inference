# Stage 1: Build stage
FROM python:3.9 AS builder

# Set working directory
WORKDIR /app

# Install build dependencies (cần cho một số thư viện như pandas, numpy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements to leverage Docker cache
COPY requirements.txt .

# Install dependencies into a temporary directory
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime stage
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy installed dependencies from builder stage
COPY --from=builder /root/.local /root/.local

# Copy only necessary files and directories
COPY api/ api/
COPY src/ src/
COPY config/ config/
COPY main.py .

# Ensure scripts in PATH
ENV PATH=/root/.local/bin:$PATH

# Install Uvicorn if not in requirements.txt
RUN pip install --no-cache-dir uvicorn

# Expose port for FastAPI
EXPOSE 8000

# Command to run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]