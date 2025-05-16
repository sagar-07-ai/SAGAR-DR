FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
 && rm -rf /var/lib/apt/lists/*

# Environment
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install Python deps
RUN pip install --upgrade pip && pip install python-multipart
RUN pip install httpx

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8317"]


# FROM python:3.11
# RUN apt-get update && apt-get install -y \
#     libgl1-mesa-glx \
#     libglib2.0-0
# RUN pip install python-multipart

# WORKDIR /app

# COPY requirements.txt /app/
# RUN pip install -r requirements.txt

# COPY . /app/

# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8317"]
