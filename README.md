
## 📦 Diabetic Retinopathy Detection API

This FastAPI-based project provides an AI-powered backend to detect and classify **Diabetic Retinopathy (DR)** from retina images. The app is containerized with Docker and exposes an API endpoint that accepts an image and returns multi-stage classification results along with optional explainability (XAI) visuals.

---

### 🚀 Features

- 🧠 **3-model architecture**:
  - `Model1`: Primary classification (`REF` vs `NON-REF`)
  - `Model2`: Further classification of `NON-REF` (e.g., `NO DR`, `MILD DR`)
  - `Model3`: Further classification of `REF` (e.g., `MODERATE`, `SEVERE`, `PROLIFERATIVE`)
- 🔄 Dynamic inference path based on primary model output
- 📈 XAI Grad-CAM support when the output is `REF`
- 🐳 Fully Dockerized FastAPI application
- 🔗 Integrated async API call to a separate XAI service

---

### 📂 Folder Structure

```
dr_fastapi/
├── app/
│   ├── inference.py              # Core prediction logic
│   ├── main.py                   # FastAPI app entry point
│   ├── model1_inference.py       # REF / NON-REF classification
│   ├── model2_inference.py       # NON-REF sub-classification
│   ├── model3_inference.py       # REF sub-classification
│   └── models/
│       ├── model1.pt
│       ├── model2.pt
│       └── model3.pt
├── Dockerfile                    # Docker image definition
├── requirements.txt              # Python dependencies
```

---

### 🔌 API Usage

#### `POST /predict`

Send a retina image to the API and receive a JSON response with DR classification and optional XAI image.

##### 📤 Request
- **Method**: `POST`
- **URL**: `http://<your-host>:<port>/predict`
- **Body (form-data)**:
  - `file`: Retina image file (`.png`, `.jpg`, etc.)

##### 📥 Response (JSON)
```json
{
  "primary_classification": {"class_name": "REF"},
  "sub_classes": {"class_name": "SEVERE DR"},
  "xai_results": {"image": "base64_encoded_gradcam.png"},
  "errors": null
}
```

---

### 🐳 Running with Docker

#### 1. **Build the image**
```bash
docker build -t dr_fastapi .
```

#### 2. **Run the container**
```bash
docker run -d -p 8000:8000 dr_fastapi
```

Then access the API at:  
📍 `http://localhost:8000/predict`  
📘 Swagger docs: `http://localhost:8000/docs`

---

### 📦 Installation (Without Docker)

If you want to run locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

### 📌 Notes

- The `xai_results` field is only populated when the primary classification is `REF`.
- The XAI Grad-CAM image is fetched via an internal API (`http://15.207.42.250:8316/xai_predict`).
- Large `.pt` model files are tracked and loaded locally.

---

### 👨‍🔬 Author    

**Sagar Sau**  
🔗 [GitHub: @sagar-07-ai](https://github.com/sagar-07-ai)

---

Let me know if you'd like a badge section (e.g., Python version, Docker, etc.) or CI/CD setup info!
