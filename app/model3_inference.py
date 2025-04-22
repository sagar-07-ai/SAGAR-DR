# app/model3_inference.py
import torch
from PIL import Image
from torchvision import transforms
import asyncio

# Load model3.pt once
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model3 = torch.jit.load("app/models/model3.pt", map_location=device)
model3.eval()

transform3 = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

async def model3_inference(image):
    image = transform3(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model3(image)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted = torch.argmax(probabilities).item()
        class_map = {0: "MODERATE DR", 1: "SEVERE DR"}
        return {
            "class_name": class_map.get(predicted, "PROLIFERATIVE DR"),
            "xai_image": None  # No API call now
        }



# import io
# import requests
# import numpy as np
# from PIL import Image
# from torchvision import transforms
# import torch
# import tritonclient.http as httpclient
# import tritonclient.grpc as grpcclient
# import asyncio
# import time

# XAI_API_URL = "http://15.207.42.250:8316/xai_predict"

# transform1 = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# def call_xai_api(image_bytes):
#     XAI_API_URL = "http://15.207.42.250:8316/xai_predict"

#     try:
#         file = {"file": ("input_image.png", image_bytes, "image/png")}
#         response = requests.post(XAI_API_URL, files=file)
#         response.raise_for_status()
#         return response.text.strip('"')
#     except requests.exceptions.RequestException:
#         return None

# def run_model3_inference(triton_client, image):
#     image = transform1(image).unsqueeze(0).numpy()
#     # inputs = httpclient.InferInput("input__0", image.shape, "FP32")
#     inputs = grpcclient.InferInput("input__0", image.shape, "FP32")
#     inputs.set_data_from_numpy(image)
#     # outputs = httpclient.InferRequestedOutput("output__0")
#     outputs = grpcclient.InferRequestedOutput("output__0")
    
#     results = triton_client.infer(model_name="model3", inputs=[inputs], outputs=[outputs])
#     output_data = results.as_numpy("output__0")

#     probabilities = np.exp(output_data) / np.sum(np.exp(output_data))
#     secondary_class = np.argmax(probabilities[0])
#     return {0: "MODERATE DR", 1: "SEVERE DR"}.get(secondary_class, "PROLIFERATIVE DR")

# async def model3_inference(triton_client, image_bytes):
#     image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

#     # ✅ Start timing XAI API submission
#     xai_start = time.time()
#     xai_task = asyncio.to_thread(call_xai_api, image_bytes)
#     xai_submit_time = time.time()
#     print(f"Submitted XAI task in {(xai_submit_time - xai_start) * 1000:.2f} ms")

#     # ✅ Time model3 inference (runs immediately)
#     model_start = time.time()
#     class_name = run_model3_inference(triton_client, image)
#     model_end = time.time()
#     print(f"Inference Time for model3 : {(model_end - model_start) * 1000:.2f} ms")

#     # ✅ Await XAI result
#     xai_result_start = time.time()
#     xai_image = await xai_task
#     xai_end = time.time()
#     print(f"XAI API response wait time: {(xai_end - xai_result_start) * 1000:.2f} ms")
#     print(f"Total XAI latency: {(xai_end - xai_start) * 1000:.2f} ms")

#     return {
#         "class_name": class_name,
#         "xai_image": xai_image
#     }
