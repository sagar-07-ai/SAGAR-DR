# app/inference.py
import io
import asyncio
import requests
from PIL import Image
from app.model1_inference import model1_inference
from app.model2_inference import model2_inference
from app.model3_inference import model3_inference
import httpx

async def call_xai_api_async(image_bytes):
    url = "http://15.207.42.250:8316/xai_predict"
    async with httpx.AsyncClient() as client:
        try:
            files = {"file": ("input_image.png", image_bytes, "image/png")}
            response = await client.post(url, files=files)
            response.raise_for_status()
            return response.text.strip('"')
        except httpx.RequestError:
            return None

async def predict(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    errors = {}

    # Step 1: Run model1 in thread
    primary_class = await asyncio.to_thread(model1_inference, image)

    if primary_class == "NON-REF":
        secondary_class = await asyncio.to_thread(model2_inference, image)
        xai_image = None

    elif primary_class == "REF":
        xai_image =await call_xai_api_async(image_bytes)
        model3_output = await asyncio.to_thread(model3_inference, image)
        secondary_class = model3_output.get("class_name")

    return {
        "primary_classification": {"class_name": primary_class},
        "sub_classes": {"class_name": secondary_class},
        "xai_results": {"image": xai_image},
        "errors": errors if errors else None
    }





# from torchvision import transforms
# from PIL import Image
# import io
# import cv2
# import torch 
# import asyncio
# import requests
# import numpy as np
# import tritonclient.http as httpclient
# import tritonclient.grpc as grpcclient
# from app.model1_inference import model1_inference
# from app.model2_inference import model2_inference
# from app.model3_inference import model3_inference

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Triton Inference Server URL
# # TRITON_SERVER_URL = "10.10.110.24:8313" #http
# TRITON_SERVER_URL = "10.10.110.24:8314" #grpc
# # triton_client = httpclient.InferenceServerClient(url=TRITON_SERVER_URL, verbose=False)
# triton_client = grpcclient.InferenceServerClient(url=TRITON_SERVER_URL, verbose=False)



# # app/inference.py
# async def predict(image_bytes):
#     triton_client = grpcclient.InferenceServerClient(url=TRITON_SERVER_URL, verbose=False)
#     image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
#     errors = {}

#     # Step 1: Run model1
#     primary_class = await asyncio.to_thread(model1_inference, triton_client, image)

#     # Step 2: Based on model1, run model2 or model3
#     if primary_class == "NON-REF":
#         # model2 is CPU/GPU heavy - make async using to_thread
#         secondary_class = await asyncio.to_thread(model2_inference, triton_client, image)
#         xai_image = None

#     elif primary_class == "REF":
#         # model3_inference already handles XAI + model3 in parallel
#         model3_output = await model3_inference(triton_client, image_bytes)
#         secondary_class = model3_output.get("class_name")
#         xai_image = model3_output.get("xai_image")
#     else:
#         secondary_class = "Unknown"
#         xai_image = None
#         errors["primary_classification"] = f"Unexpected class: {primary_class}"

#     if xai_image is None and primary_class == "REF":
#         errors["xai_image"] = "Error generating Explainable AI image"

#     return {
#         "primary_classification": {"class_name": primary_class},
#         "sub_classes": {"class_name": secondary_class},
#         "xai_results": {"image": xai_image},
#         "errors": errors if errors else None
#     }