from fastapi import FastAPI, File, UploadFile
from app.inference import predict
import asyncio
from datetime import datetime

app = FastAPI()


@app.post("/predict")
async def get_inference(left_eye: UploadFile = File(...), right_eye: UploadFile = File(...)):
    left_image_bytes = await left_eye.read()
    right_image_bytes = await right_eye.read()

    # Run both predictions concurrently using threads
    # left_task = asyncio.to_thread(predict, left_image_bytes)
    # right_task = asyncio.to_thread(predict, right_image_bytes)
    # left_result, right_result = await asyncio.gather(left_task, right_task)

    left_result, right_result = await asyncio.gather(
            predict(left_image_bytes),
            predict(right_image_bytes)
        )

    return {
        "left_eye": left_result,
        "right_eye": right_result
    }
