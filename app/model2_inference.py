# app/model2_inference.py
import torch
import cv2
import numpy as np
from PIL import Image
# from torchvision.ops import nms
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model2 = torch.jit.load("app/models/model2.pt", map_location=device)
model2.eval()

def preprocess_image(image):
    start = time.time()
    image = np.array(image)
    image = cv2.resize(image, (4000, 3000))
    end = time.time()
    print(f"🧼 Preprocessing time: {(end - start)*1000:.2f} ms")
    return image

def create_patches(image: np.ndarray, tile_size: int = 640) -> np.ndarray:
    start = time.time()
    patches = []
    for i in range(0, image.shape[0], tile_size):
        for j in range(0, image.shape[1], tile_size):
            patch = image[i:i+tile_size, j:j+tile_size]
            if patch.shape[:2] == (tile_size, tile_size):
                patches.append(patch)
    patches = np.array(patches)
    end = time.time()
    print(f"📦 Patch creation time: {(end - start)*1000:.2f} ms, total patches: {len(patches)}")
    return patches
    
    
def run_inference_batched(image, conf_threshold, model2):
    overall_start = time.time()

    pad_height = (640 - image.shape[0] % 640) % 640
    pad_width = (640 - image.shape[1] % 640) % 640
    padded_image = cv2.copyMakeBorder(image, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=0)

    patches = create_patches(padded_image, 640)

    # Batch prep
    start = time.time()
    batch_np = patches.transpose(0, 3, 1, 2) / 255.0  # [N, H, W, C] → [N, C, H, W]
    batch_np = batch_np.astype(np.float32)
    batch_tensor = torch.from_numpy(batch_np).to(device)
    end = time.time()
    print(f"📦 Batch prep time: {(end - start)*1000:.2f} ms")

    # Model inference
    start = time.time()
    with torch.no_grad():
        outputs = model2(batch_tensor)
    end = time.time()
    print(f"🧠 Model inference time (GPU): {(end - start)*1000:.2f} ms")

    # Post-processing
    start = time.time()
    no_of_detections = 0
    for output in outputs:
        output = output.cpu().numpy()  # move back to CPU
        boxes = output[:4].T
        confidences = output[4]

        valid_indices = confidences > conf_threshold
        filtered_boxes = boxes[valid_indices]
        filtered_confidences = confidences[valid_indices]

        all_boxes = []
        for box in filtered_boxes:
            x_center, y_center, width, height = box
            x_min = x_center - width / 2
            y_min = y_center - height / 2
            x_max = x_center + width / 2
            y_max = y_center + height / 2
            all_boxes.append([int(x_min), int(y_min), int(x_max), int(y_max)])

        indices = cv2.dnn.NMSBoxes(all_boxes, filtered_confidences.tolist(), 0.5, 0.5)
        final_boxes = [all_boxes[i] for i in indices.flatten()] if len(indices) > 0 else []

        no_of_detections += len(final_boxes)
    end = time.time()
    print(f"📊 Post-processing time: {(end - start)*1000:.2f} ms")

    overall_end = time.time()
    print(f"⏱️ Total end-to-end time: {(overall_end - overall_start)*1000:.2f} ms\n")

    return "MILD DR" if no_of_detections > 4 else "NO DR"

def model2_inference(image):
    image = preprocess_image(image)
    return run_inference_batched(image, conf_threshold=0.5, model2=model2)


# import cv2
# import numpy as np
# from PIL import Image
# import io
# import tritonclient.http as httpclient
# import tritonclient.grpc as grpcclient

# import time

# TRITON_SERVER_URL = "15.207.42.250:8314"
# def preprocess_image(image):
#     image = np.array(image)
#     image = cv2.resize(image, (4000, 3000))
#     return image

# def create_patches(image: np.ndarray, tile_size: int = 640) -> np.ndarray:
#     patches = []
#     for i in range(0, image.shape[0], tile_size):
#         for j in range(0, image.shape[1], tile_size):
#             patch = image[i:i+tile_size, j:j+tile_size]
#             if patch.shape[0] == tile_size and patch.shape[1] == tile_size:
#                 patches.append(patch)
#     return np.array(patches)

# def run_inference_batched(image, conf_threshold, triton_client):
#     pad_height = (640 - image.shape[0] % 640) % 640
#     pad_width = (640 - image.shape[1] % 640) % 640
#     padded_image = cv2.copyMakeBorder(image, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=0)

#     patches = create_patches(padded_image, 640)

#     # Normalize and prepare batch input
#     batch = []
#     for patch in patches:
#         image_data = patch / 255.0
#         image_data = np.transpose(image_data, (2, 0, 1))  # HWC to CHW
#         batch.append(image_data)

#     batch = np.stack(batch).astype(np.float32)  # Shape: [N, 3, 640, 640]

#     # Triton input/output config
#     # inputs = httpclient.InferInput("images", batch.shape, "FP32")
#     inputs = grpcclient.InferInput("images", batch.shape, "FP32")
#     inputs.set_data_from_numpy(batch)
#     # outputs = httpclient.InferRequestedOutput("output0")
#     outputs = grpcclient.InferRequestedOutput("output0")
#     start = time.time()
#     results = triton_client.infer(model_name="model2", inputs=[inputs], outputs=[outputs])
#     end = time.time()
#     print(f"⚡ Batched model2 inference time: {(end - start) * 1000:.2f} ms")
#     output = results.as_numpy("output0")  # Shape: [N, 85] or similar
#     no_of_detections = 0
#     cal_start = time.time()

#     for i in range(len(patches)):
#         output_i = output[i]
#         boxes = output_i[:4].T
#         confidences = output_i[4]

#         valid_indices = confidences > conf_threshold
#         filtered_boxes = boxes[valid_indices]
#         filtered_confidences = confidences[valid_indices]

#         all_boxes = []
#         for box in filtered_boxes:
#             x_center, y_center, width, height = box
#             x_min = x_center - width / 2
#             y_min = y_center - height / 2
#             x_max = x_center + width / 2
#             y_max = y_center + height / 2
#             all_boxes.append([int(x_min), int(y_min), int(x_max), int(y_max)])

#         indices = cv2.dnn.NMSBoxes(all_boxes, filtered_confidences.tolist(), 0.5, 0.5)
#         final_boxes = [all_boxes[i] for i in indices.flatten()] if len(indices) > 0 else []

#         no_of_detections += len(final_boxes)
#         cal_end = time.time()
#         print(f"⚡ Batched calculation time after model2 inference time: {(cal_end - cal_start) * 1000:.2f} ms")
#     return "MILD DR" if no_of_detections > 4 else "NO DR"

# def model2_inference(triton_client, image):
#     image = preprocess_image(image)
#     return run_inference_batched(image, conf_threshold=0.5, triton_client=triton_client)

