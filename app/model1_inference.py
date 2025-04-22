# app/model1_inference.py
import torch
from PIL import Image
from torchvision import transforms

# Load model1.pt once
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1 = torch.jit.load("app/models/model1.pt",map_location=device)
model1.eval()

# Define transformation
transform1 = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def model1_inference(image):
    image = transform1(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model1(image)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted = torch.argmax(probabilities).item()
        return "REF" if predicted == 1 else "NON-REF"

# import io
# from PIL import Image
# import numpy as np
# import torch
# from torchvision import transforms
# import tritonclient.http as httpclient
# import tritonclient.grpc as grpcclient


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# transform1 = transforms.Compose([
#     transforms.Resize((300,300)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# def model1_inference(triton_client,image):
#     triton_client = triton_client

#     image = transform1(image).unsqueeze(0).numpy()

#     # inputs = httpclient.InferInput("input__0", image.shape, "FP32")
#     inputs = grpcclient.InferInput("input__0", image.shape, "FP32")
#     inputs.set_data_from_numpy(image)
#     # outputs = httpclient.InferRequestedOutput("output__0")
#     outputs = grpcclient.InferRequestedOutput("output__0")
#     # results = triton_client.infer(model_name="model1", inputs=[inputs], outputs=[outputs])
#     results = triton_client.infer(model_name="model1", inputs=[inputs], outputs=[outputs])
#     output_data = results.as_numpy("output__0")

#     probabilities = np.exp(output_data) / np.sum(np.exp(output_data))
#     primary_class = np.argmax(probabilities[0])

#     return "REF" if primary_class == 1 else "NON-REF"
