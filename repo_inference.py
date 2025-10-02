import requests
import torch

from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection


model_name = "qubvel-hf/detr-resnet-50-finetuned-10k-cppe5"

image_processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForObjectDetection.from_pretrained(model_name)

# Load image for inference
url = "https://images.pexels.com/photos/8413299/pexels-photo-8413299.jpeg?auto=compress&cs=tinysrgb&w=630&h=375&dpr=2"
image = Image.open(requests.get(url, stream=True).raw)


inputs = image_processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# Post process model predictions 
# this include conversion to Pascal VOC format and filtering non confident boxes
width, height = image.size
target_sizes = torch.tensor([height, width]).unsqueeze(0)  # add batch dim
results = image_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
        f"Detected {model.config.id2label[label.item()]} with confidence "
        f"{round(score.item(), 3)} at location {box}"
    )
