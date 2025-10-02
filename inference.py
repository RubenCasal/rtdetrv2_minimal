import torch
import requests
from PIl import Image, ImageDraw
from transformers import AutoImageProcessor, AutoModelForObjectDetection

device = "cuda"


url = "https://images.pexels.com/photos/8413299/pexels-photo-8413299.jpeg?auto=compress&cs=tinysrgb&w=630&h=375&dpr=2"
image = Image.open(requests.get(url, stream=True).raw)


CKPT = "./ruta_checkpoint"

image_processor = AutoImageProcessor.from_pretrained(CKPT)
model = AutoModelForObjectDetection.from_pretrained(CKPT)


model = model.to(device)

inputs = image_processor(images=[image], return_tensors="pt")
inputs = inputs.to(device)

with torch.no_grad():
    outputs = model(**inputs)

target_sizes = torch.tensor([image.size[::-1]])

result = image_processor.post_process_object_detection(outputs, threshold=0.4, target_sizes=target_sizes)[0]

for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
        f"Detected {model.config.id2label[label.item()]} with confidence "
        f"{round(score.item(), 3)} at location {box}"
    )