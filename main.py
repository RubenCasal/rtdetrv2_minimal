from datasets import Dataset, DatasetDict, load_dataset
from PIL import Image, ImageDraw
import numpy as np
from transformers import AutoImageProcessor, AutoModelForObjectDetection, TrainingArguments, Trainer
import albumentations as A

from model_output import ModelOutput
from CocoDs import CPPE5Dataset
from evaluator import MAPEvaluator
from utils import filter_cppe5_dataset, collate_fn





dataset = load_dataset("cppe-5")
dataset = filter_cppe5_dataset(dataset)
print(dataset.shape)


checkpoint = "PekingU/rtdetr_v2_r50vd"
image_size = 480

if "validation" not in dataset:
    split = dataset["train"].train_test_split(0.15, seed=1337)
    dataset["train"] = split["train"]
    dataset["validation"] = split["test"]


categories = dataset["train"].features["objects"]["category"].feature.names
id2label = {index: x for index, x in enumerate(categories, start=0)}
label2id = {v: k for k, v in id2label.items()}

image_processor = AutoImageProcessor.from_pretrained(
    checkpoint,
    do_resize=True,
    size={"width": image_size, "height": image_size},
    use_fast=True,
)


train_augmentation_and_transform = A.Compose(
    [
        A.Perspective(p=0.1),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.1),
    ],
    bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True, min_area=25, min_width=1, min_height=1),
)

# to make sure boxes are clipped to image size and there is no boxes with area < 1 pixel
validation_transform = A.Compose(
    [A.NoOp()],
    bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True, min_area=1, min_width=1, min_height=1),
)


train_dataset = CPPE5Dataset(dataset["train"], image_processor, transform=train_augmentation_and_transform)
validation_dataset = CPPE5Dataset(dataset["validation"], image_processor, transform=validation_transform)
test_dataset = CPPE5Dataset(dataset["test"], image_processor, transform=validation_transform)


eval_compute_metrics_fn = MAPEvaluator(image_processor=image_processor, threshold=0.01, id2label=id2label)


model = AutoModelForObjectDetection.from_pretrained(
    checkpoint,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

training_args = TrainingArguments(
    output_dir="rtdetr-v2-r50-cppe5-finetune-2",
    num_train_epochs=40,
    max_grad_norm=0.1,
    learning_rate=5e-5,
    warmup_steps=300,
    per_device_train_batch_size=8,
    dataloader_num_workers=2,
    metric_for_best_model="eval_map",
    greater_is_better=True,
    load_best_model_at_end=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    remove_unused_columns=False,
    eval_do_concat_batches=False,
    report_to="tensorboard",  # or "wandb"
)



trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    processing_class=image_processor,
    data_collator=collate_fn,
    compute_metrics=eval_compute_metrics_fn,
)

trainer.train()