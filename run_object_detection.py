#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

# /// script
# dependencies = [
#     "transformers @ git+https://github.com/huggingface/transformers.git",
#     "albumentations >= 1.4.16",
#     "timm",
#     "datasets>=4.0",
#     "torchmetrics",
#     "pycocotools",
# ]
# ///

"""Finetuning any 🤗 Transformers model supported by AutoModelForObjectDetection for object detection leveraging the Trainer API."""

import logging
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Optional, Union

import albumentations as A
import numpy as np
import torch
from datasets import load_dataset
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from utils import filter_cppe5_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForObjectDetection,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from transformers.image_processing_utils import BatchFeature
from transformers.image_transforms import center_to_corners_format
from transformers.trainer import EvalPrediction
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version


logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.57.0.dev0")

require_version("datasets>=2.0.0", "To fix: pip install -r examples/pytorch/object-detection/requirements.txt")


@dataclass
class ModelOutput:
    logits: torch.Tensor
    pred_boxes: torch.Tensor


def format_image_annotations_as_coco(
    image_id: str, categories: list[int], areas: list[float], bboxes: list[tuple[float]]
) -> dict:
    """Format one set of image annotations to the COCO format

    Args:
        image_id (str): image id. e.g. "0001"
        categories (list[int]): list of categories/class labels corresponding to provided bounding boxes
        areas (list[float]): list of corresponding areas to provided bounding boxes
        bboxes (list[tuple[float]]): list of bounding boxes provided in COCO format
            ([center_x, center_y, width, height] in absolute coordinates)

    Returns:
        dict: {
            "image_id": image id,
            "annotations": list of formatted annotations
        }
    """
    annotations = []
    for category, area, bbox in zip(categories, areas, bboxes):
        formatted_annotation = {
            "image_id": image_id,
            "category_id": category,
            "iscrowd": 0,
            "area": area,
            "bbox": list(bbox),
        }
        annotations.append(formatted_annotation)

    return {
        "image_id": image_id,
        "annotations": annotations,
    }


def convert_bbox_yolo_to_pascal(boxes: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    """
    Convert bounding boxes from YOLO format (x_center, y_center, width, height) in range [0, 1]
    to Pascal VOC format (x_min, y_min, x_max, y_max) in absolute coordinates.

    Args:
        boxes (torch.Tensor): Bounding boxes in YOLO format
        image_size (tuple[int, int]): Image size in format (height, width)

    Returns:
        torch.Tensor: Bounding boxes in Pascal VOC format (x_min, y_min, x_max, y_max)
    """
    # convert center to corners format
    boxes = center_to_corners_format(boxes)

    # convert to absolute coordinates
    height, width = image_size
    boxes = boxes * torch.tensor([[width, height, width, height]])

    return boxes


def augment_and_transform_batch(
    examples: Mapping[str, Any],
    transform: A.Compose,
    image_processor: AutoImageProcessor,
    return_pixel_mask: bool = False,
) -> BatchFeature:
    """Apply augmentations and format annotations in COCO format for object detection task"""

    images = []
    annotations = []
    for image_id, image, objects in zip(examples["image_id"], examples["image"], examples["objects"]):
        image = np.array(image.convert("RGB"))

        # apply augmentations
        output = transform(image=image, bboxes=objects["bbox"], category=objects["category"])
        images.append(output["image"])

        # format annotations in COCO format
        formatted_annotations = format_image_annotations_as_coco(
            image_id, output["category"], objects["area"], output["bboxes"]
        )
        annotations.append(formatted_annotations)

    # Apply the image processor transformations: resizing, rescaling, normalization
    result = image_processor(images=images, annotations=annotations, return_tensors="pt")

    if not return_pixel_mask:
        result.pop("pixel_mask", None)

    return result


def collate_fn(batch: list[BatchFeature]) -> Mapping[str, Union[torch.Tensor, list[Any]]]:
    data = {}
    data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
    data["labels"] = [x["labels"] for x in batch]
    if "pixel_mask" in batch[0]:
        data["pixel_mask"] = torch.stack([x["pixel_mask"] for x in batch])
    return data

@torch.no_grad()
def compute_metrics(
    evaluation_results: EvalPrediction,
    image_processor: AutoImageProcessor,
    threshold: float = 0.0,
    id2label: Optional[Mapping[int, str]] = None,
) -> Mapping[str, float]:
    """
    Compute mean average mAP, mAR and their variants for the object detection task.

    Args:
        evaluation_results (EvalPrediction): Predictions and targets from evaluation.
        threshold (float, optional): Threshold to filter predicted boxes by confidence. Defaults to 0.0.
        id2label (Optional[dict], optional): Mapping from class id to class name. Defaults to None.

    Returns:
        Mapping[str, float]: Metrics in a form of dictionary {<metric_name>: <metric_value>}
    """

    predictions, targets = evaluation_results.predictions, evaluation_results.label_ids

    # Debug: Verificar la estructura de los datos
    print(f"Predictions type: {type(predictions)}")
    print(f"Targets type: {type(targets)}")
    if hasattr(predictions, 'shape'):
        print(f"Predictions shape: {predictions.shape}")
    if hasattr(targets, 'shape'):
        print(f"Targets shape: {targets.shape}")

    # Para RT-DETR, las predicciones vienen en un formato diferente
    # Necesitamos adaptarnos a la salida del modelo
    
    image_sizes = []
    post_processed_targets = []
    post_processed_predictions = []

    # Procesar targets - verificar si ya están en el formato correcto
    if isinstance(targets, (list, tuple)) and len(targets) > 0:
        # Si targets es una lista de batches
        for batch_idx, batch in enumerate(targets):
            if isinstance(batch, dict):
                # Formato esperado: batch es un diccionario
                batch_image_sizes = torch.tensor([x["orig_size"] for x in batch])
                image_sizes.append(batch_image_sizes)
                
                for image_target in batch:
                    if isinstance(image_target, dict) and "boxes" in image_target:
                        boxes = torch.tensor(image_target["boxes"])
                        boxes = convert_bbox_yolo_to_pascal(boxes, image_target["orig_size"])
                        labels = torch.tensor(image_target["class_labels"])
                        post_processed_targets.append({"boxes": boxes, "labels": labels})
            else:
                print(f"Batch {batch_idx} no es un diccionario, tipo: {type(batch)}")
                continue
    else:
        print(f"Targets no es una lista/tupla, tipo: {type(targets)}")
        return {"map": 0.0}  # Retornar métricas por defecto si no podemos procesar

    # Procesar predicciones - RT-DETR tiene formato diferente
    if isinstance(predictions, (list, tuple)) and len(predictions) > 0:
        for batch_idx, (batch_pred, target_sizes) in enumerate(zip(predictions, image_sizes)):
            try:
                # RT-DETR devuelve las predicciones en un formato específico
                # Asumimos que batch_pred es una tupla (loss, logits, pred_boxes) o similar
                if isinstance(batch_pred, (tuple, list)) and len(batch_pred) >= 3:
                    # Tomar logits y boxes (índices 1 y 2)
                    batch_logits, batch_boxes = batch_pred[1], batch_pred[2]
                else:
                    # Si es un array numpy, intentar interpretarlo
                    batch_logits, batch_boxes = batch_pred, batch_pred
                
                output = ModelOutput(
                    logits=torch.tensor(batch_logits) if not isinstance(batch_logits, torch.Tensor) else batch_logits,
                    pred_boxes=torch.tensor(batch_boxes) if not isinstance(batch_boxes, torch.Tensor) else batch_boxes
                )
                
                post_processed_output = image_processor.post_process_object_detection(
                    output, threshold=threshold, target_sizes=target_sizes
                )
                post_processed_predictions.extend(post_processed_output)
            except Exception as e:
                print(f"Error procesando batch {batch_idx}: {e}")
                continue
    else:
        print(f"Predictions no es una lista/tupla, tipo: {type(predictions)}")
        return {"map": 0.0}

    # Si no tenemos suficientes datos para calcular métricas, retornar valores por defecto
    if len(post_processed_predictions) == 0 or len(post_processed_targets) == 0:
        print("No hay suficientes datos para calcular métricas")
        return {"map": 0.0, "mar_100": 0.0}

    # Compute metrics
    try:
        metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
        metric.update(post_processed_predictions, post_processed_targets)
        metrics = metric.compute()

        # Reemplazar lista de métricas por clase con métricas separadas para cada clase
        if "classes" in metrics and "map_per_class" in metrics and "mar_100_per_class" in metrics:
            classes = metrics.pop("classes")
            map_per_class = metrics.pop("map_per_class")
            mar_100_per_class = metrics.pop("mar_100_per_class")
            
            for class_id, class_map, class_mar in zip(classes, map_per_class, mar_100_per_class):
                class_name = id2label[class_id.item()] if id2label is not None else class_id.item()
                metrics[f"map_{class_name}"] = class_map
                metrics[f"mar_100_{class_name}"] = class_mar

        metrics = {k: round(v.item(), 4) if hasattr(v, 'item') else v for k, v in metrics.items()}
        
    except Exception as e:
        print(f"Error calculando métricas: {e}")
        metrics = {"map": 0.0, "mar_100": 0.0}

    return metrics


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to specify
    them on the command line.
    """

    dataset_name: str = field(
        default="cppe-5",
        metadata={
            "help": "Name of a dataset from the hub (could be your own, possibly private dataset hosted on the hub)."
        },
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_val_split: Optional[float] = field(
        default=0.15, metadata={"help": "Percent to split off of train for validation."}
    )
    image_square_size: Optional[int] = field(
        default=640,  # Cambiado a 640 para RT-DETR (divisible por 32)
        metadata={"help": "Image size for RT-DETR (should be divisible by 32)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    use_fast: Optional[bool] = field(
        default=True,
        metadata={"help": "Use a fast torchvision-base image processor if it is supported for a given model."},
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="PekingU/rtdetr_v2_r50vd",  # Cambiado a RT-DETR
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    image_processor_name: str = field(default=None, metadata={"help": "Name or path of preprocessor config."})
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to raise an error if some of the weights from the checkpoint do not have the same size as the weights of the model (if for instance, you are instantiating a model with 10 labels from a checkpoint with 3 labels)."
        },
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `hf auth login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        checkpoint = get_last_checkpoint(training_args.output_dir)
        if checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # ------------------------------------------------------------------------------------------------
    # Load dataset, prepare splits
    # ------------------------------------------------------------------------------------------------

    dataset = load_dataset(
        data_args.dataset_name, cache_dir=model_args.cache_dir, trust_remote_code=model_args.trust_remote_code
    )
    dataset = filter_cppe5_dataset(dataset)
    # If we don't have a validation split, split off a percentage of train as validation
    data_args.train_val_split = None if "validation" in dataset else data_args.train_val_split
    if isinstance(data_args.train_val_split, float) and data_args.train_val_split > 0.0:
        split = dataset["train"].train_test_split(data_args.train_val_split, seed=training_args.seed)
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]

    # Get dataset categories and prepare mappings for label_name <-> label_id
    if isinstance(dataset["train"].features["objects"], dict):
        categories = dataset["train"].features["objects"]["category"].feature.names
    else:  # (for old versions of `datasets` that used Sequence({...}) of the objects)
        categories = dataset["train"].features["objects"].feature["category"].names
    id2label = dict(enumerate(categories))
    label2id = {v: k for k, v in id2label.items()}

    # ------------------------------------------------------------------------------------------------
    # Load pretrained config, model and image processor
    # ------------------------------------------------------------------------------------------------

    common_pretrained_args = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    config = AutoConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        label2id=label2id,
        id2label=id2label,
        **common_pretrained_args,
    )
    model = AutoModelForObjectDetection.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        **common_pretrained_args,
    )
    
    # CONFIGURACIÓN ESPECÍFICA PARA RT-DETR
    image_processor = AutoImageProcessor.from_pretrained(
        model_args.model_name_or_path,
        do_resize=True,
        size={"height": data_args.image_square_size, "width": data_args.image_square_size},
        do_normalize=True,
        do_rescale=True,
        do_center_crop=False,
        do_pad=False,  # RT-DETR no necesita padding
        use_fast=data_args.use_fast,
        **common_pretrained_args,
    )

    print("#######################################")
    print(f"Using model: {model_args.model_name_or_path}")
    print(f"Image size: {data_args.image_square_size}")
    print(f"Image processor: {image_processor}")
    print("#######################################")

    # ------------------------------------------------------------------------------------------------
    # Define image augmentations and dataset transforms
    # ------------------------------------------------------------------------------------------------
    max_size = data_args.image_square_size
    
    # Augmentaciones más simples para RT-DETR
    train_augment_and_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=5, p=0.3),
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True, min_area=25),
    )
    
    validation_transform = A.Compose(
        [A.NoOp()],
        bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True),
    )

    # Make transform functions for batch and apply for dataset splits
    train_transform_batch = partial(
        augment_and_transform_batch, transform=train_augment_and_transform, image_processor=image_processor
    )
    validation_transform_batch = partial(
        augment_and_transform_batch, transform=validation_transform, image_processor=image_processor
    )

    dataset["train"] = dataset["train"].with_transform(train_transform_batch)
    dataset["validation"] = dataset["validation"].with_transform(validation_transform_batch)
    dataset["test"] = dataset["test"].with_transform(validation_transform_batch)

    # ------------------------------------------------------------------------------------------------
    # Model training and evaluation with Trainer API
    # ------------------------------------------------------------------------------------------------

    eval_compute_metrics_fn = partial(
        compute_metrics, image_processor=image_processor, id2label=id2label, threshold=0.0
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"] if training_args.do_train else None,
        eval_dataset=dataset["validation"] if training_args.do_eval else None,
        processing_class=image_processor,
        data_collator=collate_fn,
        compute_metrics=eval_compute_metrics_fn,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Final evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(eval_dataset=dataset["test"], metric_key_prefix="test")
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

    # Write model card and (optionally) push to hub
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": data_args.dataset_name,
        "tags": ["object-detection", "vision"],
    }
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()
