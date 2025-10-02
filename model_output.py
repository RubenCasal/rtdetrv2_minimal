import torch
from dataclasses import dataclass


@dataclass
class ModelOutput:
    logits: torch.Tensor
    pred_boxes: torch.Tensor