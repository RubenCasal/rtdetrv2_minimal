from datasets import Dataset, DatasetDict
from PIL import Image
import numpy as np
import torch

def collate_fn(batch):
    data = {}
    data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
    data["labels"] = [x["labels"] for x in batch]
    return data


def filter_cppe5_dataset(dataset):
    """
    Filtra el dataset CPPE-5 eliminando imágenes con bboxes inválidas
    Devuelve un DatasetDict listo para usar
    
    Args:
        dataset: Dataset original de load_dataset("cppe-5")
    
    Returns:
        DatasetDict filtrado
    """
    
    def filter_invalid_bboxes(example):
        """Filtra bboxes inválidas en un solo ejemplo"""
        image = example["image"]
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_width, img_height = image.size
        objects = example["objects"]
        
        # Si no hay bboxes, mantener el ejemplo intacto
        if not isinstance(objects, dict) or 'bbox' not in objects:
            return example
        
        bboxes = objects['bbox']
        categories = objects['category']
        
        valid_bboxes = []
        valid_categories = []
        
        # Filtrar bboxes inválidas
        for j, bbox in enumerate(bboxes):
            category_id = categories[j] if j < len(categories) else categories[0]
            x, y, w, h = bbox
            
            # Validar bbox
            if (x >= 0 and y >= 0 and w > 0 and h > 0 and 
                x + w <= img_width and y + h <= img_height):
                
                valid_bboxes.append(bbox)
                valid_categories.append(category_id)
        
        # Actualizar el ejemplo con solo bboxes válidas
        filtered_objects = objects.copy()
        filtered_objects['bbox'] = valid_bboxes
        filtered_objects['category'] = valid_categories
        
        return {**example, "objects": filtered_objects}
    
    def remove_empty_images(example):
        """Elimina ejemplos que no tienen bboxes válidas"""
        objects = example["objects"]
        if isinstance(objects, dict) and 'bbox' in objects:
            return len(objects['bbox']) > 0
        return False
    
    print("🔄 Filtrando dataset...")
    
    # Aplicar filtros
    filtered_dataset = DatasetDict()
    
    for split in dataset.keys():
        print(f"📁 Procesando split: {split}")
        
        # Primero filtrar bboxes inválidas
        split_data = dataset[split].map(
            filter_invalid_bboxes,
            desc=f"Filtrando bboxes inválidas en {split}"
        )
        
        # Luego eliminar imágenes sin bboxes
        original_size = len(split_data)
        split_data = split_data.filter(
            remove_empty_images,
            desc=f"Eliminando imágenes vacías en {split}"
        )
        filtered_size = len(split_data)
        
        filtered_dataset[split] = split_data
        
        print(f"   ✅ {split}: {original_size} → {filtered_size} imágenes")
        print(f"   🗑️  Imágenes removidas: {original_size - filtered_size}")
    
    return filtered_dataset