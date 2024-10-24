import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
import os
import numpy as np
from pycocotools.coco import COCO
import albumentations as A
from albumentations.pytorch import ToTensorV2

class VehicleDataset(Dataset):
    def __init__(self, root_dir, ann_path, transforms=None, is_train=False):
        self.root_dir = root_dir
        self.is_train = is_train
        self.anno = COCO(ann_path)
        if transforms:
            self.transforms = transforms
        else:
            self.transforms = self.get_default_transforms()
    
    def get_default_transforms(self):
        """Default augmentations cho vehicle detection"""
        if self.is_train:
            return A.Compose([
                # Spatial augmentations
                A.RandomResizedCrop(height=640, width=640, scale=(0.8, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.2),
                
                # Blur/Noise - ảnh mờ
                A.OneOf([
                    A.MotionBlur(p=0.5),
                    A.MedianBlur(blur_limit=3, p=0.5),
                    A.GaussianBlur(blur_limit=3, p=0.5),
                ], p=0.3),
                
                # Color augmentations
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.RandomBrightnessContrast(p=0.4),
                
                # Weather simulations - outdoor vehicles
                A.OneOf([
                    A.RandomRain(p=0.3),
                    A.RandomFog(p=0.3),
                    A.RandomSunFlare(p=0.3),
                ], p=0.2),
                
                # Normalize và convert to tensor
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ], bbox_params=A.BboxParams(
                format='coco',  # [x_min, y_min, x_max, y_max]
                label_fields=['class_labels']
            ))
        else:
            return A.Compose([
                A.Resize(640, 640),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ], bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['class_labels']
            ))
        
    def __getitem__(self, idx):
        img_id = self.ids[idx]
    
        img_info = self.coco.loadImgs(img_id)[0]
        image_path = os.path.join(self.root_dir, img_info['file_name'])
        img = Image.open(image_path).convert('RGB')
        
        w, h = img.size
        
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)
        
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        
        for ann in annotations:
            bbox = ann['bbox']  
            # Chuyển về format [x_min, y_min, x_max, y_max]
            bbox = [
                bbox[0], 
                bbox[1], 
                bbox[0] + bbox[2], 
                bbox[1] + bbox[3]
            ]
            boxes.append(bbox)
            labels.append(ann['category_id'])
            areas.append(ann['area'])
            iscrowd.append(ann['iscrowd'])
        
        # Chuyển về tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        
        # Tạo target dictionary
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id]),
            'area': areas,
            'iscrowd': iscrowd,
            'orig_size': torch.as_tensor([h, w]),
        }
        
        # Áp dụng transforms
        if self.transform is not None:
            img = self.transform(img)
            
            # Chuẩn hóa boxes nếu cần
            if len(boxes):
                boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
                target['boxes'] = boxes
        
        return img, target

    def __len__(self):
        return len(self.ids)
    
    def get_height_and_width(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        return img_info['height'], img_info['width']

    def prepare_for_evaluation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            scores = prediction["scores"]
            labels = prediction["labels"]

            boxes = boxes.tolist()
            scores = scores.tolist()
            labels = labels.tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results