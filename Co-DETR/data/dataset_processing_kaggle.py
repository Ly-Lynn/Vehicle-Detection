import os
import json
from sklearn.model_selection import train_test_split
import os
import shutil
def filter_annotations_by_image_ids(image_ids, annotations):
    return [annotation for annotation in annotations if annotation['image_id'] in image_ids]

def save_coco_split(file_path, images, annotations, coco_data):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    coco_split = {
        'info': coco_data['info'],
        'licenses': coco_data['licenses'],
        'categories': coco_data['categories'],
        'images': images,
        'annotations': annotations
    }
    
    with open(file_path, 'w') as f:
        json.dump(coco_split, f)

def move_images(image_list, source_folder, dest_folder):
    for image in image_list:
        image_file = image['file_name']
        shutil.copy(os.path.join(source_folder, image_file), os.path.join(dest_folder, image_file))


if __name__ == "__main__":
    train_path = '/kaggle/Vehicle-Detection/Co-DETR/data/vehicle/train'
    val_path = '/kaggle/Vehicle-Detection/Co-DETR/data/vehicle/val'
    test_path = '/kaggle/Vehicle-Detection/Co-DETR/data/vehicle/test'
    json_train = '/kaggle/Vehicle-Detection/Co-DETR/data/vehicle/annotations/train_annotations.coco.json'
    json_val = '/kaggle/Vehicle-Detection/Co-DETR/data/vehicle/annotations/val_annotations.coco.json'
    json_test = '/kaggle/Vehicle-Detection/Co-DETR/data/vehicle/annotations/test_annotations.coco.json'

    coco_annotation_path = '/kaggle/input/track1-traffic-vehicle-detection/daytime/daytime/_annotations.coco.json'

    with open(coco_annotation_path, 'r') as f:
        coco_data = json.load(f)

    images = coco_data['images']  
    annotations = coco_data['annotations']  

    # 70% train, 15% validation, 15% test
    train_images, temp_images = train_test_split(images, test_size=0.3, random_state=42)
    val_images, test_images = train_test_split(temp_images, test_size=0.5, random_state=42)

    print(f"Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")

    train_image_ids = [img['id'] for img in train_images]
    val_image_ids = [img['id'] for img in val_images]
    test_image_ids = [img['id'] for img in test_images]

    train_annotations = filter_annotations_by_image_ids(train_image_ids, annotations)
    val_annotations = filter_annotations_by_image_ids(val_image_ids, annotations)
    test_annotations = filter_annotations_by_image_ids(test_image_ids, annotations)

    save_coco_split(json_train, train_images, train_annotations, coco_data)
    save_coco_split(json_val, val_images, val_annotations, coco_data)
    save_coco_split(json_test, test_images, test_annotations, coco_data)

    image_folder = '/kaggle/input/track1-traffic-vehicle-detection/daytime/daytime/train'

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    move_images(train_images, image_folder, train_path)
    move_images(val_images, image_folder, val_path)
    move_images(test_images, image_folder, test_path)



