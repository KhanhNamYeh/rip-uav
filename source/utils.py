import json
import os
import random
import shutil
import yaml
from pathlib import Path

def split_data(json_path, output_dir, split_ratio=0.8):
    # output path structure
    labels= Path(output_dir) / 'labels/'
    labels.mkdir(parents=True, exist_ok=True)

    # read file JSON        
    with open(json_path, 'r') as f:
        data = json.load(f)

    images = {img['id']: img for img in data['images']}
    categories = data['categories']

    # debug load data
    # if len(data['images']) == len(images):
    #     print("images match found in file json: ", len(images))
    # else:
    #     print("images not match found in file json, missing: ", len(data['images']) - len(images))
    # print(len(categories), "categories found in the dataset.")
    # print(next(iter(images.items()))) # Print first item for debugging
    

    # Split dataset
    img_ids = list(images.keys())
    random.seed(42)  # For reproducibility
    random.shuffle(img_ids)
    split_idx = int(len(img_ids) * split_ratio)
    train_ids = img_ids[:split_idx]
    val_ids = img_ids[split_idx:]

    if not os.path.exists(labels / 'split_ids.json'):
        with open(labels / 'split_ids.json', 'w') as f:
            json.dump({"train": train_ids, "val": val_ids}, f)

    if not os.path.exists(labels / 'train_path.txt'):
        with open(labels / 'train_path.txt', 'w') as f:
            for img_id in train_ids:
                f.write(f"data/train/train_images/{images[img_id]['file_name']}\n")

    if not os.path.exists(labels / 'val_path.txt'):
        with open(labels / 'val_path.txt', 'w') as f:
            for img_id in val_ids:
                f.write(f"data/train/train_images/{images[img_id]['file_name']}\n")


def rename_folder(old_name, new_name):
    old_path = Path(old_name)
    new_path = Path(new_name)

    if old_path.exists():
        os.rename(old_path, new_path)
    else:
        print("error")

def replace(train_path, val_path, val_file='labels/val_path.txt'):
    """
    Di chuyển bộ 3 (Image, Detection, Segmentation) từ train sang val.
    """
    # Danh sách hậu tố khớp với cấu trúc thư mục của bạn
    suffixes = ['images', 'labels_detection', 'labels_segmentation']

    if not os.path.exists(val_file):
        print(f"Lỗi: Không tìm thấy file danh sách tại {val_file}")
        return

    with open(val_file, 'r') as f:
        val_lines = [line.strip() for line in f.readlines() if line.strip()]

    moved_count = 0
    for path_in_txt in val_lines:
        # Lấy tên file gốc từ đường dẫn trong file txt
        img_filename = os.path.basename(path_in_txt)
        # Tên file label tương ứng (đuôi .txt)
        label_filename = os.path.splitext(img_filename)[0] + '.txt'

        for sfx in suffixes:
            # logic: thay thế chữ 'images' trong đường dẫn bằng các hậu tố khác
            src_dir = train_path.replace('images', sfx)
            dst_dir = val_path.replace('images', sfx)

            os.makedirs(dst_dir, exist_ok=True)

            # File ảnh giữ nguyên đuôi, file nhãn dùng đuôi .txt
            current_file = img_filename if 'images' in sfx else label_filename
            
            src_path = os.path.join(src_dir, current_file)
            dst_path = os.path.join(dst_dir, current_file)

            if os.path.exists(src_path):
                shutil.move(src_path, dst_path)
        
        moved_count += 1

