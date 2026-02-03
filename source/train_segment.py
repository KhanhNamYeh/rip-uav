from ultralytics import YOLO
import os
from utils import rename_folder

def main():
    # 1. Load model YOLO12n-seg (Cập nhật từ bản Small theo args.yaml)
    model = YOLO("runs/segment/TrainRip/yolo12n_seg_run1/weights/best.pt")


    rename_folder(
        old_name="data/train/labels_segmentation",
        new_name="data/train/labels"
    )

    rename_folder(
        old_name='data/val/labels_segmentation',
        new_name='data/val/labels'
    )

    # Kiểm tra file data.yaml
    if not os.path.exists("data.yaml"):
        print("Error: data.yaml không tồn tại!")
        return

    # 2. Train với các thông số từ args.yaml
    results = model.train(
        # --- CẤU HÌNH CƠ BẢN ---
        data="data.yaml",
        device='0',
        workers=12,                # Giảm từ 16 xuống 12
        project="TrainRip",
        name="yolo12n_seg_run2",  # Tên run mới
        
        # --- CẤU HÌNH TRAINING ---
        epochs=100,                # Giảm từ 150 xuống 100
        imgsz=640,
        batch=16,
        patience=20,               # Giảm từ 30 xuống 20
        save=True,
        pretrained=True,
        optimizer="SGD",           # Đổi từ AdamW sang SGD
        verbose=True,
        seed=0,
        deterministic=True,
        
        # --- LEARNING RATE & MOMENTUM ---
        lr0=0.01,                  # Tăng từ 0.0069
        lrf=0.0125,                # Giảm từ 0.04405
        momentum=0.8,              # Giảm từ 0.98
        weight_decay=0.005,        # Tăng đáng kể (từ 0.00014)
        
        warmup_epochs=3.42,
        warmup_momentum=0.80147,
        warmup_bias_lr=0.1,        # Thêm mới theo cấu hình
        
        # --- LOSS WEIGHTS ---
        box=4.1822,
        cls=0.42185,
        dfl=0.98998,
        
        # --- AUGMENTATION (BIẾN ĐỔI ẢNH) ---
        hsv_h=0.00826,
        hsv_s=0.64762,
        hsv_v=0.39278,
        degrees=9.9931,
        translate=0.15121,
        scale=0.23702,
        shear=0.00979,
        perspective=0.0001,
        flipud=0.68771,
        fliplr=0.66764,
        bgr=0.00553,
        
        mosaic=1.0,
        mixup=0.1,                 # Tăng nhẹ từ 0.08951
        copy_paste=0.30407,
        auto_augment="randaugment", # Thêm mới
        erasing=0.4,               # Thêm mới
        
        close_mosaic=9,            # Tắt Mosaic ở 9 epoch cuối
    )

    print("✓ Training completed!")
    
    # 3. Validation
    metrics = model.val(data="data.yaml", split="val")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    
    # 4. Export
    model.export(format="onnx", imgsz=640, simplify=True)
    model.export(format="torchscript")

    rename_folder(
        old_name="data/train/labels",
        new_name="data/train/labels_segmentation"
    )

    rename_folder(
        old_name='data/val/labels',
        new_name='data/val/labels_segmentation'
    )

if __name__ == '__main__':
    import torch.multiprocessing
    torch.multiprocessing.set_start_method('spawn', force=True)  
    main()