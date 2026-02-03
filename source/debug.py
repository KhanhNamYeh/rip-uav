from utils import split_data, rename_folder, replace

# split_data(
#     json_path="data/train/train_annotations.json",
#     output_dir=".",
#     split_ratio=0.8
# )

# rename_folder(
#         old_name="train_labels",
#         new_name="train_label_segmentation"
#     )

replace(
    train_path="data/train/train_images",
    val_path="data/train/val_images",
    )