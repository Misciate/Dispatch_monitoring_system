import os
import shutil

src_root = "Dataset/classification"
dst_root = "Dataset/classification_flat"

os.makedirs(dst_root, exist_ok=True)

for parent in os.listdir(src_root):
    parent_path = os.path.join(src_root, parent)
    if not os.path.isdir(parent_path):
        continue

    for sub in os.listdir(parent_path):
        src_dir = os.path.join(parent_path, sub)
        dst_dir = os.path.join(dst_root, f"{parent}_{sub}")
        os.makedirs(dst_dir, exist_ok=True)

        for file in os.listdir(src_dir):
            src_file = os.path.join(src_dir, file)
            dst_file = os.path.join(dst_dir, file)
            shutil.copy2(src_file, dst_file)

print("✅ Đã flatten xong vào thư mục Dataset/classification_flat")
