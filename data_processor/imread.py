import cv2
import os

image_files = ["mask_img.png", "glass_img3.png", "sunglass_img2.png"]
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"当前脚本所在目录: {current_dir}")

for filename in image_files:
    file_path = os.path.join(current_dir, filename)
    print(f"尝试加载文件: {file_path}")
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"错误: 无法加载 {filename}，请检查文件是否存在或是否损坏。")
    else:
        print(f"成功加载 {filename}，图片形状: {img.shape}")