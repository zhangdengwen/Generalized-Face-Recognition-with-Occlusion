# import sys
# sys.path.append("/home/srp/face_recognition/clip")  # CLIP 源码路径
# import clip
# import torch
# from PIL import Image
# import os
# from tqdm import tqdm

# # 设备
# device = "cuda" if torch.cuda.is_available() else "cpu"

# # 加载模型
# model, preprocess = clip.load("RN50x16", device=device)
# model.eval()

# # 定义检测 prompt
# prompts = [
#     "A human face",
#     "A human face with eyes masked",
#     "A human face with mouth masked"
# ]
# text_tokens = clip.tokenize(prompts).to(device)

# # 数据集路径
# dataset_path = "/home/srp/face_recognition/test_data/calfw_crop"

# # 初始化计数器
# counts = {prompt: 0 for prompt in prompts}
# total = 0

# for img_name in tqdm(os.listdir(dataset_path), desc="Processing images"):
#     img_path = os.path.join(dataset_path, img_name)
#     try:
#         image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)

#         with torch.no_grad():
#             logits_per_image, _ = model(image, text_tokens)
#             probs = logits_per_image.softmax(dim=-1)[0]

#         # 阈值判断，可调
#         for i, prob in enumerate(probs):
#             if prob > 0.5:
#                 counts[prompts[i]] += 1

#         total += 1
#     except Exception as e:
#         print(f"Error processing {img_name}: {e}")

# # 输出统计结果
# print(f"\nTotal images: {total}")
# for prompt in prompts:
#     print(f"{prompt}: {counts[prompt]} ({counts[prompt]/total:.2%})")


 #  PYTHONPATH=. python -m test_protocol.test_clip


import sys
sys.path.append("/home/srp/face_recognition/clip")  # CLIP 源码路径
import clip
import torch
from PIL import Image
import os
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载模型
model, preprocess = clip.load("RN50x16", device=device)
model.eval()

# 定义检测 prompt
prompts = [
    "A human face in a mask",
    "A human face with glasses",
    "A human face with sunglasses"
]
text_tokens = clip.tokenize(prompts).to(device)

# 数据集路径
dataset_path = "/home/srp/face_recognition/test_data/CALFW/CALFW_reid_random_sunglasses4"

# 初始化计数器
counts = {prompt: 0 for prompt in prompts}
total = 0

for root, _, files in os.walk(dataset_path):
    for img_name in files:
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img_path = os.path.join(root, img_name)
        try:
            image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)

            with torch.no_grad():
                logits_per_image, _ = model(image, text_tokens)
                probs = logits_per_image.softmax(dim=-1)[0]

            for i, prob in enumerate(probs):
                if prob > 0.5:
                    counts[prompts[i]] += 1

            total += 1
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

# 输出统计结果
print(f"\nTotal images: {total}")
for prompt in prompts:
    ratio = counts[prompt] / total if total > 0 else 0
    print(f"{prompt}: {counts[prompt]} ({ratio:.2%})")
