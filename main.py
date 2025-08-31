

import clip
import torch

# 加载 RN50x16 模型（会自动下载到 ~/.cache/clip/）
model, preprocess = clip.load("RN50x16", device="cuda" if torch.cuda.is_available() else "cpu")

print("模型加载成功！")
