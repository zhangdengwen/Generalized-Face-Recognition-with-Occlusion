import torch

print("=" * 40)
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
print(f"当前 CUDA 版本: {torch.version.cuda}")
print(f"设备数量: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"当前使用设备: {torch.cuda.get_device_name(0)}")

# 尝试一次张量计算
try:
    x = torch.randn(3, 3).cuda()
    y = torch.randn(3, 3).cuda()
    z = x + y
    print("张量运算成功，结果如下：")
    print(z)
except Exception as e:
    print("GPU 运算失败：", e)

print("=" * 40)
