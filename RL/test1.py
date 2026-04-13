import torch
import time

print("PyTorch 版本:", torch.__version__)
print("CUDA 是否可用:", torch.cuda.is_available())
print("CUDA 版本:", torch.version.cuda)
print("GPU 型号:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无")

if torch.cuda.is_available():
    # 大矩阵乘法，测真实算力
    a = torch.randn(20000, 20000, device='cuda')
    b = torch.randn(20000, 20000, device='cuda')

    torch.cuda.synchronize()
    t0 = time.time()

    c = a @ b  # 矩阵乘法

    torch.cuda.synchronize()
    t1 = time.time()

    print(f"矩阵乘法耗时: {t1 - t0:.2f} 秒")
else:
    print("CUDA 不可用，训练肯定慢！")