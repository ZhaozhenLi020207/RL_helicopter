import os
from ultralytics import YOLO
import warnings

warnings.filterwarnings("ignore")

# 基础配置
# 数据集配置文件路径
DATA_YAML = "F:/python/RL_project/helicopter_dateset/helicopter.yaml"
# 预训练模型（n=轻量，s=标准，m=中等，l=大，xl=超大）
BASE_MODEL = "yolov8n.pt"
# 训练参数
EPOCHS = 50  # 训练轮数（新手建议30-50）
BATCH_SIZE = 8  # 批次大小
IMG_SIZE = 640  # 输入图片尺寸
DEVICE = "cpu"  # 显卡ID
SAVE_DIR = "runs/helicopter_train"  # 训练结果保存路径

# 加载预训练模型
print("加载预训练模型：", BASE_MODEL)
model = YOLO(BASE_MODEL)

# 开始微调训练
print("\n 开始微调YOLOv8模型")
results = model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    batch=BATCH_SIZE,
    imgsz=IMG_SIZE,
    device=DEVICE,
    save_dir=SAVE_DIR,
    patience=10,  # 早停（10轮无提升则停止）
    lr0=0.01,  # 初始学习率
    lrf=0.01,  # 最终学习率
    weight_decay=0.0005,  # 权重衰减（防止过拟合）
    warmup_epochs=3,  # 热身轮数
    augment=True,  # 数据增强（提升泛化能力）
    save=True,  # 保存模型
    val=True  # 训练时验证
)

# 训练完成后评估
print("\n训练完成，开始评估模型...")
metrics = model.val()  # 在验证集上评估
print(f"验证集mAP50：{metrics.box.map50:.3f}")  # 关键指标（越高越好）
print(f"最佳模型路径：{os.path.join(SAVE_DIR, 'train/weights/best.pt')}")
