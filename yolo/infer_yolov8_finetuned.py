import cv2
import numpy as np
import math
from ultralytics import YOLO
import warnings

warnings.filterwarnings("ignore")

# 加载YOLOv8模型
# 替换为微调后的模型路径（训练完成后在runs/helicopter_train/train/weights/下）
FINETUNED_MODEL_PATH = "../runs/helicopter_train/weights/best.pt"
model = YOLO(FINETUNED_MODEL_PATH)

# 甲板检测+位姿计算逻辑
def euler_from_rotation_matrix(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])

def detect_deck_corners_robust(img):
    h, w = img.shape[:2]
    blur_ksize = (5 if w > 500 else 3, 5 if h > 500 else 3)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, blur_ksize, 0)
    median_val = np.median(blur)
    lower = int(max(0, 0.6 * median_val))
    upper = int(min(255, 1.4 * median_val))
    edges = cv2.Canny(blur, lower, upper)
    kernel_size = 5 if max(w, h) > 800 else 3
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("未检测到甲板轮廓")
    max_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(max_contour)
    box = cv2.boxPoints(rect)
    img_pts = np.array(box, dtype=np.float32)
    img_pts = img_pts[np.lexsort((img_pts[:, 0], img_pts[:, 1]))]
    return img_pts, max_contour

# 识别直升机
def detect_helicopter_with_finetuned_yolo(img, conf_threshold=0.5):
    """用微调后的YOLOv8检测直升机"""
    # 执行检测（仅检测自定义的helicopter类别）
    results = model(img, conf=conf_threshold)
    if len(results[0].boxes) == 0:
        print(f" 未检测到直升机（置信度阈值={conf_threshold}）")
        return None, None, 0.0

    # 取置信度最高的直升机框
    best_box = results[0].boxes[0]
    x1, y1, x2, y2 = map(int, best_box.xyxy[0].cpu().numpy())
    conf = best_box.conf[0].cpu().numpy()
    cls = best_box.cls[0].cpu().numpy()
    cls_name = model.names[int(cls)]

    # 提取直升机ROI，计算最小外接矩形
    aircraft_roi = img[y1:y2, x1:x2]
    gray_roi = cv2.cvtColor(aircraft_roi, cv2.COLOR_BGR2GRAY)
    blur_roi = cv2.GaussianBlur(gray_roi, (3, 3), 0)
    edges_roi = cv2.Canny(blur_roi, 50, 150)
    contours_roi, _ = cv2.findContours(edges_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours_roi:
        # 用检测框生成外接矩形
        aircraft_rect = cv2.minAreaRect(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]))
    else:
        # 转换为原图坐标
        max_contour_roi = max(contours_roi, key=cv2.contourArea)
        max_contour_roi = max_contour_roi + np.array([x1, y1])
        aircraft_rect = cv2.minAreaRect(max_contour_roi)

    # 计算直升机外接矩形和中心
    aircraft_box = cv2.boxPoints(aircraft_rect)
    aircraft_box = np.intp(aircraft_box)
    aircraft_center = (int(aircraft_rect[0][0]), int(aircraft_rect[0][1]))

    print(f"\n 直升机检测结果：")
    print(f"类别：{cls_name} | 置信度：{conf:.3f}")
    print(f"检测框：x1={x1}, y1={y1}, x2={x2}, y2={y2}")
    print(f"中心坐标：{aircraft_center}")
    # 修复点2：返回值增加conf，让主函数能访问
    return aircraft_box, aircraft_center, conf

# 主函数
def helicopter_pose_analysis_finetuned(image_path):
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f" 无法读取图片：{image_path}")
    h, w = img.shape[:2]
    img_display = img.copy()
    print(f"图片信息：分辨率 {w}x{h}")

    # 甲板检测 + 位姿计算
    try:
        img_pts, deck_contour = detect_deck_corners_robust(img)
        # 位姿计算
        obj_pts = np.array([[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]], dtype=np.float32)
        K = np.array([[w, 0, w // 2], [0, h, h // 2], [0, 0, 1]], dtype=np.float32)
        dist = np.zeros((5,))
        success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_EPNP)
        if success:
            R, _ = cv2.Rodrigues(rvec)
            euler = euler_from_rotation_matrix(R)
            roll, pitch, yaw = np.degrees(euler)
            print("\n 甲板位姿结果：")
            print(f"旋转矩阵：\n{R.round(4)}")
            print(f"平移向量（米）：{tvec.round(4).flatten()}")
            print(f"欧拉角：Roll={roll:.2f}°, Pitch={pitch:.2f}°, Yaw={yaw:.2f}°")
        # 绘制甲板标注
        deck_box = cv2.boxPoints(cv2.minAreaRect(deck_contour))
        deck_box = np.intp(deck_box)
        cv2.drawContours(img_display, [deck_box], 0, (0, 255, 0), 2)
        deck_center = np.mean(deck_box, axis=0).astype(np.int32)
        dir_x = (deck_box[0] - deck_box[1]).astype(np.float32)
        dir_x = dir_x / np.linalg.norm(dir_x)
        dir_y = np.array([dir_x[1], -dir_x[0]])
        x_end = (deck_center + dir_x * 100).astype(np.int32)
        y_end = (deck_center + dir_y * 100).astype(np.int32)
        cv2.arrowedLine(img_display, tuple(deck_center), tuple(x_end), (0, 0, 255), 2, tipLength=0.1)
        cv2.arrowedLine(img_display, tuple(deck_center), tuple(y_end), (0, 255, 0), 2, tipLength=0.1)
        cv2.putText(img_display, "X (Deck)", tuple(x_end + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(img_display, "Y (Deck)", tuple(y_end + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    except Exception as e:
        print(f" 甲板检测失败：{str(e)}")

    # 微调后YOLO检测直升机
    aircraft_box, aircraft_center, conf = detect_helicopter_with_finetuned_yolo(img, conf_threshold=0.3)
    if aircraft_box is not None and aircraft_center is not None:
        # 绘制直升机外接矩形（红色粗线）
        cv2.drawContours(img_display, [aircraft_box], 0, (0, 0, 255), 3)
        # 绘制中心圆点（黄色）
        cv2.circle(img_display, aircraft_center, 6, (0, 255, 255), -1)
        # 绘制机身朝向箭头（蓝色粗线）
        rect = cv2.minAreaRect(aircraft_box)
        (cx, cy), (rw, rh), angle = rect
        if rw > rh:
            angle_rad = math.radians(angle)
        else:
            angle_rad = math.radians(angle + 90)
        dir_x = math.cos(angle_rad)
        dir_y = math.sin(angle_rad)
        if dir_y < 0:
            dir_x = -dir_x
            dir_y = -dir_y
        arrow_end = (int(aircraft_center[0] + dir_x * 120), int(aircraft_center[1] + dir_y * 120))
        cv2.arrowedLine(img_display, aircraft_center, arrow_end, (255, 0, 0), 3, tipLength=0.2)
        cv2.putText(img_display, "Helicopter Forward", (arrow_end[0] + 10, arrow_end[1] + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        # 标注置信度
        cv2.putText(img_display, f"Conf: {conf:.2f}", (aircraft_box[0][0], aircraft_box[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 保存+显示
    save_path = "../result/finetuned_yolo_helicopter_result.png"
    cv2.imwrite(save_path, img_display)
    print(f"\n 结果已保存：{save_path}")
    # 自适应窗口
    cv2.namedWindow("Finetuned YOLOv8 Helicopter Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Finetuned YOLOv8 Helicopter Detection", 900, 700)
    cv2.imshow("Finetuned YOLOv8 Helicopter Detection", img_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img_display


# 运行入口
if __name__ == "__main__":
    # 测试图片路径
    TEST_IMAGE_PATH = "../test_image/helicopter_test2.png"
    try:
        helicopter_pose_analysis_finetuned(TEST_IMAGE_PATH)
    except Exception as e:
        print(f" 执行错误：{str(e)}")