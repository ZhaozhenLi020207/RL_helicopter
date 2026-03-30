import cv2
import numpy as np
import math


def euler_from_rotation_matrix(R):
    """从旋转矩阵计算欧拉角"""
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
    """鲁棒提取甲板角点 + 轮廓检测"""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    edges = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 11, 2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = (h * w) * 0.01
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]

    if not valid_contours:
        raise ValueError("未检测到有效轮廓（甲板/飞机）")

    valid_contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)
    deck_contour = valid_contours[0]
    rect = cv2.minAreaRect(deck_contour)
    box = cv2.boxPoints(rect)
    img_pts = np.array(box, dtype=np.float32)
    img_pts = img_pts[np.lexsort((img_pts[:, 0], img_pts[:, 1]))]

    return img_pts, valid_contours


def calculate_aircraft_direction_and_box(img, valid_contours):
    """计算飞机朝向 + 最小外接矩形"""
    h, w = img.shape[:2]
    aircraft_contour = None
    for cnt in valid_contours:
        area = cv2.contourArea(cnt)
        if area < cv2.contourArea(valid_contours[0]) * 0.9:
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (rw, rh), angle = rect
            aspect_ratio = max(rw, rh) / min(rw, rh) if min(rw, rh) > 0 else 0
            if aspect_ratio > 1.5:
                aircraft_contour = cnt
                break

    if aircraft_contour is None:
        return np.array([0.2, 0.98]), np.array([w // 2, h // 2]), None

    # 计算飞机最小外接矩形
    aircraft_rect = cv2.minAreaRect(aircraft_contour)
    aircraft_box = cv2.boxPoints(aircraft_rect)
    aircraft_box = np.intp(aircraft_box)

    # PCA计算朝向
    pts = np.array(aircraft_contour).reshape(-1, 2).astype(np.float32)
    mean, eigenvectors = cv2.PCACompute(pts, mean=None)
    dir_vec = eigenvectors[0]
    aircraft_dir = np.array([dir_vec[0], dir_vec[1]], dtype=np.float32)

    M = cv2.moments(aircraft_contour)
    aircraft_center_x = int(M["m10"] / M["m00"])
    aircraft_center_y = int(M["m01"] / M["m00"])
    aircraft_center = np.array([aircraft_center_x, aircraft_center_y])

    end_pt = mean + 50 * dir_vec
    if end_pt[1] < aircraft_center_y:
        aircraft_dir = -aircraft_dir

    aircraft_dir = aircraft_dir / np.linalg.norm(aircraft_dir)
    return aircraft_dir, aircraft_center, aircraft_box


def helicopter_deck_pose_estimation(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片：{image_path}")
    h, w = img.shape[:2]

    fx = w * 1.2
    fy = h * 1.2
    cx = w / 2.0
    cy = h / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    dist = np.zeros((5,), dtype=np.float32)

    img_pts, valid_contours = detect_deck_corners_robust(img)
    aircraft_dir, aircraft_center, aircraft_box = calculate_aircraft_direction_and_box(img, valid_contours)

    print(f"飞机中心坐标：{aircraft_center}")
    print(f"自动计算的飞机朝向向量：{aircraft_dir.round(4)}")

    obj_pts = np.array([
        [-0.5, -0.5, 0.0],
        [0.5, -0.5, 0.0],
        [0.5, 0.5, 0.0],
        [-0.5, 0.5, 0.0]
    ], dtype=np.float32)
    success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_EPNP)
    if not success:
        raise RuntimeError("PnP 求解失败")

    R, _ = cv2.Rodrigues(rvec)
    euler = euler_from_rotation_matrix(R)
    roll, pitch, yaw = np.degrees(euler)

    display_scale = 0.8 if max(w, h) > 1000 else 1.0
    img_display = cv2.resize(img, (int(w * display_scale), int(h * display_scale)))
    img_pts_display = img_pts * display_scale
    aircraft_center_display = (aircraft_center * display_scale).astype(np.int32)

    for (x, y) in img_pts_display:
        cv2.circle(img_display, (int(x), int(y)), 8, (0, 0, 255), -1)
    cv2.drawContours(img_display, [img_pts_display.astype(np.int32)], -1, (0, 255, 0), 3)

    # 绘制飞机最小外接矩形（黄色框）
    if aircraft_box is not None:
        aircraft_box_display = (aircraft_box * display_scale).astype(np.int32)
        cv2.drawContours(img_display, [aircraft_box_display], 0, (0, 255, 255), 2)  # 黄色框

    deck_center = np.mean(img_pts_display, axis=0).astype(np.int32)
    dir_x = (img_pts_display[1] - img_pts_display[0]).astype(np.float32)
    dir_x = dir_x / np.linalg.norm(dir_x)
    dir_y = np.array([-dir_x[1], dir_x[0]], dtype=np.float32)
    x_end = (deck_center + dir_x * 100).astype(np.int32)
    y_end = (deck_center + dir_y * 100).astype(np.int32)

    cv2.arrowedLine(img_display, tuple(deck_center), tuple(x_end), (0, 0, 255), 3, tipLength=0.1)
    cv2.arrowedLine(img_display, tuple(deck_center), tuple(y_end), (0, 255, 0), 3, tipLength=0.1)

    aircraft_end = (aircraft_center_display + aircraft_dir * 150).astype(np.int32)
    cv2.arrowedLine(img_display, tuple(aircraft_center_display), tuple(aircraft_end),
                    (255, 0, 0), 4, tipLength=0.15)
    cv2.circle(img_display, tuple(aircraft_center_display), 5, (0, 255, 255), -1)

    cv2.putText(img_display, "X (Deck)", tuple(x_end + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img_display, "Y (Deck)", tuple(y_end + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(img_display, "Aircraft Forward", tuple(aircraft_end + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    print("\n 直升机甲板位姿结果")
    print(f"旋转矩阵 R：\n{R.round(4)}")
    print(f"平移向量 t (米)：\n{tvec.round(4)}")
    print(f"欧拉角：Roll={roll:.2f}°, Pitch={pitch:.2f}°, Yaw={yaw:.2f}°")

    cv2.imwrite("result/result_with_aircraft_box.png", img_display)
    cv2.imshow("Result (With Aircraft Box)", img_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return R, tvec, euler


if __name__ == "__main__":
    image_path = "test_image/helicopter_test2.png"  # 替换图片路径
    try:
        helicopter_deck_pose_estimation(image_path)
    except Exception as e:
        print("错误：", e)