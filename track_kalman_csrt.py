import cv2
import sys
import numpy as np
import os

global isTracking
global bbox
global ok
global img2
global tracker
global frame

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

# 初始化卡尔曼滤波器
def create_kalman():
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
    kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    return kalman

def create_tracker(tracker_type):
    if tracker_type == 'BOOSTING':
        return cv2.legacy.TrackerBoosting_create()
    elif tracker_type == 'MIL':
        return cv2.legacy.TrackerMIL_create()
    elif tracker_type == 'KCF':
        return cv2.legacy.TrackerKCF_create()
    elif tracker_type == 'TLD':
        return cv2.legacy.TrackerTLD_create()
    elif tracker_type == 'MEDIANFLOW':
        return cv2.legacy.TrackerMedianFlow_create()
    elif tracker_type == 'MOSSE':
        return cv2.legacy.TrackerMOSSE_create()
    elif tracker_type == 'CSRT':
        return cv2.legacy.TrackerCSRT_create()
    else:
        raise ValueError("Unknown tracker type")

def on_mouse(event, x, y, flags, param):
    global img2, point1, point2, isTracking, bbox, tracker, frame, kalman
    if event == cv2.EVENT_LBUTTONDOWN:
        point1 = (x, y)
        img2 = frame.copy()
        cv2.circle(img2, point1, 10, (0, 255, 0), 5)
        cv2.imshow('tracking', img2)
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
        img2 = frame.copy()
        cv2.rectangle(img2, point1, (x, y), (255, 0, 0), thickness=2)
        cv2.imshow('tracking', img2)
    elif event == cv2.EVENT_LBUTTONUP:
        point2 = (x, y)
        img2 = frame.copy()
        cv2.rectangle(img2, point1, point2, (0, 0, 255), thickness=2)
        cv2.imshow('tracking', img2)
        if point1 != point2:
            min_x = min(point1[0], point2[0])
            min_y = min(point1[1], point2[1])
            width = abs(point1[0] - point2[0])
            height = abs(point1[1] - point2[1])
            bbox = (min_x, min_y, width, height)
            # 重新创建追踪器和卡尔曼滤波器
            tracker = create_tracker(tracker_type)
            tracker.init(frame, bbox)
            kalman = create_kalman()
            # 初始化卡尔曼状态
            cx = min_x + width / 2
            cy = min_y + height / 2
            kalman.statePre = np.array([[cx], [cy], [0], [0]], np.float32)
            isTracking = True

def get_next_filename(base_path, prefix='kalman', ext='.mp4'):
    idx = 1
    while True:
        filename = f"{prefix}_{idx}{ext}"
        full_path = os.path.join(base_path, filename)
        if not os.path.exists(full_path):
            return full_path
        idx += 1

if __name__ == '__main__':

    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    # 推荐使用CSRT或KCF
    tracker_type = 'CSRT'

    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        tracker = create_tracker(tracker_type)

    video = cv2.VideoCapture(r"A:\Project\YOLO_base\testvscode\YOLO11-deploy-customize-your-own-functions\kalman-filter-in-single-object-tracking-main\kalman-filter-in-single-object-tracking-main\data\testvideo1.mp4")
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    base_path = r"A:\Project\YOLO_base\testvscode\YOLO11-deploy-customize-your_own-functions"
    out_path = get_next_filename(base_path, prefix='kalman', ext='.mp4')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    print(f"Saving result to: {out_path}")

    isTracking = False
    kalman = create_kalman()
    lost_count = 0
    max_lost = 30  # 最大丢失帧数

    cv2.namedWindow('tracking')
    cv2.setMouseCallback('tracking', on_mouse)

    ok, frame = video.read()
    if not ok:
        print("could not open the video")
        sys.exit()
    img2 = frame.copy()
    cv2.imshow('tracking', img2)

    while not isTracking:
        cv2.imshow('tracking', img2)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            video.release()
            cv2.destroyAllWindows()
            sys.exit()

    while True:
        ok, frame = video.read()
        if not ok:
            break

        timer = cv2.getTickCount()

        if isTracking:
            ok, bbox = tracker.update(frame)
            if ok:
                cx = bbox[0] + bbox[2] / 2
                cy = bbox[1] + bbox[3] / 2
                measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
                kalman.correct(measurement)
                lost_count = 0
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
                cv2.putText(frame, "Tracking", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            else:
                lost_count += 1
                prediction = kalman.predict()
                pred_x = int(prediction[0] - bbox[2] / 2)
                pred_y = int(prediction[1] - bbox[3] / 2)
                pred_w = int(bbox[2])
                pred_h = int(bbox[3])
                # 限制预测框在画面内
                pred_x = max(0, min(pred_x, width - pred_w))
                pred_y = max(0, min(pred_y, height - pred_h))
                cv2.rectangle(frame, (pred_x, pred_y), (pred_x + pred_w, pred_y + pred_h), (0, 255, 255), 2)
                cv2.putText(frame, "Tracking failure, Kalman predict", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
                if lost_count > max_lost:
                    print("Re-initializing tracker using Kalman prediction.")
                    tracker = create_tracker(tracker_type)
                    bbox = (pred_x, pred_y, pred_w, pred_h)
                    tracker.init(frame, bbox)
                    # 重置卡尔曼状态
                    kalman = create_kalman()
                    kalman.statePre = np.array([[pred_x + pred_w/2], [pred_y + pred_h/2], [0], [0]], np.float32)
                    lost_count = 0

            fps_disp = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
            cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
            cv2.putText(frame, "FPS : " + str(int(fps_disp)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        cv2.imshow("tracking", frame)

        out.write(frame)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

    video.release()
    out.release()
    cv2.destroyAllWindows()