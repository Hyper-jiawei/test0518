import cv2
import sys
import os

global isTracking
global bbox
global ok
global img2

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

def on_mouse(event, x, y, flags, param):
    global img2, point1, point2, g_rect, isTracking
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击,则在原图打点
        point1 = (x, y)
        cv2.circle(img2, point1, 10, (0, 255, 0), 5)
        cv2.imshow('Tracking', img2)

    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):  # 按住左键拖曳，画框
        cv2.rectangle(img2, point1, (x, y), (255, 0, 0), thickness=2)
        cv2.imshow('Tracking', img2)

    elif event == cv2.EVENT_LBUTTONUP:  # 左键释放，显示
        point2 = (x, y)
        cv2.rectangle(img2, point1, point2, (0, 0, 255), thickness=2)
        cv2.imshow('Tracking', img2)
        if point1 != point2:
            min_x = min(point1[0], point2[0])
            min_y = min(point1[1], point2[1])
            width = abs(point1[0] - point2[0])
            height = abs(point1[1] - point2[1])

            # 定义一个初始边界框
            bbox = (min_x, min_y, width, height)
            # 用第一帧和包围框初始化跟踪器
            tracker.init(frame, bbox)
            isTracking=True

def get_next_filename(base_path, prefix='kalman', ext='.mp4'):
    idx = 1
    while True:
        filename = f"{prefix}_{idx}{ext}"
        full_path = os.path.join(base_path, filename)
        if not os.path.exists(full_path):
            return full_path
        idx += 1

if __name__ == '__main__':

    # 建立追踪器
    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    tracker_type = tracker_types[6]

    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.legacy.TrackerBoosting_create()
        elif tracker_type == 'MIL':
            tracker = cv2.legacy.TrackerMIL_create()
        elif tracker_type == 'KCF':
            tracker = cv2.legacy.TrackerKCF_create()
        elif tracker_type == 'TLD':
            tracker = cv2.legacy.TrackerTLD_create() #动态追踪，可调整追踪框大小
        elif tracker_type == 'MEDIANFLOW':
            tracker = cv2.legacy.TrackerMedianFlow_create() # #动态追踪，可调整追踪框大小更高帧率
        elif tracker_type == 'MOSSE':
            tracker = cv2.legacy.TrackerMOSSE_create()
        elif tracker_type == 'CSRT':
            tracker = cv2.legacy.TrackerCSRT_create()


    # 读取视频
    video = cv2.VideoCapture(r"A:\Project\YOLO_base\testvscode\YOLO11-deploy-customize-your-own-functions\kalman-filter-in-single-object-tracking-main\kalman-filter-in-single-object-tracking-main\data\testvideo1.mp4")

    # 如果视频没有打开，退出。
    if not video.isOpened():
        print("Could not open video")
        sys.exit()


     # 获取视频参数用于保存
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或 'XVID'
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    base_path = r"A:\Project\YOLO_base\testvscode\YOLO11-deploy-customize-your-own-functions"
    out_path = get_next_filename(base_path, prefix='1', ext='.mp4')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    print(f"Saving result to: {out_path}")


    isTracking=False

    cv2.namedWindow('tracking')
    cv2.setMouseCallback('tracking', on_mouse)

    ok, frame = video.read()
    if not ok:
        print ("could not open the video")
        sys.exit
    img2 = frame.copy()
    cv2.imshow('tracking',img2)

    while not isTracking:
        cv2.imshow('tracking',img2)
        k = cv2.waitKey(1) & 0xff
        if k ==27:
            video.release()
            cv2.destroyAllWindows()
            sys.exit()
    #go to mian while after chose the target 
    while True:
        
        # 读取一个新的帧
        ok, frame = video.read()
        if not ok:
            break

        img2=frame

        # 启动计时器
        timer = cv2.getTickCount()

        if isTracking:
            # 更新跟踪器
            ok, bbox = tracker.update(frame)

            # 计算帧率(FPS)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

            # 绘制包围框
            if ok:
                # 跟踪成功
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
            else:
                # 跟踪失败
                cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            # 在帧上显示跟踪器类型名字
            cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

            # 在帧上显示帧率FPS
            cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        # 显示结果
        cv2.imshow("Tracking", frame)

        # 保存帧到输出视频
        out.write(frame)

        # 按ESC键退出
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

video.release()
out.release()
cv2.destroyAllWindows()