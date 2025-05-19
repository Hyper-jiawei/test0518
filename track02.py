from ultralytics import YOLO
import cv2
import numpy as np
from collections import deque
from filterpy.kalman import KalmanFilter

class EnhancedTracker:
    def __init__(self):
        self.model = YOLO("yolo11n.pt")
        self.target_id = None
        self.prev_target_pos = None
        self.tracking_lost_count = 0
        self.max_lost_frames = 10  # Allowed consecutive lost frames
        self.trajectory = deque(maxlen=30)  # Stores last 30 positions
        self.kalman_filter = self.init_kalman_filter()
        self.detections_history = []
        
    def init_kalman_filter(self):
        """Initialize Kalman filter for target tracking"""
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.F = np.array([[1, 0, 1, 0],  # State transition matrix
                         [0, 1, 0, 1],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0],  # Measurement function
                         [0, 1, 0, 0]])
        kf.P *= 1000.  # Covariance matrix
        kf.R = 5  # Measurement noise
        kf.Q = np.eye(4) * 0.1  # Process noise
        return kf
    
    def update_kalman(self, measurement):
        """Update Kalman filter with new measurement"""
        self.kalman_filter.predict()
        self.kalman_filter.update(measurement)
        return self.kalman_filter.x[:2]  # Return predicted position
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for target selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            for det in self.detections_history[-1]:
                x1, y1, x2, y2, id, conf, cls = det
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self.target_id = int(id)
                    self.tracking_lost_count = 0
                    # Reset Kalman filter for new target
                    center = ((x1+x2)/2, (y1+y2)/2)
                    self.kalman_filter.x = np.array([center[0], center[1], 0, 0])
                    self.trajectory.clear()
                    print(f"Switched to tracking target ID: {self.target_id}")
                    break
    
    def draw_trajectory(self, frame):
        """Draw the target's movement trajectory"""
        if len(self.trajectory) > 1:
            points = np.array(self.trajectory, dtype=np.int32)
            cv2.polylines(frame, [points], False,(255, 200, 100), 2)
    
    def process_frame(self, frame):
        """Process a single frame with all tracking logic"""
        # Track objects
        results = self.model.track(frame, persist=True, conf=0.3, iou=0.5)
        
        # Extract detections
        current_detections = []
        target_found = False
        target_box = None
        
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else []
            confs = result.boxes.conf.cpu().numpy()
            clss = result.boxes.cls.cpu().numpy()
            
            for i, box in enumerate(boxes):
                if i < len(track_ids):
                    det = (*box, track_ids[i], confs[i], clss[i])
                    current_detections.append(det)
                    
                    if self.target_id is not None and track_ids[i] == self.target_id:
                        target_found = True
                        target_box = box
        
        self.detections_history.append(current_detections)
        
        # Handle target tracking
        if self.target_id is not None:
            if target_found:
                self.tracking_lost_count = 0
                x1, y1, x2, y2 = target_box
                center = ((x1+x2)/2, (y1+y2)/2)
                self.trajectory.append(center)
                
                # Update Kalman filter
                smoothed_pos = self.update_kalman(np.array(center))
                self.prev_target_pos = smoothed_pos
            else:
                self.tracking_lost_count += 1
                # Use Kalman prediction when target is temporarily lost
                if self.prev_target_pos is not None:
                    self.kalman_filter.predict()
                    smoothed_pos = self.kalman_filter.x[:2]
                    self.trajectory.append(smoothed_pos)
        
        # Draw all detections
        for det in current_detections:
            x1, y1, x2, y2, id, conf, cls = det
            color = (0, 255, 0)  # Green for other objects
            thickness = 1
            
            if self.target_id is not None and id == self.target_id:
                color = (0, 0, 255)  # Red for selected target
                thickness = 3
                label = f"TARGET {int(id)} {conf:.2f}"
                
                # Draw Kalman-predicted position when target is lost
                if not target_found and self.tracking_lost_count < self.max_lost_frames:
                    kx, ky = self.kalman_filter.x[:2]
                    cv2.circle(frame, (int(kx), int(ky)), 8, (255, 255, 0), -1)
                    label = f"PREDICTED {label}"
            else:
                label = f"{int(id)} {conf:.2f}"
            
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
            cv2.putText(frame, label, (int(x1), int(y1)-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw trajectory for current target
        if self.target_id is not None:
            self.draw_trajectory(frame)
        
        # Check if tracking is lost
        if self.tracking_lost_count >= self.max_lost_frames:
            cv2.putText(frame, "TRACKING LOST! Click new target", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame, True
        
        return frame, False
    
    def run(self, video_path):
        cap = cv2.VideoCapture(video_path)
        cv2.namedWindow("Enhanced Tracking")
        cv2.setMouseCallback("Enhanced Tracking", self.mouse_callback)
        
        paused = False
        tracking_lost = False
        
        while cap.isOpened():
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
            
            frame, tracking_lost = self.process_frame(frame)
            
            # Pause if tracking is lost
            if tracking_lost and not paused:
                paused = True
            
            cv2.imshow("Enhanced Tracking", frame)
            
            key = cv2.waitKey(1 if not paused else 0) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # Space to pause/resume
                paused = not paused
            elif key == ord('c') and paused:  # Continue after lost
                paused = False
                tracking_lost = False
                self.tracking_lost_count = 0
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    #video_path = r"A:\Project\classification\YOLO_dota\kalman-filter-in-single-object-tracking-main\kalman-filter-in-single-object-tracking-main\data\testvideo1.mp4"
    video_path = r"A:\BaiduNetdiskDownload\drone-detect\drone-detect\drone_track\test.mp4"
    tracker = EnhancedTracker()
    tracker.run(video_path)