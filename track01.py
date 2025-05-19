# from ultralytics import YOLO

# # Load the model and run the tracker with a custom configuration file
# model = YOLO("yolo11n.pt")
# results = model.track(source= r"A:\Project\classification\YOLO_dota\kalman-filter-in-single-object-tracking-main\kalman-filter-in-single-object-tracking-main\data\testvideo1.mp4", conf=0.3, iou=0.5, show=True)

from ultralytics import YOLO
import cv2
import numpy as np

class TargetSelector:
    def __init__(self):
        self.selected_id = None
        self.detections = []
        self.frame = None
    
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for det in self.detections:
                x1, y1, x2, y2, id, conf, cls = det
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self.selected_id = int(id)
                    print(f"Selected target ID: {self.selected_id}")
                    break
    
    def select_target(self, frame, detections):
        self.frame = frame.copy()
        self.detections = detections
        self.selected_id = None
        
        # Draw all detections with IDs
        for det in detections:
            x1, y1, x2, y2, id, conf, cls = det
            color = (0, 255, 0)  # Green for unselected
            cv2.rectangle(self.frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f"ID: {int(id)}"
            cv2.putText(self.frame, label, (int(x1), int(y1)-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        cv2.namedWindow("Select Target (Click on object)")
        cv2.setMouseCallback("Select Target (Click on object)", self.mouse_callback)
        
        while self.selected_id is None:
            cv2.imshow("Select Target (Click on object)", self.frame)
            if cv2.waitKey(100) & 0xFF == 27:  # ESC to cancel
                break
        
        cv2.destroyWindow("Select Target (Click on object)")
        return self.selected_id

def track_with_mouse_selection():
    # Load the model
    model = YOLO("yolo11n.pt")
    
    # Open video source
    video_path = r"A:\Project\classification\YOLO_dota\kalman-filter-in-single-object-tracking-main\kalman-filter-in-single-object-tracking-main\data\testvideo1.mp4"
    cap = cv2.VideoCapture(video_path)
    
    # Get first frame for selection
    ret, frame = cap.read()
    if not ret:
        print("Error reading video")
        return
    
    # Get initial detections
    results = model.track(frame, persist=True, conf=0.3, iou=0.5)
    
    # Extract detections from first frame
    detections = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else []
        confs = result.boxes.conf.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy()
        
        for i, box in enumerate(boxes):
            if i < len(track_ids):
                detections.append((*box, track_ids[i], confs[i], clss[i]))
    
    # Let user select target by clicking
    selector = TargetSelector()
    target_id = selector.select_target(frame, detections)
    
    if target_id is None:
        print("No target selected")
        return
    
    # Now process full video with selected target
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Track objects
        results = model.track(frame, persist=True, conf=0.3, iou=0.5)
        
        # Process results
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else []
            confs = result.boxes.conf.cpu().numpy()
            clss = result.boxes.cls.cpu().numpy()
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                cls_id = int(clss[i])
                conf = float(confs[i])
                
                if i < len(track_ids):
                    current_id = int(track_ids[i])
                    
                    # Highlight the selected target
                    if current_id == target_id:
                        color = (0, 0, 255)  # Red for selected target
                        thickness = 3
                        label = f"TARGET {current_id} {conf:.2f}"
                    else:
                        color = (0, 255, 0)  # Green for other objects
                        thickness = 1
                        label = f"{current_id} {conf:.2f}"
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                    cv2.putText(frame, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    track_with_mouse_selection()