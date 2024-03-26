import supervision as sv
from ultralytics import YOLO
import cv2

model = YOLO("yolov8x.pt")
frame_generator = sv.get_video_frames_generator('vehicles.mp4')
bounding_box_annotator = sv.BoundingBoxAnnotator()

for frame in frame_generator:
    result = model(frame)[0]
    detections = sv.Detections.from_ultralytics(result)

    annotated_frame = frame.copy()
    annotated_frame = bounding_box_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
    
    imS = cv2.resize(annotated_frame, (960, 540))
    cv2.imshow("Processed Video", imS)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()