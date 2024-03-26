import supervision as sv
from ultralytics import YOLO
import cv2
import numpy as np

from tqdm import tqdm
from supervision.assets import VideoAssets, download_assets
from collections import defaultdict, deque

from supervision.detection.tools.polygon_zone import  PolygonZoneAnnotator, PolygonZone 

SOURCE_VIDEO_PATH = "vehicles.mp4"
TARGET_VIDEO_PATH = "vehicles-result.mp4"
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5
MODEL_NAME = "yolov8x.pt"
MODEL_RESOLUTION = 1280

# POLY_COORDS = np.array([
#     [1252, 787],
#     [2298, 803],
#     [5039, 2159],
#     [-550, 2159]
# ])

POLY_COORDS = np.array([
    [[1252, 787],  #LB
     [1800, 803],  #RB
     [2500, 2159],  #RO
     [-550, 2159]],  #LO
    
    [[1800, 803],
     [2298, 803],
     [5039, 2159],
     [2500, 2159]]
])

num_polys = len(POLY_COORDS)

TARGET_WIDTH = 25
TARGET_HEIGHT = 250

TARGET = np.array([
    [0, 0],
    [TARGET_WIDTH - 1, 0],
    [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
    [0, TARGET_HEIGHT - 1],
])

frame_generator = sv.get_video_frames_generator(source_path=SOURCE_VIDEO_PATH)
frame_iterator = iter(frame_generator)
frame = next(frame_iterator)

annotated_frame = frame.copy()
annotated_frame = sv.draw_polygon(scene=annotated_frame, polygon=POLY_COORDS[0], color=sv.Color.RED, thickness=4)
annotated_frame = sv.draw_polygon(scene=annotated_frame, polygon=POLY_COORDS[1], color=sv.Color.BLUE, thickness=4)
sv.plot_image(annotated_frame)

class ViewTransformer:

    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

# view_transformer = ViewTransformer(source=POLY_COORDS[0], target=TARGET)
###############################################################################################################

model = YOLO(MODEL_NAME)

video_info = sv.VideoInfo.from_video_path(video_path=SOURCE_VIDEO_PATH)
frame_generator = sv.get_video_frames_generator(source_path=SOURCE_VIDEO_PATH)

# tracer initiation
byte_track = sv.ByteTrack(
    frame_rate=video_info.fps, track_thresh=CONFIDENCE_THRESHOLD
)

# annotators configuration
thickness = sv.calculate_dynamic_line_thickness(
    resolution_wh=video_info.resolution_wh
)
text_scale = sv.calculate_dynamic_text_scale(
    resolution_wh=video_info.resolution_wh
)
bounding_box_annotator = sv.BoundingBoxAnnotator(
    thickness=thickness
)
label_annotator = sv.LabelAnnotator(
    text_scale=text_scale,
    text_thickness=thickness,
    text_position=sv.Position.BOTTOM_CENTER
)
trace_annotator = sv.TraceAnnotator(
    thickness=thickness,
    trace_length=video_info.fps * 2,
    position=sv.Position.BOTTOM_CENTER
)

# polygon_zone = PolygonZone(
#     polygon=POLY_COORDS[0],
#     frame_resolution_wh=video_info.resolution_wh
# )

polygon_annotator = PolygonZoneAnnotator(
    zone=POLY_COORDS[0], 
    color=sv.Color.red(), 
    text_thickness=4, 
    text_scale=2
)

coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))

num_detections = np.zeros(num_polys, dtype=int)

# loop over source video frame
for frame in tqdm(frame_generator, total=video_info.total_frames):

    result = model(frame, imgsz=MODEL_RESOLUTION, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(result)

    # filter out detections by class and confidence
    detections = detections[detections.confidence > CONFIDENCE_THRESHOLD]
    detections = detections[detections.class_id != 0]

    ogdetections = detections

    # filter out detections outside the zones     
    for i in range(num_polys):
        polygon_zone = PolygonZone(
        polygon=POLY_COORDS[i],
        frame_resolution_wh=video_info.resolution_wh
        )
        if (i == 0):
            bollean_arr = polygon_zone.trigger(ogdetections) 
            detections = ogdetections[bollean_arr]  
        else:  
            bollean_arr = bollean_arr | polygon_zone.trigger(ogdetections)
            detections = ogdetections[bollean_arr] 
        num_detections[i] = polygon_zone.current_count 

    # print(f"Num detections: {polygon_zone.current_count}")

    # refine detections using non-max suppression
    detections = detections.with_nms(IOU_THRESHOLD)

    # pass detection through the tracker
    detections = byte_track.update_with_detections(detections=detections)

    points = detections.get_anchors_coordinates(
        anchor=sv.Position.BOTTOM_CENTER
    )

    # calculate the detections position inside the target RoI
    view_transformer = ViewTransformer(source=POLY_COORDS[i], target=TARGET)
    points = view_transformer.transform_points(points=points).astype(int)

    # store detections position
    for tracker_id, [_, y] in zip(detections.tracker_id, points):
        coordinates[tracker_id].append(y)

    # format labels
    labels = []

    for tracker_id in detections.tracker_id:
        if len(coordinates[tracker_id]) < video_info.fps / 2:
            labels.append(f"#{tracker_id}")
        else:
            # calculate speed
            coordinate_start = coordinates[tracker_id][-1]
            coordinate_end = coordinates[tracker_id][0]
            distance = abs(coordinate_start - coordinate_end)
            time = len(coordinates[tracker_id]) / video_info.fps
            speed = distance / time * 3.6
            labels.append(f"#{tracker_id} {int(speed)} km/h")

    # annotate frame
    annotated_frame = frame.copy()
    annotated_frame = trace_annotator.annotate(
        scene=annotated_frame, detections=detections
    )
    annotated_frame = bounding_box_annotator.annotate(
        scene=annotated_frame, detections=detections
    )
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, detections=detections, labels=labels
    )
    # line_counter.update(detections=detections)
        # annotate and display frame
    # frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)

    for i in range(num_polys):
        if (i == 0):
            color = sv.Color.RED
        else:
            color = sv.Color.BLUE
        polygon_zone = PolygonZone(
        polygon=POLY_COORDS[i],
        frame_resolution_wh=video_info.resolution_wh
        )
        polygon_annotator = PolygonZoneAnnotator(
        zone=POLY_COORDS[i], 
        color=color, 
        text_thickness=4, 
        text_scale=2
        )
        annotated_frame = polygon_annotator.annotate(
            scene=annotated_frame, label = f"{num_detections[i]}")

    imS = cv2.resize(annotated_frame, (960, 540))
    cv2.imshow("Processed Video", imS)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()