import cv2
import time
import numpy as np

color_ranges = {
    'Red': [
        (np.array([0, 120, 70], dtype=np.uint8), np.array([10, 255, 255], dtype=np.uint8)),
        (np.array([170, 120, 70], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8))
    ],
    'Yellow': [
        (np.array([12, 55, 80], dtype=np.uint8), np.array([32, 255, 255], dtype=np.uint8))
    ],
    'Green': [
        (np.array([50, 5, 150], dtype=np.uint8), np.array([86, 250, 250], dtype=np.uint8))
    ]
}

def increase_saturation(frame, saturation_scale=1):
    """Increase the saturation of the frame."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(s * saturation_scale, 0, 255).astype(np.uint8)
    enhanced_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)

def non_maximum_suppression(boxes, overlap_thresh=0.3):
    """
    Perform Non-Maximum Suppression (NMS) to reduce overlapping bounding boxes.
    Args:
        boxes: List of bounding boxes [(x, y, w, h, color)].
        overlap_thresh: Threshold for overlap (IoU) to suppress.
    Returns:
        List of filtered bounding boxes.
    """
    if len(boxes) == 0:
        return []

    # Convert bounding boxes to NumPy array (ensure numeric data types)
    boxes_array = np.array([(int(x), int(y), int(x + w), int(y + h), color) for x, y, w, h, color in boxes], dtype=object)

    # Extract coordinates
    x1 = boxes_array[:, 0].astype(float)
    y1 = boxes_array[:, 1].astype(float)
    x2 = boxes_array[:, 2].astype(float)
    y2 = boxes_array[:, 3].astype(float)

    # Compute area of each bounding box
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.argsort(y2)

    keep = []

    while order.size > 0:
        i = order[-1]
        keep.append(i)

        # Compute IoU with the remaining boxes
        xx1 = np.maximum(x1[i], x1[order[:-1]])
        yy1 = np.maximum(y1[i], y1[order[:-1]])
        xx2 = np.minimum(x2[i], x2[order[:-1]])
        yy2 = np.minimum(y2[i], y2[order[:-1]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        intersection = w * h
        iou = intersection / (areas[i] + areas[order[:-1]] - intersection)

        # Keep boxes with IoU below the threshold
        order = order[np.where(iou <= overlap_thresh)[0]]

    return [boxes[idx] for idx in keep]

def detect_traffic_light(hsv_frame, color_ranges):
    """
    Detect traffic light color in the given frame
    Args:
        hsv_frame: Frame in hsv format color.
    Returns:
        Bounding box with detected color.
    """
    bounding_boxes = []
    detected_colors = []

    for color, ranges in color_ranges.items():
        for lower, upper in ranges:
            mask = cv2.inRange(hsv_frame, lower, upper)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if cv2.contourArea(contour) > 100:
                    x, y, w, h = cv2.boundingRect(contour)
                    bounding_boxes.append((int(x), int(y), int(w), int(h), color))
                    detected_colors.append(color)

    # Apply Non-Maximum Suppression (NMS)
    bounding_boxes = non_maximum_suppression(bounding_boxes, overlap_thresh=0.6)

    # Find the largest bounding box based on area
    if bounding_boxes:
        largest_box = max(bounding_boxes, key=lambda box: box[2] * box[3])
        return [largest_box], detected_colors

    return [], []  # If no bounding boxes

def log_detection(log_file, roi_label, current_color, previous_color, timestamp):
    """Log detected color and ROI to a file if the color has changed."""
    if current_color != previous_color:  # Log only if color changes
        with open(log_file, 'a') as f:
            f.write(f"{timestamp} - {roi_label}: {current_color}\n")

def main():
    video_path = "input.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return
    
    log_file = "traffic_light_log.txt"
    with open(log_file, 'w') as f:
        f.write("Traffic Light Detection Log\n")

    # Define two ROIs
    rois = [
        {"label": "Traffic Light 1", "x": 3100, "y": 1000, "width": 100, "height": 280, "last_detected_color": None},
        {"label": "Traffic Light 2", "x": 2850, "y": 950, "width": 100, "height": 200, "last_detected_color": None}
    ]

    # Get original FPS of the video
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    playback_speed = 4
    wait_time = max(1, int(1000 / (original_fps * playback_speed)))

    # Get video properties for output
    output_path = "output.mp4"
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    out = cv2.VideoWriter(output_path, fourcc, original_fps, (frame_width * 2, frame_height * 2))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

        # Process both ROIs
        for roi in rois:
            x, y, width, height = roi["x"], roi["y"], roi["width"], roi["height"]

            # Draw rectangle around the ROI
            cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2)

            # Crop the ROI
            cropped_roi = frame[y:y + height, x:x + width]

            # Increase saturation in ROI
            saturated_roi = increase_saturation(cropped_roi, saturation_scale=1.5)

            # Convert ROI to HSV
            hsv_roi = cv2.cvtColor(saturated_roi, cv2.COLOR_BGR2HSV)

            # Detect traffic lights in ROI
            bounding_boxes, detected_colors = detect_traffic_light(hsv_roi, color_ranges)

            # Draw bounding boxes and labels in ROI
            for (bx, by, bw, bh, color) in bounding_boxes:
                cv2.rectangle(saturated_roi, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
            
            if detected_colors:
                cv2.putText(frame, color, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                current_color = detected_colors[0]
                # Log detections
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000))
                log_detection(log_file, roi["label"], current_color, roi["last_detected_color"], timestamp)
            
            roi["last_detected_color"] = current_color
            
            # Replace the ROI back into the frame
            frame[y:y + height, x:x + width] = saturated_roi

        # Write processed frame to output video
        out.write(frame)

        # Display the processed frame
        cv2.namedWindow("Traffic Light Detection", cv2.WINDOW_NORMAL)
        cv2.imshow("Traffic Light Detection", frame)

        # Skip frames to achieve playback speed
        for _ in range(playback_speed - 1):
            cap.read()

        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
