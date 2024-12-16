# Traffic Light Recognition

## Description
A simple computer vision project based on Python, utilizing OpenCV for most of the video processing, frame handling, and image processing logic (HSV method).

## Objective
How can the colors of traffic lights be identified solely using the HSV method while maintaining stable and reliable predictions?

## Program Flow
1. The program begins by defining HSV color ranges for detecting red, yellow, and green representing traffic light colors. Several utility functions are prepared, including enhancing saturation (increase_saturation), a non-maximum suppression method to filter overlapping bounding boxes (non_maximum_suppression), a traffic light detection function (detect_traffic_light), and a logging function (log_detection) to record detected color changes everytime the color changes.

2. Program loads a video file (input.mp4). Initializing a log file (traffic_light_log.txt) and defines two regions of interest (ROIs) for each traffic light area. Video output properties such as resolution and codec are set for the output needings (output.mp4).

3. Within the loop for each frame, the program doubles the resolution for better visibility and processes each ROI. For every ROI, the saturation is enhanced and converted to HSV format. Traffic light detection is performed using HSV masks to identify bounding boxes and colors. The detected colors are written on the frame and any color changes in traffic light are logged. The processed ROI is then re-integrated into the main frame.

4. The program writes the processed frames to the output video file and displays them in a live preview window. Program allows for user interruption ('q' key) to terminate. Finally, all processes such as video capture and display windows, are properly released to ensure a clean exit.

## Future Enhancements
1. **Dynamic Color Ranges**  
   Currently, the `color_ranges` are static, which may result in suboptimal performance when applied to different scenarios with varying conditions. Dynamic adaptation of these ranges could improve robustness.

2. **Additional Color Identification Methods**  
   The current implementation relies solely on contours to assist HSV-based color detection. Incorporating additional variables like edge masks and circle detection could enhance prediction accuracy.

3. **Traffic Light Localization**  
   The system currently uses hardcoded ROI (Region of Interest) coordinates to identify traffic lights. This approach is less effective when the camera is not stationary. Implementing algorithms that can dynamically track traffic lights would make the system more flexible and capable of handling a broader range of conditions.

## Input and Output
### Input: [Link](https://drive.google.com/file/d/1-SKPkE74Xq8BtRSePhprVGyY2TUbi3in/view?usp=sharing)
### Output: [Link](https://drive.google.com/file/d/1-mEzTlYOO3tCcXRRk1kyzRQQUp62e42k/view?usp=sharing)
