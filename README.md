# Detection of lane lines on a road using OpenCV
Employed the use of a popular and powerful computer vision library to detect and identify the presence of lane lines on the video of a road.
The tools used here are pyhton and OpenCV. 

The approach was to maximize the performance using the HoughLinesP() function and later averaging out a part of the lines on both left and right side to obtain singular values for both. Using these, straight lines were drawn to fit the lane lines with maximum efficiency.

The result, well, as expected was a near perfect detection of lane lines on a road.
