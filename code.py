import cv2
import numpy as np
#import matplotlib.pyplot as plt

#img = cv2.imread('test_image.jpg')
#lane_img = np.copy(img)


def canny(img):
    #gray = cv2.cvtColor(lane_img, cv2.COLOR_RGB2GRAY)
    #blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(img, 50, 150)
    return canny

def region_of_interest(img):
    height = img.shape[0] #top first index is 0
    polygons = np.array([[(200,height),(1100,height),(550,250)]]) #an array of polygons
    mask = np.zeros_like(img)
    cv2.fillPoly(mask,polygons,255) #color of polygon 255 white; polygons cuz fillpoly takes an array of polygons
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def make_coordinates(img, line_parameters):
    slope, intercept = line_parameters
    y1 = img.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1,y1,x2,y2])

def average_slope_intercepts(img, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2), (y1,y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis = 0)
    right_fit_average = np.average(right_fit, axis = 0)
    left_line = make_coordinates(img, left_fit_average)
    right_line = make_coordinates(img, right_fit_average)
    return np.array([left_line, right_line])
    

def display_lines(img, lines):
    line_img = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_img, (x1,y1), (x2,y2), (0,0,255), 10)
    return line_img

'''
canny_img = canny(lane_img)
cropped = region_of_interest(canny_img)
lines = cv2.HoughLinesP(cropped, 2, np.pi/180, 100, np.array([]), minLineLength = 40, maxLineGap=5)
averaged_lines = average_slope_intercepts(lane_img, lines)
line_img = display_lines(lane_img, averaged_lines)
combo_img = cv2.addWeighted(lane_img, 0.8, line_img, 1, 1)
cv2.imshow("result", combo_img)
'''



#plt.imshow(canny)
#plt.show()

cap = cv2.VideoCapture("test.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    canny_img = canny(frame)
    cropped = region_of_interest(canny_img)
    lines = cv2.HoughLinesP(cropped, 2, np.pi/180, 100, np.array([]), minLineLength = 40, maxLineGap=5)
    averaged_lines = average_slope_intercepts(frame, lines)
    line_img = display_lines(frame, averaged_lines)
    combo_img = cv2.addWeighted(frame, 0.8, line_img, 1, 1)
    cv2.imshow("result", combo_img)
    k = cv2.waitKey(5) & 0xFF #press escape to close the video
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
#cv2.waitKey(0)

