import sys
import math
import cv2 as cv
import numpy as np

# class Image2TimeConverter:
def line_length(line):
    x1, y1, x2, y2 = line
    point1 = np.array((x1, y1))
    point2 = np.array((x2, y2))
    d = np.linalg.norm([point1 - point2])  # Euclidean distance
    print(d)
    return d

def convert_angle_to_time(hour_angle, minute_angle, second_angle):
    hour = int(hour_angle*12 / 360)
    minute = int(minute_angle*60 / 360)
    second = int(second_angle*60 / 360)
    return hour, minute, second

def get_angle(x1, y1, x2, y2):
    """Calculate the angle between two points and the positive x-axis"""
    # Use atan2 to compute the angle between the points
    angle_radians = math.atan2(y2 - y1, x2 - x1)  # atan2 gives the angle in radians
    angle_degrees = math.degrees(angle_radians)  # Convert radians to degrees
    angle_degrees = (90 + angle_degrees) % 360
    return angle_degrees

def get_lines_in_image(img):
    src = cv.imread(cv.samples.findFile(img), cv.IMREAD_GRAYSCALE)
    
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image!')
        print ('Usage: hough_lines.py [image_name -- default ' + default_file + '] \n')
        return []
    dst = cv.Canny(src, 50, 200, None, 3)
    
    # Copy edges to the images that will display the results in BGR
    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)
    cdstPnodoublelines = np.copy(cdst)
    lines = cv.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
    
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv.line(cdst, pt1, pt2, (0,0,255), 3, cv.LINE_AA)
    
    
    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
    sorted_lines = np.array(sorted(linesP, key=lambda line: line_length(line[0]), reverse=True), dtype=float)

    linesP_noDoubles=[]
    if linesP is not None and len(linesP) > 4:
        for i in range(0, len(linesP[:7])):
            l = linesP[i][0]
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
            if (i%2 == 0):
                cv.line(cdstPnodoublelines, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
                linesP_noDoubles.append(linesP[i])
                
    # 4debug:            
    cv.imshow("Source", src)
    cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
    cv.imshow("Detected Lines (in red) - Probabilistic Line Transform no double lines", cdstPnodoublelines)
    
    cv.waitKey()
     
    return lines, linesP_noDoubles

def get_time_from_clock_image(filename='clknobound.png'):
    if filename=='':
        return -1
        
    lines, linesP = get_lines_in_image(filename)
              
    if len(lines) < 3: # not enough lines in image
        return -1
    hands=[]
        
    for line in linesP[:3]:
        x1, y1, x2, y2 = line[0]
        angle = get_angle(x1, y1, x2, y2)
        length = line_length(line[0])
        hands.append((x1, y1, x2, y2, angle, length))
        
    second_hand = hands[0]
    minute_hand = hands[1]
    hour_hand = hands[2]
    
    hour, minute, second = convert_angle_to_time(hour_hand[4], minute_hand[4], second_hand[4])
    
    print(f'{hour}:{minute}:{second}')
print(get_time_from_clock_image('Figure 2025-01-19 095253.png'))