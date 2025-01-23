"""
Created on Fri Jan 17 09:59:50 2025

@author: ayalac

Converts analog clock to time, converts time to analog clock
"""
import os
import sys
import math
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class Line(self, line):
    def __init__(self, line):
        self.p1 = Point(line[0], line[1])
        self.p2 = Point(line[2], line[3])
    
    def get_line_length(self):
        return p1.euclidean_distance(p2)
    
    
class Point: # a point in a 2D plane
    def __init__(self, x,y):
        self.x=x
        self.y=y
        
    # def get_normalized_point(self, width, height): # normalize point according to the plane it's in
    #     x_normalized = self.x / width
    #     y_normalized = self.y / height
    #     return Point(x_normalized, y_normalized)
    
    def euclidean_distance(self, point):
        p1 = np.array((self.x, self.y))
        p2 = np.array((point.x, point.y))
        return np.linalg.norm([p1 - p2])

        
class ClockImageTimeConverter:    
    def __init__(self):
        print("Hello from clock converter constructor!")
        
    def line_length(self, line):
        x1, y1, x2, y2 = line
        point1 = np.array((x1, y1))
        point2 = np.array((x2, y2))
        d = np.linalg.norm([point1 - point2])  # Euclidean distance
        return d
        
    def convert_angle_to_time(self, hour_angle, minute_angle):
        """
        Computes & returns the hour and minute according to the angles of the hands.
    
        Parameters
        ----------
        hour_angle : FLOAT
            Hour angle in degrees (0 to 360).
        minute_angle : FLOAT
            Minute angle in degrees (0 to 360).
    
        Returns
        -------
        hour : INT
            Hour (1-12).
        minute : INT
            Minute (0-59).
        """
        # Calculate minute based on the minute_angle (mapping 0-360 degrees to 0-59 minutes)
        minute = int(minute_angle * 60 / 360)  # Minute is an integer (0 to 59)
    
        # Calculate hour based on the hour_angle (mapping 0-360 degrees to 0-12 hours)
        if minute >= 30:
            hour= int(hour_angle * 12 / 360)
        else:
            hour = round(hour_angle * 12 / 360)  # Hour in range (0 to 12)
        
        if hour == 0:
            hour =12
        return hour, minute

    
    def get_angle_from_center(self, point, center):
        """
        

        Parameters
        ----------
        point : POINT
            The point that we want to compute its angle with the circle's center.
        center : POINT
            The center of the circle (clock in our case).

        Returns
        -------
        angle : FLOAT
            The angle between the point, and the negative Y axis (up). 
            Where the axis center is the center of the circle (clock)
            The axis: Y positive down and X positive right. I wanted angle 0 to be hour 12.

        """
        dx = point.x - center.x
        dy = point.y - center.y
        
        if dx == 0 and dy < 0:
            return 0
        if dx == 0 and dy > 0:
            return 180
        if dy == 0 and dx < 0:
            return 270
        if dy == 0 and dx > 0:
            return 90
        
        # find point's quarter position
        quarter_num=0
        if point.x > center.x and point.y >= center.y:
            quarter_num=1
        elif point.x < center.x and point.y > center.y:
            quarter_num=2
        elif point.x < center.x and point.y < center.y:
            quarter_num=4
        elif point.x > center.x and point.y < center.y:
            quarter_num=3

        angle=180
        
        # Normalize the angle to [0, 360] range clockwise from the negative Y-axis (up)
        if quarter_num==1:
            if dx == 0:
                return 180
            rad = math.atan(dy/ dx)
            angle = math.degrees(rad) 
            return 90+angle
        elif quarter_num==2:
            if dy == 0:
                return 270
            rad = math.atan(dx/ dy)
            angle = -1*math.degrees(rad) 
            return (180+angle)
        elif quarter_num==3:
            if dx ==0:
                return 0
            rad = math.atan(dy/ dx)
            angle = -1*math.degrees(rad) 
            return 90-angle
        elif quarter_num==4:
            if dx ==0:
                0
            rad = math.atan(dy/ dx)
            angle = math.degrees(rad) 
            return 270+angle
        
        return angle
    
    def line_angle(self, line):
        x1, y1, x2, y2 = line
        dx = x2 - x1
        dy = y2 - y1
        angle = math.atan2(dy, dx)  # Angle in radians
        return math.degrees(angle)  # Convert to degrees
    
    def are_angles_close(self, angle1, angle2, angle_threshold=10):
        """Check if the angles are close within a threshold (in degrees)."""
        # Normalize angle difference to be within [0, 180] degrees (shortest distance on a circle)
        diff = abs(angle1 - angle2) % 360  # Ensure the difference is within [0, 360]
        if diff > 180:
            diff = 360 - diff  # Shortest distance on a circle (wraparound)
        return diff < angle_threshold  # Return True if the angles are within the threshold
  
    # Function to remove double lines (lines with similar angles and lengths)
    def remove_double_lines(self, lines, length_threshold=100, angle_threshold=5):
        unique_lines = []
        lines_to_remove = []
        
        lines = np.array(sorted(lines, key=lambda line: self.line_length(line[0]), reverse=True))
        
        if len(lines) == 1:
            return lines
        
        for i, line in enumerate(lines):
            p = self.farthest_point(Point(line[0][0], line[0][1]), Point(line[0][2], line[0][3]),center=Point(472/2, 472/2))
            line_ang = self.get_angle_from_center(point=p, center=Point(472/2, 472/2))
            line_len = self.line_length(line[0])
            is_duplicate = False
            
            for j, unique_line in enumerate(unique_lines):
                p_unique = self.farthest_point(Point(unique_line[0][0], unique_line[0][1]), Point(unique_line[0][2], unique_line[0][3]),center=Point(472/2, 472/2))
                unique_ang = self.get_angle_from_center(point=p_unique, center=Point(472/2, 472/2))
                unique_len = self.line_length(unique_line[0])

                if self.are_angles_close(line_ang, unique_ang, angle_threshold) and abs(line_len-unique_len) < length_threshold :
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_lines.append(line)
        
        return unique_lines
        
    def get_lines_in_image(self, src, debug=0):
        """
        Using the Hough Lines Probablistic algorithm we find the lines inside our image,
        this will help us find the clock hands.
        Prequisite - image most contain only the analog clock's face and has to be of similar format
        of the clock generated in draw_analog_clock method.

        Parameters
        ----------
        src : STRING
           Preprocessed image cropped to clock bounding box.

        Returns
        -------
        list
            List of line vectors in image.

        """
        
        # reshape image to have a constant center coordinate
        src = cv.resize(src, (472, 472)) # clock center (0.5, 0.5) in the normalized coordinates
 
        dst = cv.Canny(src, 200, 300, None, 3)
        cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
        cdstP = np.copy(cdst)
        
        linesP = cv.HoughLinesP(image=dst, rho=1, theta=np.pi / 180, threshold =23, minLineLength =80, maxLineGap =10)
        linesP = np.array(sorted(linesP, key=lambda line: self.line_length(line[0]), reverse=True))
        linesP = self.shift_lines(linesP, Point(472/2, 472/2))
        linesP= self.remove_double_lines(linesP)

        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
                if debug:
                    cv.imshow(f"Detected Lines (in red) {i}- Probabilistic Line Transform no double lines", cdstP)

        if debug:
            cv.waitKey()
        return linesP
    
    def farthest_point(self, p1, p2, center): # compute farthest point from the center
        l1 = self.line_length([p1.x, p1.y, center.x, center.y])
        l2 = self.line_length([p2.x, p2.y, center.x, center.y])
        
        if l1 > l2:
            return p1
        return p2
    
    def shift_lines(self, lines, center_point):
        dy=0
        dx=0
        
        for i, line in zip(range(len(lines)), lines):
            x1, y1, x2, y2 = line[0]
            p1 = Point(x1,y1)
            p2 = Point(x2, y2)
            
            dist_p1 = center_point.euclidean_distance(p1)
            dist_p2 = center_point.euclidean_distance(p2)
            if  dist_p1 < dist_p2:
                shift_x = p1.x - center_point.x
                shift_y = p1.y - center_point.y
            else:
                shift_x = p2.x - center_point.x
                shift_y = p2.y - center_point.y
                
            lines[i][0] = np.array([x1 - shift_x, y1 - shift_y, x2 - shift_x, y2 - shift_y])

        return lines
 
    def get_clock_circle(self, filename, debug=0):
        img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
        assert img is not None, "file could not be read, check with os.path.exists()"
        cimg = cv.cvtColor(img,cv.COLOR_GRAY2BGR) 
        circles = cv.HoughCircles(img,cv.HOUGH_GRADIENT,1,20,
                                    param1=50,param2=30,minRadius=10,maxRadius=0)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            circles = np.array(sorted(circles, key=lambda circle: circle[2], reverse=True))
            
            x, y, radius = circles[0][0]
            
            mask = np.zeros_like(img)
            
            cv.circle(mask, (x,y), radius, (255,255,255), -1)
            
            result = cv.bitwise_and(img, mask)
            
            # Crop the image to the circle's bounding box (optional)
            x1, y1 = max(x - radius, 0), max(y - radius, 0)
            x2, y2 = min(x + radius, img.shape[1]), min(y + radius, img.shape[0])
            cropped_image = result[y1:y2, x1:x2]
            
            # Show the result
            if debug:
                cv.imshow("Cropped Image", cropped_image)
                cv.waitKey(0)
                cv.destroyAllWindows()
            
            return cropped_image
        else:
            print("No circles detected!")
            return None
     
    def get_time_from_clock_image(self, filename='clknobound.png', debug=0):
        """
        Returns time of analog clock image.

        Parameters
        ----------
        filename : STRING
            Path to PNG image file with analog clock. Prequistions:
                Must be of the same format as the analog clock generated in draw_analog_clock method.

        Raises
        ------
        Exception
            If file is of invalid type (not PNG) or if the image doesn't have at least 
            3 lines (3 clock hands).

        Returns
        -------
        hour : INT
            Hour result (1-12).
        minute : TYPE
            Minute result (0-60).
        second : TYPE
            Second result (0-60).

        """
        if filename == '' or not (filename.lower().endswith('.png') or filename.lower().endswith('.jpg')):
            raise Exception("Invalid file name")

        cropped_image = self.get_clock_circle(filename, debug) 
        if cropped_image == None:
            raise Exception("Invalid image!! no clocks found!!")
        linesP = self.get_lines_in_image(cropped_image, debug)
                  
        if len(linesP) < 1: # not enough lines in image
            raise Exception("No valid clock detected")
        
        linesP = self.shift_lines(linesP, Point(472/2, 472/2))
        hands=[]
        
        if (len(linesP)==1):
            x1, y1, x2, y2 = linesP[0][0]
            p1 = Point(x1,y1)
            p2 = Point(x2,y2)
            p = self.farthest_point(p1,p2,center=Point(472/2, 472/2))
            angle = self.get_angle_from_center(point=p, center=Point(472/2, 472/2))
            hour, minute = self.convert_angle_to_time(angle, angle)
            
            print(f'{hour}:{minute}')
            return hour, minute
        
        for line in linesP:
            x1, y1, x2, y2 = line[0]
            p1 = Point(x1,y1)
            p2 = Point(x2,y2)
            p = self.farthest_point(p1,p2,center=Point(472/2, 472/2))
            angle = self.get_angle_from_center(point=p, center=Point(472/2, 472/2))
            length = self.line_length(line[0])
            hands.append((p1, p2, angle, length))
            
        minute_hand = hands[0]
        hour_hand = hands[1]
        
        hour, minute = self.convert_angle_to_time(hour_hand[2], minute_hand[2])
        
        print(f'{hour}:{minute}')
        return hour, minute
            
    def draw_analog_clock(self, hour, minute):
        """
        Creates PNG images of analog clock that shows the time given.

        Parameters
        ----------
        hour : INT
            Hour (1-12).
        minute : INT
            Minutes (0-60).

        Raises
        ------
        Exception
            If the time given is invalid.

        Returns
        -------
        plot_path : STRING
            Path to the generated PNG image of the analog clock.

        """
        clk_center=0.5
        radius=0.45
        
        if hour < 0 or hour > 12 or minute < 0 or minute > 60:
            raise Exception("Invalid input time")
        
        
        # Create a figure and axis for the clock
        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
        
        # Set the aspect ratio to be equal
        ax.set_aspect('equal')
        
        # Draw the clock face (circle)
        clock_face = plt.Circle(xy=(clk_center, clk_center), radius=radius, edgecolor='black', facecolor='white', lw=4)
        ax.add_artist(clock_face)
        
        # Draw the tick marks for each hour (12 ticks total)
        for i in range(12):
            angle = np.deg2rad(60 - 360*i/12)
            x_start = clk_center + (radius-0.02) * np.cos(angle)
            y_start = clk_center + (radius-0.02) * np.sin(angle)
            x_end = clk_center + radius * np.cos(angle)
            y_end = clk_center + radius * np.sin(angle)
            ax.plot([x_start, x_end], [y_start, y_end], color='black', lw=2)
            
            # Add numbers around the clock
            num_x = clk_center + (radius-0.10) * np.cos(angle)
            num_y = clk_center + (radius-0.10) * np.sin(angle)
            ax.text(num_x, num_y, str(i+1), ha='center', va='center', fontsize=16, fontweight='bold')
    
        # Draw the hour hand
        hour_angle = np.deg2rad(60 - 360*(hour-1 + (minute-1)/60)/12)
        hour_x = clk_center + (radius-0.2) * np.cos(hour_angle)
        hour_y = clk_center + (radius-0.2) * np.sin(hour_angle)
        ax.plot([clk_center, hour_x], [clk_center, hour_y], color='black', lw=6)
        
        # Draw the minute hand
        minute_angle = np.deg2rad(90 - 360*((minute)/60))
        minute_x = clk_center + 0.35 * np.cos(minute_angle)
        minute_y = 0.5 + 0.35 * np.sin(minute_angle)
        ax.plot([0.5, minute_x], [0.5, minute_y], color='blue', lw=4)
    
        # Draw the center of the clock
        ax.plot(0.5, 0.5, 'ko', markersize=8)
        
        # Remove axis
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Show the clock image
        plt.axis('off') # remove borders
        dt = datetime.now()
        ts = datetime.timestamp(dt)

        if not os.path.exists('images'):
            os.makedirs('images')
        plot_path = f'images/clock_image_{int(ts)}.png'
        plt.tight_layout()
        plt.savefig(plot_path, dpi=100)
        plt.show()
        
        return plot_path
