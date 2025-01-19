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

class Point: # a point in a 2D plane
    def __init__(self, x,y):
        self.x=x
        self.y=y
        
    def get_normalized_point(self, width, height): # normalize point according to the plane it's in
        x_normalized = self.x / width
        y_normalized = self.y / height
        return Point(x_normalized, y_normalized)
    
class ClockImageTimeConverter:    
    def __init__(self):
        print("Hello from clock converter constructor!")
        
    def line_length(self, line):
        x1, y1, x2, y2 = line
        point1 = np.array((x1, y1))
        point2 = np.array((x2, y2))
        d = np.linalg.norm([point1 - point2])  # Euclidean distance
        return d
    
    def convert_angle_to_time(self, hour_angle, minute_angle, second_angle):
        """
        Computes & returns the hour according to the angle of the hands.

        Parameters
        ----------
        hour_angle : FLOAT
            Hour angle in degrees.
        minute_angle : FLOAT
            Minute angle in degrees.
        second_angle : FLOAT
            Second angle in degrees.

        Returns
        -------
        hour : INT
            Hour (1-12).
        minute : INT
            Minute (0-60).
        second : INT
            Second (0-60).

        """
        hour = int(hour_angle*12 / 360)
        if (hour == 0):
            hour = 12
        minute = int(minute_angle*60 / 360)
        second = int(second_angle*60 / 360)
        return hour, minute, second
    
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
        
        # find point's quarter position
        quarter_num=0
        if point.x > center.x and point.y > center.y:
            quarter_num=1
        elif point.x < center.x and point.y > center.y:
            quarter_num=2
        elif point.x < center.x and point.y < center.y:
            quarter_num=4
        elif point.x > center.x and point.y < center.y:
            quarter_num=3

        angle=0
        
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
            angle = -1*math.degrees(rad) 
            return 270+angle
        
        return angle
    
    def line_angle(self, line):
        x1, y1, x2, y2 = line
        dx = x2 - x1
        dy = y2 - y1
        angle = math.atan2(dy, dx)  # Angle in radians
        return math.degrees(angle)  # Convert to degrees
    
    # Function to remove double lines (lines with similar angles)
    def remove_double_lines(self, lines, angle_threshold=1):
        unique_lines = []
        lines_to_remove = []
        
        lines = np.array(sorted(lines, key=lambda line: self.line_length(line[0]), reverse=True))
        
        for i, line in enumerate(lines):
            line_ang = self.line_angle(line[0])
            
            is_duplicate = False
            
            for j, unique_line in enumerate(unique_lines):
                unique_ang = self.line_angle(unique_line[0])
                
                # Check if the angle is within the threshold
                if abs(line_ang - unique_ang) < angle_threshold: # I would also check length if the 
                                                                    # length is the same but that lead to more issues b/c of overlap
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_lines.append(line)
        
        return unique_lines
        
    def get_lines_in_image(self, img):
        """
        Using the Hough Lines Probablistic algorithm we find the lines inside our image,
        this will help us find the clock hands.
        Prequisite - image most contain only the analog clock's face and has to be of similar format
        of the clock generated in draw_analog_clock method.

        Parameters
        ----------
        img : STRING
            Path to PNG image file with analog clock. Prequistions:
                Must be of the same format as the analog clock generated in draw_analog_clock method.

        Returns
        -------
        list
            List of line vectors in image.

        """
        src = cv.imread(cv.samples.findFile(img), cv.IMREAD_GRAYSCALE)
        
        # Check if image is loaded fine
        if src is None:
            print ('Error opening image!')
            print ('Usage: hough_lines.py [image_name -- default ' + default_file + '] \n')
            return []
        
        # reshape image to have a constant center coordinate
        src = cv.resize(src, (472, 472)) # clock center (0.5, 0.5) in the normalized coordinates
 
        dst = cv.Canny(src, 50, 200, None, 3)
    
        linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)

        if linesP is not None and len(linesP) > 4:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)

        # remove same edge lines (doubles)
        linesP_noDoubles = self.remove_double_lines(linesP)
        for i in range(len(linesP_noDoubles)):
            l = linesP_noDoubles[i][0]
         
        return linesP_noDoubles
    
    def farthest_point(self, p1, p2, center): # compute farthest point from the center
        l1 = self.line_length([p1.x, p1.y, center.x, center.y])
        l2 = self.line_length([p2.x, p2.y, center.x, center.y])
        
        if l1 > l2:
            return p1
        return p2
    
    def get_time_from_clock_image(self, filename='clknobound.png'):
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
        if filename == '' or not filename.lower().endswith('.png'):
            raise Exception("Invalid file name")

            
        linesP = self.get_lines_in_image(filename)
                  
        if len(linesP) < 3: # not enough lines in image
            raise Exception("No valid clock detected")
            
        hands=[]
        
        for line in linesP[:3]:
            x1, y1, x2, y2 = line[0]
            p1 = Point(x1,y1)
            p2 = Point(x2,y2)
            p = self.farthest_point(p1,p2,center=Point(472/2, 472/2))
            angle = self.get_angle_from_center(point=p, center=Point(472/2, 472/2))
            length = self.line_length(line[0])
            hands.append((p1, p2, angle, length))
            
        second_hand = hands[0]
        minute_hand = hands[1]
        hour_hand = hands[2]
        
        hour, minute, second = self.convert_angle_to_time(hour_hand[2], minute_hand[2], second_hand[2])
        
        print(f'{hour}:{minute}:{second}')
        return hour, minute, second
            
    def draw_analog_clock(self, hour, minute, second):
        """
        Creates PNG images of analog clock that shows the time given.

        Parameters
        ----------
        hour : INT
            Hour (1-12).
        minute : INT
            Minutes (0-60).
        second : INT
            Seconds (0-60).

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
        
        if hour < 0 or hour > 12 or minute < 0 or minute > 60 or second < 0 or second > 60:
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
        minute_angle = np.deg2rad(90 - 360*((minute)/60 + (second)/3600))
        minute_x = clk_center + 0.35 * np.cos(minute_angle)
        minute_y = 0.5 + 0.35 * np.sin(minute_angle)
        ax.plot([0.5, minute_x], [0.5, minute_y], color='blue', lw=4)
        
        # Draw the second hand
        second_angle = np.deg2rad(90 - 360 * second / 60)
        second_x = clk_center + 0.4 * np.cos(second_angle)
        second_y = clk_center + 0.4 * np.sin(second_angle)
        ax.plot([clk_center, second_x], [clk_center, second_y], color='red', lw=2)
    
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
