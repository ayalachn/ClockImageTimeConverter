"""
Created on Fri Jan 17 09:59:50 2025

@author: ayalac

Converts analog clock to time, converts time to analog clock
"""
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from utils.LineManager import Line
from utils.PointManager import Point
        
class ClockImageTimeConverter:    
    def __init__(self):
        print("Hello from clock converter constructor!")
        
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
        # Calculate minute based on the minute_angle
        minute = int(minute_angle * 60 / 360)
    
        # Calculate hour based on the hour_angle
        if minute >= 30:
            hour= int(hour_angle * 12 / 360)
        else:
            hour = round(hour_angle * 12 / 360)
        
        if hour == 0:
            hour =12
        return hour, minute

    # Check if the angles are close within a thresholds
    def are_angles_close(self, angle1, angle2, angle_threshold=10): 
        diff = abs(angle1 - angle2) % 360
        if diff > 180:
            diff = 360 - diff  # Shortest distance on a circle (wraparound)
        return diff < angle_threshold  # Return True if the angles are within the threshold
  
    # Remove double lines (lines with similar angles and lengths)
    def remove_double_lines(self, lines, length_threshold=100, angle_threshold=5):
        unique_lines = []
        lines_to_remove = []
        
        # sort lines by length - longest first
        lines = np.array(sorted(lines, key=lambda line: Line(line[0]).get_line_length(), reverse=True))
        
        if len(lines) == 1:
            return lines # hour and minute hands are overlapping
        
        for i, line in enumerate(lines):
            l = Line(line[0])
            p = l.get_farthest_point(center=Point(472/2, 472/2))
            line_ang = p.get_angle_from_center(center=Point(472/2, 472/2))
            line_len = l.get_line_length()
            is_duplicate = False
            
            for j, unique_line in enumerate(unique_lines):
                unique_l = Line(unique_line[0])
                p_unique = unique_l.get_farthest_point(center=Point(472/2, 472/2))
                unique_ang = p_unique.get_angle_from_center(center=Point(472/2, 472/2))
                unique_len = unique_l.get_line_length()

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
        Prequisite - image most contain only the analog clock with hour and minute hands, without seconds.

        Parameters
        ----------
        src : STRING
           Preprocessed image cropped to clock bounding box.
          
        debug : INT
            set to 1 if you want to see images of line/circle detection results (for future improvements of this repo)
            set to 0 if you don't want to see images.
        Returns
        -------
        list
            List of line vectors in image.

        """
        
        # reshape image to have a constant center coordinate
        src = cv.resize(src, (472, 472)) # clock center (0.5, 0.5) in the normalized coordinates
 
        dst = cv.Canny(src, 200, 300, None, 3)
        
        # for debug:
        cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
        cdstP = np.copy(cdst)
        ##
        
        linesP = cv.HoughLinesP(image=dst, rho=1, theta=np.pi / 180, threshold =23, minLineLength =80, maxLineGap =10)
        linesP = np.array(sorted(linesP, key=lambda line: Line(line[0]).get_line_length(), reverse=True))
        linesP = self.shift_lines(linesP, Point(472/2, 472/2))
        linesP= self.remove_double_lines(linesP)

        if debug: # show images of detected lines
            if linesP is not None:
                for i in range(0, len(linesP)):
                    l = linesP[i][0]
                    cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)                
                    cv.imshow(f"Detected Lines (in red) {i}- Probabilistic Line Transform no double lines", cdstP)
            cv.waitKey()
            
        return linesP
    
    def shift_lines(self, lines, center_point): # shift lines to center of circle to get a more accurate time result
        for i, line in zip(range(len(lines)), lines):
            l = Line(line[0])
            l.shift_line(center_point)
            lines[i][0] = l.get_numpy_line()
        return lines
 
    def get_clock_circle(self, filename, debug=0):
        """
        Part of the preproccessing of input image, removes surrounding noise and returns clock's bounding box.

        Parameters
        ----------
        filename : STRING
            Path to clock image (PNG or JPG).
        debug : INT, optional
            Set to 1 if you want to see the clock detection result image (used for future improvements of this repository). 
            The default is 0.

        Raises
        ------
        Exception
            If file is not found or not in correct image formate (PNG, JPG, TIF).

        Returns
        -------
        cropped_image : Array of uint8 (matrix)
            Input image cropped to largest circle (clock) bounding box.

        """
        if filename == '' or not (filename.lower().endswith('.png') or filename.lower().endswith('.jpg') or filename.lower().endswith('.tif')):
            raise Exception("Invalid file type (must be PNG, JPG or TIF)")
            
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
            raise Exception("No valid clock detected")
     
    def get_time_from_clock_image(self, filename='', debug=0):
        """
        Returns time of analog clock image.

        Parameters
        ----------
        filename : STRING
            Path to image file with analog clock. 
            Must be of type PNG/JPG/TIF.

        Raises
        ------
        Exception
            If file is of invalid type (not PNG/JPG/TIF) or if the image doesn't contain a clock.

        Returns
        -------
        hour : INT
            Hour result (1-12).
        minute : TYPE
            Minute result (0-60).

        """
        if filename == '' or not (filename.lower().endswith('.png') or filename.lower().endswith('.jpg') or filename.lower().endswith('.tif')):
            raise Exception("Invalid file type (must be PNG, JPG or TIF)")

        cropped_image = self.get_clock_circle(filename, debug) # preprocesses image, get clock face bounding box
        if len(cropped_image) < 1: # no circles detected
            raise Exception("No valid clock detected")
        linesP = self.get_lines_in_image(cropped_image, debug) # get clock hands
                  
        if len(linesP) < 1: # not enough lines in image
            raise Exception("No valid clock detected")
        
        linesP = self.shift_lines(linesP, Point(472/2, 472/2))
        hands=[]
        
        if (len(linesP)==1): # hour and minute hand are overlapping
            l = Line(linesP[0][0])
            p = l.get_farthest_point(center=Point(472/2, 472/2))
            angle = p.get_angle_from_center(center=Point(472/2, 472/2))
            hour, minute = self.convert_angle_to_time(angle, angle)
            
            print(f'{hour}:{minute}')
            return hour, minute
        
        for line in linesP: # hour and minute hand are not overlapping
            l = Line(line[0])
            p = l.get_farthest_point(center=Point(472/2, 472/2))
            angle = p.get_angle_from_center(center=Point(472/2, 472/2))
            length = l.get_line_length()
            hands.append((l.p1, l.p2, angle, length))
            
        minute_hand = hands[0]
        hour_hand = hands[1]
        
        hour, minute = self.convert_angle_to_time(hour_hand[2], minute_hand[2])
        
        print(f'{hour}:{minute}')
        return hour, minute
            
    def draw_analog_clock(self, hour, minute, image_format='png'):
        """
        Creates PNG images of analog clock that shows the time given (hour:minute).

        Parameters
        ----------
        hour : INT
            Hour (1-12).
        minute : INT
            Minutes (0-60).
        image_format : STRING, optional
            Set the format of the clock image (png/jpg/tif). 
            The default is png.
            
        Raises
        ------
        Exception
            If the time given is invalid.

        Returns
        -------
        plot_path : STRING
            Path to the generated PNG image of the analog clock.

        """
        if image_format == '' or not (image_format.lower()=='png' or image_format.lower()=='jpg' or image_format.lower()=='tif'):
            raise Exception("Invalid file format (must be PNG, JPG or TIF)")
            
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
        plot_path = f'images/clock_image_{int(ts)}.{image_format}'
        plt.tight_layout()
        plt.savefig(plot_path, dpi=100)
        plt.show()
        
        return plot_path
