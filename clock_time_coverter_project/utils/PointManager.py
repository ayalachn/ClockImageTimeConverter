# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 00:29:50 2025

@author: ayalac

Utility class for point in 2D space
"""
import math
import numpy as np
class Point: # a point in a 2D plane
    def __init__(self, x,y):
        self.x=x
        self.y=y
    
    def euclidean_distance(self, point): # returns euclidean distance between two points
        p1 = np.array((self.x, self.y))
        p2 = np.array((point.x, point.y))
        return np.linalg.norm([p1 - p2])

    def get_angle_from_center(self, center):
        """
        

        Parameters
        ----------
        center : POINT
            The center of the circle (clock in our case).

        Returns
        -------
        angle : FLOAT
            The angle between the point, and the negative Y axis (up). 
            Where the axis center is the center of the circle (clock)
            The axis: Y positive down and X positive right. I wanted angle 0 to be hour 12.

        """
        dx = self.x - center.x
        dy = self.y - center.y
        
        # edge cases
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
        if self.x > center.x and self.y >= center.y:
            quarter_num=1
        elif self.x < center.x and self.y > center.y:
            quarter_num=2
        elif self.x < center.x and self.y < center.y:
            quarter_num=4
        elif self.x > center.x and self.y < center.y:
            quarter_num=3

        angle=180
        
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