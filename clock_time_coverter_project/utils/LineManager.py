# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 00:29:50 2025

@author: ayalac

Utility class for a line in 2D space
"""
import numpy as np
from .PointManager import Point
class Line:
    def __init__(self, line):
        self.p1 = Point(line[0], line[1])
        self.p2 = Point(line[2], line[3])
    
    def get_line_length(self): # returns euclidean distance of line
        return self.p1.euclidean_distance(self.p2)
    
    def get_numpy_line(self): # returns numpy array representation of line
        return np.array([self.p1.x, self.p1.y, self.p2.x, self.p2.y])
    
    def shift_line(self, center_point): # shift line to center point
        dist_p1 = center_point.euclidean_distance(self.p1)
        dist_p2 = center_point.euclidean_distance(self.p2)
        if  dist_p1 < dist_p2:
            shift_x = self.p1.x - center_point.x
            shift_y = self.p1.y - center_point.y
        else:
            shift_x = self.p2.x - center_point.x
            shift_y = self.p2.y - center_point.y
        
        self.p1.x -= shift_x
        self.p1.y -= shift_y
        self.p2.x -= shift_x
        self.p2.y -= shift_y
        
    def get_farthest_point(self, center): # return farthest point from the center
            l1 = self.p1.euclidean_distance(center)
            l2 = self.p2.euclidean_distance(center)
            
            if l1 > l2:
                return self.p1
            return self.p2