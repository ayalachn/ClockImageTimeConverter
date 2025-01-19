# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 00:29:50 2025

@author: ayalac

Unit test for the ClockImageTimeConverter
"""
import unittest

import sys
sys.path.insert(1, '../clock_time_coverter_project')

from ClockImageTimeConverter import ClockImageTimeConverter
from ClockImageTimeConverter import Point

class unitTestClockImageTime(unittest.TestCase):
    
    def setUp(self):
        self.converter = ClockImageTimeConverter()  # Create an instance of the ClockImageTimeConverter class
        self.project_path='../clock_time_coverter_project/'
    
    def test_clock_converter(self):
        hour=1
        minute=30
        second=9
        
        # create analog clock image
        img_result_path = self.converter.draw_analog_clock(hour=hour, minute=minute, second=second)
        
        # check result
        hour_, minute_, second_ = self.converter.get_time_from_clock_image(img_result_path)

        self.assertEqual(hour, hour_)
        self.assertEqual(minute, minute_)
        self.assertEqual(second, second_)
        

if __name__ == "__main__":
    unittest.main()
    