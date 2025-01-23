# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 00:29:50 2025

@author: ayalac

A simple unit test for the ClockImageTimeConverter to check the clock image to time convertion functionality.
"""
import unittest

import sys
sys.path.insert(1, '../clock_time_coverter_project')

from ClockImageTimeConverter import ClockImageTimeConverter
from utils.PointManager import Point

class unitTestClockImageTime(unittest.TestCase):
    
    def setUp(self):
        self.converter = ClockImageTimeConverter()  # Create an instance of the ClockImageTimeConverter class
        self.project_path='../clock_time_coverter_project/'
    
    def test_clock_converter_sanity(self):
        hour=1
        minute=30
        
        # create analog clock image
        img_result_path = self.converter.draw_analog_clock(hour=hour, minute=minute)
        
        # check result
        hour_, minute_ = self.converter.get_time_from_clock_image(img_result_path)

        self.assertEqual(hour, hour_)
        self.assertAlmostEqual(minute, minute_, delta=1)
        
    def test_clock_converter_same_angle(self):
        hour=12
        minute=0
        
        # create analog clock image
        img_result_path = self.converter.draw_analog_clock(hour=hour, minute=minute)
        
        # check result
        hour_, minute_= self.converter.get_time_from_clock_image(img_result_path)

        self.assertEqual(hour, hour_)
        self.assertAlmostEqual(minute, minute_, delta=1)

    def test_clock_converter_1st_quarter(self):
        hour=4
        minute=2
        
        # create analog clock image
        img_result_path = self.converter.draw_analog_clock(hour=hour, minute=minute)
        
        # check result
        hour_, minute_= self.converter.get_time_from_clock_image(img_result_path)

        self.assertEqual(hour, hour_)
        self.assertAlmostEqual(minute, minute_, delta=1)        
        
    def test_clock_converter_2nd_quarter(self):
        hour=8
        minute=50
        
        # create analog clock image
        img_result_path = self.converter.draw_analog_clock(hour=hour, minute=minute)
        
        # check result
        hour_, minute_= self.converter.get_time_from_clock_image(img_result_path)

        self.assertEqual(hour, hour_)
        self.assertAlmostEqual(minute, minute_, delta=1)      

    def test_clock_converter_3th_quarter(self):
        hour=2
        minute=0
        
        # create analog clock image
        img_result_path = self.converter.draw_analog_clock(hour=hour, minute=minute)
        
        # check result
        hour_, minute_= self.converter.get_time_from_clock_image(img_result_path)

        self.assertEqual(hour, hour_)
        self.assertAlmostEqual(minute, minute_, delta=1)   
          
    def test_clock_converter_4th_quarter(self):
        hour=10
        minute=42
        
        # create analog clock image
        img_result_path = self.converter.draw_analog_clock(hour=hour, minute=minute)
        
        # check result
        hour_, minute_= self.converter.get_time_from_clock_image(img_result_path)

        self.assertEqual(hour, hour_)
        self.assertAlmostEqual(minute, minute_, delta=1)

    def test_clock_converter_4th_quarter(self):
        hour=10
        minute=42

        # create analog clock image
        img_result_path = self.converter.draw_analog_clock(hour=hour, minute=minute)
        
        # check result
        hour_, minute_= self.converter.get_time_from_clock_image(img_result_path)

        self.assertEqual(hour, hour_)
        self.assertAlmostEqual(minute, minute_, delta=1)
if __name__ == "__main__":
    unittest.main()
    
