# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 09:59:50 2025

@author: ayalac
"""
import matplotlib.pyplot as plt
import numpy as np

def draw_analog_clock(hour, minute, second):
    clk_center=0.5
    radius=0.45
    
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
        x_start = clk_center + (radius-0.05) * np.cos(angle)
        y_start = clk_center + (radius-0.05) * np.sin(angle)
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
    # minute_angle = np.deg2rad(60 - 360*((minute)/60 + (second)/3600))
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
    plt.show()

# Example usage
hour = 12
minute = 10
second = 0
draw_analog_clock(hour, minute, second)

