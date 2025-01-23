# ClockImageTimeConverter
Converts time to analog clock image, converts analog clock image to time.

## Create analog clock
Inorder to create an analog clock image (PNG format) call the *draw_analog_clock* method that's in the *ClockImageTimeConverter* class.
Usage example - create an analog clock that shows the time _10:15_:
```python
converter = ClockImageTimeConverter()
converter.draw_analog_clock(10, 15)
```

Result:

![image](https://github.com/user-attachments/assets/ad02d3b9-4de4-49cd-a390-36d6a43de02b)

## Convert analog clock image to time
Inorder to get the time shown in an analog clock image call the *get_time_from_clock_image* method that's in the *ClockImageTimeConverter* class.
```python
converter = ClockImageTimeConverter()
hour, minutes = converter.get_time_from_clock_image('my_analog_clock.png')
```

Example:
Let's get the time of this analog clock:

![image](https://github.com/user-attachments/assets/c34dc92f-d4c9-4200-9ec2-bc08e6dd46a4)

Code:
```python
converter = ClockImageTimeConverter()
hour_, minute_= converter.get_time_from_clock_image('H-4570.png')
```
Result:

![image](https://github.com/user-attachments/assets/a96c7fa9-6ba1-442d-a4ca-0b56cf5c81e7)

### Prequisites
- No support for analog clock image with second hand.
- Install libraries used in this project (requirements.txt):
  ```python
    pip install -r requirements.txt
  ```
## Sources:
- https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html

- https://docs.opencv.org/4.x/da/d53/tutorial_py_houghcircles.html
