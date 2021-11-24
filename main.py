from mss import mss
from PIL import Image, ImageGrab
import cv2, numpy

def capture_screenshot():
    # Capture entire screen
    with mss() as sct:
        monitor = sct.monitors[1]
        sct_img = sct.grab(monitor)
        # Convert to PIL/Pillow Image
        return Image.frombytes('RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX')

img = capture_screenshot()
img.show()

"""
image = Image.open('jelly.jpg')
cropped = image.crop((0, 80, 200, 400))
cropped.save('/path/to/photos/cropped_jelly.png')
"""