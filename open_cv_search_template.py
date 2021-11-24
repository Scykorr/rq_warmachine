import cv2  
import numpy as np  
image = cv2.imread("Large.png")  
template = cv2.imread("small.png")  
result = cv2.matchTemplate(image,template,cv2.TM_CCOEFF_NORMED)  
print(np.unravel_index(result.argmax(),result.shape))