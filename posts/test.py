import cv2
import numpy as np
from matplotlib import pyplot as plt




img = cv2.imread('static/img/mface.jpg')
# img = cv2.resize(img, None, fx=0.3, fy=0.3)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

image = faceDetect(gray)
# cv2.imshow("Face Detector", image)
plt.imshow(image, cmap='gray')
plt.show()
k=cv2.waitKey(0)
cv2.destroyAllWindows()