import cv2 as cv
import cv2
import numpy as np

image = cv.imread("static/img/objets4.jpg", cv.IMREAD_GRAYSCALE)
r = 4
image = cv2.medianBlur(image, 3, None)
imageOut = image

if r == 3:
    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, 0.01) * 255.0, 0, 255)
    image = cv2.LUT(image, lookUpTable)
    imageOut=image

if r == 2:
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    dft_shift[227:233, 219:225] = 255
    dft_shift[227:233, 236:242] = 255

    f_ishift = np.fft.ifftshift(dft_shift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    # min, max = np.amin(img_back, (0, 1)), np.amax(img_back, (0, 1))
    img = cv2.normalize(img_back, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    imageOut = img
    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, 5) * 255.0, 0, 255)
    res = cv2.LUT(img, lookUpTable)
    kernel = np.ones((9, 9), np.uint8)
    res = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)
    # res = cv2.erode(res, None, iterations=2)
    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, 0.7) * 255.0, 0, 255)

    res = cv2.LUT(res, lookUpTable)
    image = cv2.adaptiveThreshold(res, 255.0, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 71, -25)

output_adapthresh = cv.adaptiveThreshold (image, 255.0,
		cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 51, -20.0)
cv.imshow("Adaptive Thresholding", output_adapthresh)

if r==4:
    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, 2.5) * 255.0, 0, 255)
    img = cv2.LUT(image, lookUpTable)

    thresh = cv2.adaptiveThreshold(img, 255.0, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51, 6)
    thresh = cv2.medianBlur(thresh, 5)
    # cv2.imshow("res1", thresh)
    kernel = np.ones((9,9),np.uint8)
    output_adapthresh = cv2.dilate(thresh, None, iterations=7)
    kernel = np.ones((5, 5), np.uint8)
    output_adapthresh = cv.erode(output_adapthresh, kernel, iterations=2)
    # cv2.imshow("res2", res)


kernel = np.ones((5,5),np.uint8)
output_erosion = cv.erode(output_adapthresh, kernel, iterations=1)
cv.imshow("Morphological Erosion", output_erosion)

contours, _ = cv.findContours(output_erosion, cv.RETR_EXTERNAL,  cv.CHAIN_APPROX_SIMPLE)
output_contour = cv.cvtColor(imageOut, cv.COLOR_GRAY2BGR)
for (i, c) in enumerate(contours):
    ((x, y), r) = cv2.minEnclosingCircle(c)
    cv2.drawContours(output_contour, [c], -1, (0, 255, 0), 2)
    cv2.putText(output_contour, "{}".format(i + 1), (int(x) - 10, int(y) + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
print("Number of detected contours", len(contours))
cv.imshow("Contours", output_contour)
#cv.imwrite('rice_contours.png', output_contour)

# wait for key press
cv.waitKey(0)
cv.destroyAllWindows()