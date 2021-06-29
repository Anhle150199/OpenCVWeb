import cv2
import numpy as np
import matplotlib.pyplot as plt


def counting(image, choice):
    image = np.uint8(image)
    if choice == 3:
        # --------- cach 1 ---------#
        # image = np.power(image, 0.01)
        # max_val = np.max(image.ravel())
        # image = image / max_val * 255
        # image = image.astype(np.uint8)

        # -------- cach 2 -----------#
        lookUpTable = np.empty((1, 256), np.uint8)
        for i in range(256):
            lookUpTable[0, i] = np.clip(pow(i / 255.0, 0.01) * 255.0, 0, 255)
        image = cv2.LUT(image, lookUpTable)

    image = cv2.medianBlur(image, 3, None)
    imageOut = image
    imgSmooth = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if choice == 2:
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
        imgSmooth = img.copy()
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

    # imgSmooth = imageOut
    thresh = cv2.adaptiveThreshold(image, 255.0,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, -20.0)

    if choice == 4:
        lookUpTable = np.empty((1, 256), np.uint8)
        for i in range(256):
            lookUpTable[0, i] = np.clip(pow(i / 255.0, 2.5) * 255.0, 0, 255)
        img = cv2.LUT(image, lookUpTable)

        thresh = cv2.adaptiveThreshold(img, 255.0, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51, 6)
        thresh = cv2.medianBlur(thresh, 5)
        kernel = np.ones((9, 9), np.uint8)
        thresh = cv2.dilate(thresh, None, iterations=7)
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.erode(thresh, kernel, iterations=2)

    imgThresh = thresh
    kernel = np.ones((5, 5), np.uint8)
    output_erosion = cv2.erode(thresh, kernel, iterations=1)
    # cv.imshow("Morphological Erosion", output_erosion)
    imgErode = output_erosion

    contours, _ = cv2.findContours(output_erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # image = cv2.cvtColor(imageOut, cv2.COLOR_GRAY2BGR)
    image = imageOut
    for (i, c) in enumerate(contours):
        ((x, y), r) = cv2.minEnclosingCircle(c)
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        cv2.putText(image, "{}".format(i + 1), (int(x) - 10, int(y) + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        # image = output_contour

    return [image, imgSmooth, imgThresh, imgErode, (i+1)]

# image = cv2.imread("static/img/gao3.png")
# img = counting(image,2)
# cv2.imshow("image", img[0])
# cv2.imshow("image1", img[1])
# cv2.imshow("image2", img[2])
# cv2.imshow("image3", img[3])
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
