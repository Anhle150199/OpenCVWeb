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
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image =cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if choice == 2:
        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        dft_shift[227:233, 219:225] = 255
        dft_shift[227:233, 236:242] = 255

        f_ishift = np.fft.ifftshift(dft_shift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

        min, max = np.amin(img_back, (0, 1)), np.amax(img_back, (0, 1))
        img = cv2.normalize(img_back, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # rice = cv2.medianBlur(rice, 3, None)
    img1 = img
    thresh = cv2.adaptiveThreshold(img, 255.0, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, -20.0)
    img2 = thresh
    distMap = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    cv2.normalize(distMap, distMap, 0.0, 255.0, cv2.NORM_MINMAX)
    distMap = np.uint8(distMap)
    # cv2.imshow('distMap', distMap)
    img3 = distMap
    foreground = cv2.threshold(distMap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # foreground = cv2.dilate(foreground, None, 1)
    foreground = cv2.erode(foreground, None, 3)
    # cv2.imshow('foreground', foreground)

    unknowZones = cv2.subtract(thresh, foreground)
    # cv2.imshow('unknowZones', unknowZones)
    ret, markers = cv2.connectedComponents(foreground, connectivity=8, ltype=cv2.CV_32S)
    markers = markers + 1
    markers[unknowZones == 255] = 0
    # for m in np.unique(markers):
    #     print(markers[m])
    markers = cv2.watershed(image, markers)
    # for m in np.unique(markers):
    #     print(markers[m])
    cnts = []
    for m in np.unique(markers):
        if m < 2:
            continue
        mask = np.zeros(markers.shape, dtype="uint8")
        mask[markers == m] = 255
        c = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        cnts.extend(c)

    for (i, c) in enumerate(cnts):
        ((x, y), r) = cv2.minEnclosingCircle(c)
        cv2.drawContours(image, [c], -1, (0, 0, 255), 1)
        cv2.putText(image, "{}".format(i + 1), (int(x) - 10, int(y) + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 50, 0), 2)

    return [image, (i + 1), img1, img2, img3, foreground, unknowZones]

# image = cv2.imread("static/img/gao4.png")
# img = counting(image,4)[0]
# cv2.imshow("image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
