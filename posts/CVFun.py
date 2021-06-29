import numpy as np
import cv2
from matplotlib import pyplot as plt

def cannyFilter(img):

    return img

def objDetect(img1, img2):
    MIN_MATCH_COUNT = 10
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    img = cv2.drawKeypoints(img1, kp1, img1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("sift1", img)
    kp2, des2 = sift.detectAndCompute(img2,None)
    img = cv2.drawKeypoints(img2, kp2, img2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("sift2", img)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    # cv.drawMatchesKnn expects list of lists as matches.
    # img1=cv2.resize(img1, None, fx=0.5, fy=0.5)
    img = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # cv2.imshow("sift3", img)
    plt.imshow(img)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w,d = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    return img2

def faceDetect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray",gray)
    face_cascade = cv2.CascadeClassifier("static/haarcascades/haarcascade_frontalface_alt.xml")
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)
    i=0
    for x,y,w,h in faces:
        img = cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0),4)
        i+=1
    print("The face's count is detected: ", i)
    return img

img= cv2.imread("static/img/face/face1.jpg")
img= cv2.resize(img,None, fx=0.5, fy=0.5)
img = faceDetect(img)
plt.imshow(img)
plt.show()
cv2.waitKey(0)