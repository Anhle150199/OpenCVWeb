import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('static/img/gao3.png',0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
# calculate amplitude spectrum
mag_spec = 20*np.log(np.abs(fshift))

r = f.shape[0]/2        # number of rows/2
c = f.shape[1]/2        # number of columns/2
p = 3
n = 1                   # to suppress all except for the DC component
fshift2 = np.copy(fshift)

# suppress upper part
fshift2[0:r-n, c-p:c+p] = 0.001
# suppress lower part
fshift2[r+n:r+r, c-p:c+p] = 0.001
# calculate new amplitude spectrum
mag_spec2 = 20*np.log(np.abs(fshift2))
inv_fshift = np.fft.ifftshift(fshift2)
# reconstruct image
img_recon = np.real(np.fft.ifft2(inv_fshift))

plt.subplot(131),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(mag_spec, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(mag_spec2, cmap = 'gray')
plt.title('Magnitude Spectrum after suppression'), plt.xticks([]), plt.yticks([])
plt.show()

plt.figure()
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_recon, cmap = 'gray')
plt.title('Output Image'), plt.xticks([]), plt.yticks([])
plt.show()
