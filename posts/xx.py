dft = cv2.dft(np.float32(gray),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
dft_shift[227:233, 219:225] = 255
dft_shift[227:233, 236:242] = 255

f_ishift = np.fft.ifftshift(im_fft2)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

plt.imshow(img_back, cmap="gray")
plt.show()
min, max = np.amin(img_back, (0,1)), np.amax(img_back, (0,1))
img_back = cv2.normalize(img_back, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32S)
img_back = np.uint8(img_back)
img = counting(image, img_back)
