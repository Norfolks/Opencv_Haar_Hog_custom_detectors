import numpy as np
import cv2

img = cv2.imread('lena.bmp', 0)
# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
cv2.imshow('image', img)
key_pressed = cv2.waitKey(0) & 0xFF
if key_pressed == 27:
    cv2.destroyAllWindows()
elif key_pressed == ord('s'):
    cv2.imwrite('lena.png', img)
    cv2.destroyAllWindows()

