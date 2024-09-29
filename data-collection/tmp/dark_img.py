# import the required library
import cv2

# read the input image
image = cv2.imread('dummy.jpg')

# define the alpha and beta
alpha = 1.5 # Contrast control
beta = 1 # Brightness control

# call convertScaleAbs function
adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# display the output image
cv2.imshow('adjusted', adjusted)
cv2.waitKey()
cv2.destroyAllWindows()