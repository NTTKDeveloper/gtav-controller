import cv2 

image = cv2.imread("./data/circle_image_1.png")

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

median_blurred = cv2.medianBlur(image, 1)  # Kernel size = 5

# Làm mờ Gaussian làm mờ mà vẫn giữ chi tiết 
gaussian_blurred = cv2.GaussianBlur(median_blurred, (1,1), 0)  # Kernel size (5x5) và sigma = 0

# Phát hiện cạnh bằng Canny
edges = cv2.Canny(gaussian_blurred, 0, 15)

cv2.imshow("Display", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()