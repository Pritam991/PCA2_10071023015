import cv2 as cv 


img = cv.imread("P:\\PCA2_10071023015\\nature.jpg")


if img is not None:
    cv.imshow("2.py", img)
    cv.waitKey(0)
    cv.destroyAllWindows()
else:
    print("Error loading the image. Please check the file path.")
