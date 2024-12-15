import cv2


#path=input("Enter the image path and name of the image:")
#print("you enter this:",path)

img1=cv2.imread(r"C:\Users\akash\Pictures\download.jpeg",1)

img1=cv2.resize(img1,(800,700))
img1=cv2.flip(img1, 1)
cv2.imshow("converted image",img1)

cv2.waitKey()
cv2.destroyAllWindows()
"""
k = cv2.waitKey(0)
if k==ord("s"):
    cv2.imwrite("C:\\Users\\akash\\Pictures\\Infosys\\output1.png", img1)
else:
    cv2.destroyAllWindows()
    """