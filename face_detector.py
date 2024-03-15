import cv2

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml") #Stores the xml file

img=cv2.imread('photo.jpg')
gray_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # COLOR_BGR2GRAY converts the image to grayscale image

faces=face_cascade.detectMultiScale(gray_img, #detectMultiScale uses the xml file to detect the face
scaleFactor=1.05, # Narrows the image for better accuracy of detecting objects
minNeighbors=5)

for x, y,w, h in faces: #Creates the rectangle around the face
   img=cv2.rectangle(img, (x,y), (x+w,y+h),(0,255,0),3) #4 paramters: image, point1, point2, rectangle color, thickness

print(type(faces)) #type: array
print(faces) #Tells the point "face" starts and length

resized=cv2.resize(img, (int(img.shape[0]/3), int(img.shape[1]/3)))

cv2.imshow("Gray",resized)
cv2.waitKey(0)
cv2.destroyAllWindows()




