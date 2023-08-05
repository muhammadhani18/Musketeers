import numpy as np
import cv2


image_path = './man2.jpg'


# ####################################################### DETECT SMILE ######################################################

# read input image
def detect_smile():
    
    img = cv2.imread(image_path)

    # convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # read haar cascade for face detection
    face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

    # read haar cascade for smile detection
    smile_cascade = cv2.CascadeClassifier('./haarcascade_smile.xml')

    # Detects faces in the input image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    print('Number of detected faces:', len(faces))

    if len(faces) == 0:
      print("Smile not detected.")

# loop over all the faces detected
    for (x,y,w,h) in faces:
    
    # draw a rectangle in a face
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        cv2.putText(img, "Face", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        # detecting smile within the face roi
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
        if len(smiles) > 0:
            print(f"Smile detected. Score: {len(smiles)}")
    
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2)
                cv2.putText(roi_color, "smile", (sx, sy),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            print(f"Smile not detected. Score: {len(smiles)}")

    # Display an image in a window
    cv2.imshow('Smile Image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


########################################### MOUTH CLOSED OR NOT #########################################

def mouth_aspect_ratio(mouth):

  top_lip = mouth[0:6]
  bottom_lip = mouth[6:12]
  width = np.linalg.norm(top_lip[0] - bottom_lip[0])
  height = np.linalg.norm(top_lip[1] - bottom_lip[1])
  return width / height

def mouth_closed_or_not():
  image = cv2.imread(image_path)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  faces = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml").detectMultiScale(
      gray,
      scaleFactor=1.1,
      minNeighbors=3,
      minSize=(35, 35))

  for (x, y, w, h) in faces:
    mouth = mouth_aspect_ratio(gray[y:y + h, x:x + w])
  
  if mouth > 0.6:
    print(f"Mouth is open. Score: {mouth}")
    text = "Mouth is open"
    color = (0, 0, 255)
  else:
    print(f"Mouth is closed. Score: {mouth}")
    text = "Mouth is closed"
    color = (255, 0, 0)

    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

  cv2.imshow("Image", image)
  cv2.waitKey(0)


########################################## USER FACING IMAGE AND EYES FACING CAMERA OR NOT #########################################


#check if user is facing the camera and their eyes are facing the camera or not

def is_user_facing_camera_with_eyes(image):
  
  if image is None:
    return False

  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  eyes = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml").detectMultiScale(
      gray,
      scaleFactor=1.1,
      minNeighbors=5,
      minSize=(30, 30))

  if len(eyes) == 2:
    center_of_eyes = (eyes[0][0] + eyes[1][0]) / 2
    center_of_image = (image.shape[1] / 2, image.shape[0] / 2)

    
    angle = angle = np.arctan2(center_of_eyes - center_of_image[1], center_of_eyes - center_of_image[0])
    angle = np.rad2deg(angle)
   
    if angle > 45 and angle < 135:
      return True, True
    else:
      return True, False
  else:
    return False, False

def eyes_and_face_facing_camera_or_not():
  image = cv2.imread(image_path)
  is_facing, is_eyes_facing = is_user_facing_camera_with_eyes(image)

  if is_facing and is_eyes_facing:
    text = "The user is facing the camera and their eyes are facing the camera. Score: 1.0"
    print(text)
    color = (0, 255, 0)
    
  elif is_facing and not is_eyes_facing:
    text = "The user is facing the camera but their eyes are not facing the camera. Score: 0.0"
    print(text)
    color = (255, 100, 0)
  
  else:
    text = "The user is not facing the camera and their eyes are also not facing the camera. Score: 0.0"
    print(text)
    color = (255, 0, 0)

  cv2.putText(image, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
  cv2.imshow("Image", image)
  cv2.waitKey(0)




if __name__ == "__main__":
    print("\n---------------------------------------------\n")
    mouth_closed_or_not()
    detect_smile()
    eyes_and_face_facing_camera_or_not()
    print("\n---------------------------------------------\n")
    
    
    
    