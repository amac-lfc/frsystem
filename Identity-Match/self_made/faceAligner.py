import cv2
import numpy as np

def faceAligner(image, facial_landmarks, face_location):
  """
  This function transforms a face into aligned face for better recognition

  ## Args:
      image (array): array of pixels
      facial_landmarks (dict): dictionary of facial landmarks i.e. left and right eye coordinates
      face_location (tuple): tuple consisting of face location rectangle on an image (x, y, width, height)

  ## Returns:
      array: transformed face image
  """

  # Let's find and angle of the face. First calculate 
  # the center of left and right eye by using eye landmarks.

  (x1, y1, width, height) = face_location

  leftEyeCenter = facial_landmarks['left_eye']
  rightEyeCenter = facial_landmarks['right_eye']

  leftEyeCenter = np.array(leftEyeCenter).astype("int")
  rightEyeCenter = np.array(rightEyeCenter).astype("int")

  # find and angle of line by using slope of the line.
  dY = rightEyeCenter[1] - leftEyeCenter[1]
  dX = rightEyeCenter[0] - leftEyeCenter[0]
  angle = np.degrees(np.arctan2(dY, dX))

  # to get the face at the center of the image,
  # set desired left eye location. Right eye location 
  # will be found out by using left eye location.
  # this location is in percentage.
  desiredLeftEye=(0.35, 0.35)

  #Set the croped image(face) size after rotaion.
  desiredFaceWidth = width
  desiredFaceHeight = height

  desiredRightEyeX = 1.0 - desiredLeftEye[0]
  
  # determine the scale of the new resulting image by taking
  # the ratio of the distance between eyes in the *current*
  # image to the ratio of distance between eyes in the
  # *desired* image
  dist = np.sqrt((dX ** 2) + (dY ** 2))
  desiredDist = (desiredRightEyeX - desiredLeftEye[0])
  desiredDist *= desiredFaceWidth
  scale = desiredDist / dist

  # compute center (x, y)-coordinates (i.e., the median point)
  # between the two eyes in the input image
  eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
      (leftEyeCenter[1] + rightEyeCenter[1]) // 2)

  # grab the rotation matrix for rotating and scaling the face
  M = cv2.getRotationMatrix2D(eyesCenter, angle, 1)

  # update the translation component of the matrix
  tX = desiredFaceWidth * 0.5
  tY = desiredFaceHeight * desiredLeftEye[1]
  M[0, 2] += (tX - eyesCenter[0])
  M[1, 2] += (tY - eyesCenter[1])

  # apply the affine transformation
  (w, h) = (desiredFaceWidth, desiredFaceHeight) 
          
  output = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

  return output