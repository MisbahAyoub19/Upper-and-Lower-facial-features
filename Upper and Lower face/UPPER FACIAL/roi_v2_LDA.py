import os

import cv2



import math


# Getting to know blink ratio

counter = 1

def midpoint(point1, point2):
 return (point1.x + point2.x) / 2, (point1.y + point2.y) / 2


def euclidean_distance(point1, point2):
 return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def get_eyebrow_points(eye_points, facial_landmarks):
    eb_point1 = (facial_landmarks.part(eye_points[0]).x,
                 facial_landmarks.part(eye_points[0]).y)

    eb_point2 = (facial_landmarks.part(eye_points[1]).x,
                 facial_landmarks.part(eye_points[1]).y)

    eb_point3 = (facial_landmarks.part(eye_points[2]).x,
                 facial_landmarks.part(eye_points[2]).y)

    eb_point4 = (facial_landmarks.part(eye_points[3]).x,
                 facial_landmarks.part(eye_points[3]).y)

    eb_point5 = (facial_landmarks.part(eye_points[4]).x,
                 facial_landmarks.part(eye_points[4]).y)

    Midpoint_center = midpoint(facial_landmarks.part(eye_points[9]),
                               facial_landmarks.part(eye_points[10]))

    vlength_p1 = euclidean_distance(eb_point1, Midpoint_center)  # vertical length of point one of eyebrow landmar (17)
    vlength_p2 = euclidean_distance(eb_point2, Midpoint_center)
    vlength_p3 = euclidean_distance(eb_point3, Midpoint_center)
    vlength_p4 = euclidean_distance(eb_point4, Midpoint_center)
    vlength_p5 = euclidean_distance(eb_point5, Midpoint_center)

    return vlength_p1, vlength_p2, vlength_p3, vlength_p4, vlength_p5


def get_blink_ratio(eye_points, facial_landmarks):
  # loading all the required points



  corner_left = (facial_landmarks.part(eye_points[0]).x,
                 facial_landmarks.part(eye_points[0]).y)
  corner_right = (facial_landmarks.part(eye_points[4]).x,
                  facial_landmarks.part(eye_points[4]).y)

  Midpoint_left = midpoint(facial_landmarks.part(eye_points[0]),
                          facial_landmarks.part(eye_points[5]))
  Midpoint_right = midpoint(facial_landmarks.part(eye_points[4]),
                          facial_landmarks.part(eye_points[8]))



  center_top = midpoint(facial_landmarks.part(eye_points[0]),
                        facial_landmarks.part(eye_points[4]))
  center_bottom = midpoint(facial_landmarks.part(eye_points[5]),
                           facial_landmarks.part(eye_points[8]))

  # calculating distance
  horizontal_length = euclidean_distance(Midpoint_left, Midpoint_right)
  vertical_length = euclidean_distance(center_top, center_bottom)


  ratio = horizontal_length / vertical_length

  return ratio


# Changed definition to include only one path, rather than paths for eye, eyebrow etc
def rect1(predictor,i,shotname,picturepath,imgPath ,SheetPath,FeaturesPath, shapeArray, borders, img, dets):

    # Function for ROI detection
 #print(imgPath)

 # create a file PATH to store Eye Path picture
 # Rather than a specific path, we are going to choose what path to pass in
    # So then the path will be general
 if not os.path.exists(imgPath):
    os.mkdir(os.path.join(imgPath))
 cur_dir =os.path.join(imgPath, shotname)
 if not os.path.exists(cur_dir):
     os.mkdir(os.path.join(cur_dir))

 #print('inside the eye')

 BLINK_RATIO_THRESHOLD = 2.4
    # these landmarks are based on the image above
 left_eye_landmarks = [17,18,19,20,21, 36,37,38, 39, 40, 41]
 right_eye_landmarks = [22,23,24,25,26,27,42, 43, 44, 45, 46, 47]

 #Left_nostrill_ratio = [27,28,29,30,33,34,35]
 #right_nostril_ratio = [27,28,29,30,31,32,33]

 x = [0]
 y = [0]


 for k, d in enumerate(dets):
     # Go through and calculate right eye (on person)
     #print (k,d)
     shape = predictor(img,d)

     # pass in and use new shapepoints array
     #print('using shape points ', shapeArray)

     leftpoint = shape.part(shapeArray[0]).x
     rightpoint = shape.part(shapeArray[1]).x
     toppoint = shape.part(shapeArray[2]).y
     buttompoint = shape.part(shapeArray[3]).y
     # leftpoint_mouth = shape.part(shapeArray[5]).x
     # #rightpoint_mouth = shape.part(shapeArray[54]).x
     # toppoint_mouth = shape.part(shapeArray[6]).y
     # #buttompoint_mouth = shape.part(shapeArray[58]).y


     ROIval= img[toppoint+borders[0]:buttompoint+borders[1], leftpoint+borders[2]:rightpoint+borders[3]]
     ROIpath = cur_dir + '/' + str('%02d' % i) + '.jpg'
     cv2.imwrite(ROIpath, ROIval)

     count = 1
     total = 0

     # landmarks = predictor(img, d)
     # Calculating blink ratio for one eye-----
     left_eye_ratio = get_blink_ratio(left_eye_landmarks, shape)
     right_eye_ratio = get_blink_ratio(right_eye_landmarks, shape)


     left_eyebrow_dist = get_eyebrow_points(left_eye_landmarks, shape)
     right_eyebrow_dist = get_eyebrow_points(right_eye_landmarks, shape)


     #print(left_eyebrow_dist, "left eyebrow distance")
     #print(Right_eyebrow_dist, "Right_eyebrow distance")
     #b_ratio = list(pleft[0])
     blink_ratio = (left_eye_ratio + right_eye_ratio) / 2




     # Extract gabor features of right eye
     return(i,blink_ratio,left_eye_ratio,right_eye_ratio,left_eyebrow_dist,right_eyebrow_dist)