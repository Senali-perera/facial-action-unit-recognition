import os
from collections import OrderedDict
import dlib
import cv2
import numpy as np
from PIL import Image

# define a dictionary that maps the indexes of the facial
FACIAL_LANDMARKS_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    # ("inner_mouth", (60, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    # ("jaw", (0, 17))
])


def resize(image, width, inter=cv2.INTER_AREA):
    if isinstance(image, Image.Image):  # Check if it's a PIL image
        image = np.array(image)  # Convert to NumPy array

    # initialize the dimensions of the image to be resized and
    # grab the image size
    (h, w) = image.shape[:2]

    # calculate the ratio of the width and construct the dimensions
    r = width / float(w)
    dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def visualize_facial_landmarks(image, shape):
    output = image.copy()

    for (name, (i, j)) in FACIAL_LANDMARKS_IDXS.items():
        (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
        if w > 0 and h > 0:
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # return the output image
    return output


def facial_landmarks_detection(image):
    # initialize dlib's face detector and then predict the facial landmarks
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./visualize_facial_landmarks/shape_predictor_68_face_landmarks.dat')

    # load the input image, resize it, and convert it to grayscale
    # image = cv2.imread('example_01.jpeg')
    image = cv2.imread(image)
    resized_image = resize(image, 500)
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray_image, 1)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray_image, rect)
        coords = np.zeros((shape.num_parts, 2), dtype="int")

        # loop over all facial landmarks and convert them
        # to a 2-tuple of (x, y)-coordinates
        for i in range(0, shape.num_parts):
            coords[i] = (shape.part(i).x, shape.part(i).y)

        shape = coords

        # visualize all facial landmarks
        output = visualize_facial_landmarks(resized_image, shape)
        if not os.path.exists('./static/images/'):
            os.makedirs('./static/images/')
        cv2.imwrite('./static/images/facial_landmark_file.jpg', output)
        # cv2.imshow("Image", output)
        # cv2.waitKey(0)
        return output

    # cv2.waitKey(0)


# facial_landmarks_detection('example_01.jpeg')