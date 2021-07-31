import sys

import cv2
import mediapipe as mp
import numpy as np

LEFT_EYE_CORNER_LM = 7
RIGHT_EYE_CORNER_LM = 249
CHIN_LM = 152


def get_face_landmarks(face_mesh, img):
    for i in range(20):
        mesh_result = face_mesh.process(img)
    return mesh_result


def determine_angle(a, b):
    unit_a = a / np.linalg.norm(a)
    unit_b = b / np.linalg.norm(b)
    dot_product = np.dot(unit_a, unit_b)
    return np.arccos(dot_product)


def rotate_vector(vector, angleA_rad):
    return np.array([[np.cos(angleA_rad), -np.sin(angleA_rad)], [np.sin(angleA_rad), np.cos(angleA_rad)]]).dot(vector)


def process_image(face_mesh, img):
    mesh_result = get_face_landmarks(face_mesh, img)
    left_eye_lm = mesh_result.multi_face_landmarks[0].landmark[LEFT_EYE_CORNER_LM]
    right_eye_lm = mesh_result.multi_face_landmarks[0].landmark[RIGHT_EYE_CORNER_LM]
    chin_lm = mesh_result.multi_face_landmarks[0].landmark[CHIN_LM]

    output_height = 1024
    output_width = int(1024 * 3 / 2)

    input_height = img.shape[0]
    input_width = img.shape[1]

    left_orig = np.array([input_width * left_eye_lm.x, input_height * left_eye_lm.y])
    right_orig = np.array([input_width * right_eye_lm.x, input_height * right_eye_lm.y])
    chin_orig = np.array([input_width * chin_lm.x, input_height * chin_lm.y])

    left_target = np.array([output_width * 0.47, output_height / 2])
    right_target = np.array([output_width * 0.53, output_height / 2])

    chin_target = calculate_chin_target(chin_orig, left_orig, left_target, right_orig, right_target)

    # note: it was probably easier to manually create the transformation matrix based just on the eye coordinates
    # try that later.
    M = cv2.getAffineTransform(
        np.float32([left_orig, right_orig, chin_orig]),
        np.float32([left_target, right_target, chin_target]))
    img_alpha = add_alpha_channel(img)
    overlay = cv2.warpAffine(img_alpha, M, (output_width, output_height))

    return img, overlay


def calculate_chin_target(chin_orig, left_orig, left_target, right_orig, right_target):
    AB = right_orig - left_orig
    AC = chin_orig - left_orig
    angleA = determine_angle(AB, AC)
    ab = right_target - left_target
    ab_rotated = rotate_vector(ab, angleA)
    ac = ab_rotated * np.linalg.norm(AC) / np.linalg.norm(AB)
    chin_target = left_target + ac
    return chin_target


def add_alpha_channel(img):
    channel1, channel2, channel3 = cv2.split(img)
    alpha_channel = np.ones(channel1.shape, dtype=channel1.dtype) * 255
    return cv2.merge((channel1, channel2, channel3, alpha_channel))


def load_image(num):
    img = cv2.imread('examples/me/photo%03d.jpg' % (num + 1))
    return img


def main():
    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=2)
    mp_draw = mp.solutions.drawing_utils

    current_image = 3
    current_landmark = 0
    img, result = process_image(face_mesh, load_image(current_image))

    while True:
        cv2.imshow('input', img)
        cv2.imshow('result', result)

        key = cv2.waitKey(1)
        if key == ord('a'):
            current_image += 1
            current_image %= 4
            img, result = process_image(face_mesh, load_image(current_image))
        elif key == ord('n'):
            current_landmark += 1
            current_landmark %= 500
        elif key == ord('p'):
            current_landmark += 499
            current_landmark %= 500
        elif key == ord('q'):
            sys.exit(0)


def draw_face_mesh(img, mesh_result):
    if mesh_result.multi_face_landmarks:
        # print("got landmarks", mesh_result.multi_face_landmarks)
        for face in mesh_result.multi_face_landmarks:
            mp_draw.draw_landmarks(img, face)


def draw_eye_landmarks(img, mesh_result):
    if not mesh_result.multi_face_landmarks:
        return

    landmarks = mesh_result.multi_face_landmarks[0]
    draw_landmarks(img, landmarks)


def draw_landmarks(img, landmarks):
    for i in range(len(landmarks.landmark)):
        draw_single_landmark(img, landmarks.landmark[i], str(i))


def draw_single_landmark(img, landmark, text=None):
    height = img.shape[0]
    width = img.shape[1]
    lm_pos = (int(landmark.x * width), int(landmark.y * height))
    cv2.circle(img, lm_pos, 1, (0, 255, 0), 2)
    if text is not None:
        cv2.putText(img, text, lm_pos, cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255))


if __name__ == "__main__":
    main()
