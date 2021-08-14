import sys
import time

import cv2
import mediapipe as mp
import numpy as np

import project

LEFT_EYE = 7
RIGHT_EYE = 249


class Settings:
    def __init__(self, fade_in_millis=500, show_millis=2000, fade_out_millis=1000, target_size=(768, 1024),
                 fps = 25):
        self.fade_in_millis = fade_in_millis
        self.show_millis = show_millis
        self.fade_out_millis = fade_out_millis
        self.target_size = target_size
        self.fps = fps


def main():
    if len(sys.argv) != 3:
        print("usage: %s <project.json> <output.mov>")
        return

    project_file = sys.argv[1]
    output_file = sys.argv[2]

    proj = project.Project.load(project_file)
    settings = Settings(fps=60)  # TODO: make settings part of the project?

    generate_facemovie(proj, settings, output_file)


def generate_facemovie(project, settings, output_file):
    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=10)
    slides = calculate_slide_data(project, settings, face_mesh)
    next_print_time = time.time()
    total_frames = calculate_total_frames(settings, len(slides))
    w, h = (settings.target_size[1], settings.target_size[0])
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    vid = cv2.VideoWriter(output_file, fourcc, settings.fps, (w, h))
    frame_num = 0
    for frame in generate_frames(settings, slides):
        frame_num += 1
        if time.time() > next_print_time:
            print("frame %s / %s" % (frame_num, total_frames))
            next_print_time += 1

        vid.write(frame)
    print("done")
    vid.release()


def calculate_slide_data(project, settings, face_mesh):
    """ returns a list of (path, src, dst) tuples, where src and dst are reference points used for orienting and scaling the source image to the output shape"""

    result = []
    for item in project.playlist:
        img = cv2.imread(item[0])
        mesh_result = get_face_landmarks(face_mesh, img)

        if not mesh_result:
            print("no face mesh result")
            continue

        if not mesh_result.multi_face_landmarks:
            print("No faces found in %s. Skipped." % (item[0]))
            continue

        # TODO: if there are multiple faces, use the one closest to item[1]
        src = get_src_points(mesh_result.multi_face_landmarks[0], img.shape)
        dst = get_target_points(src, settings.target_size)
        result.append((item[0], src, dst))

    return result


def find_nearest_face(img, target_location, faces):
    h, w = img.shape[0], img.shape[1]

    if len(faces) == 0:
        return None
    return faces[0]


def get_transformation_points(img, face, target_shape):
    src = get_src_points(face, img.shape)
    dst = get_target_points(src, target_shape)

    return src, dst


def get_src_points(face_landmarks, input_shape):
    h, w = input_shape[0], input_shape[1]
    left_eye = np.array([face_landmarks.landmark[LEFT_EYE].x * w, face_landmarks.landmark[LEFT_EYE].y * h])
    right_eye = np.array([face_landmarks.landmark[RIGHT_EYE].x * w, face_landmarks.landmark[RIGHT_EYE].y * h])
    return np.float32([left_eye, right_eye, (0, 0)])


def get_target_points(src, target_shape):
    target_height, target_width = target_shape[0], target_shape[1]
    left_eye = src[0]
    right_eye = src[1]
    target_left_eye = np.array([(target_width / 2) - 30, (target_height / 2)])
    target_right_eye = np.array([(target_width / 2) + 30, (target_height / 2)])
    target_origin = calculate_target_origin_location(left_eye, right_eye, target_left_eye, target_right_eye)
    return np.float32([target_left_eye, target_right_eye, target_origin])


def calculate_target_origin_location(left, right, target_left, target_right):
    alpha = calculate_angle(right - left, -left)

    ab = right - left
    AB = target_right - target_left
    rotMatrix = np.array([[np.cos(alpha), -np.sin(alpha)],
                          [np.sin(alpha), np.cos(alpha)]])

    AB_unit = AB.dot(rotMatrix) / np.linalg.norm(AB)
    AC = AB_unit * np.linalg.norm(left) * np.linalg.norm(AB) / np.linalg.norm(ab)
    return target_left + AC


# def calculate_target_origin_location(left, right, target_left, target_right):
#     # alpha = calculate_angle(left, right)
#     alpha = 90
#
#     # a = left, B = right, c = (0,0)
#     # A = target left, B = target right, C = target origin
#
#     AB = target_right - target_left
#     rotMatrix = np.array([[np.cos(alpha), -np.sin(alpha)],
#                           [np.sin(alpha), np.cos(alpha)]])
#
#     AC = AB.dot(rotMatrix) * np.linalg.norm(left) * np.linalg.norm(right - left) / np.linalg.norm(target_right - target_left) / np.linalg.norm(target_right - target_left)
#     return left - AC


def calculate_angle(a, b):
    # calculates angle between 2 vectors in radians
    left_unit = a / np.linalg.norm(a)
    right_unit = b / np.linalg.norm(b)
    dot_product = np.dot(left_unit, right_unit)
    return np.arccos(dot_product)


def generate_frames(settings, slides):
    slide_duration = calculate_slide_duration(settings)

    backdrop = np.zeros((settings.target_size[0], settings.target_size[1], 3))
    current_slide = None
    current_face_img = None
    added_to_backdrop = False

    # draw an ugly blue border because we can (BGR color space!)
    # cv2.rectangle(backdrop, (20, 20), (settings.target_size[1] - 20, settings.target_size[0] - 20), (255, 0, 0), 40)

    total_frames = calculate_total_frames(settings, len(slides))

    for frame_num in range(0, total_frames):
        frame_time_millis = (frame_num * 1000 / settings.fps)
        slide_idx = min(int(frame_time_millis / slide_duration), len(slides) - 1)

        if slides[slide_idx] != current_slide:
            current_slide = slides[slide_idx]
            print('loading face', current_slide[0])
            img = cv2.imread(current_slide[0])
            current_face_img = orient_image(img, current_slide[1], current_slide[2], settings.target_size)

            added_to_backdrop = False

        fade_start = slide_duration * slide_idx
        fade_end = slide_duration * slide_idx + settings.fade_in_millis

        # print("fade_start", fade_start)
        # print("fade_end", fade_end)
        # print("frame_time_millis", frame_time_millis)
        alpha = (frame_time_millis - fade_start) / (fade_end - fade_start)
        alpha = max(0, min(1, alpha))
        # print("alpha", alpha)

        if alpha < 1:
            # print("frame = backdrop + current@", alpha)
            frame = overlay_image(backdrop.copy(), current_face_img, alpha)

        if alpha == 1 and not added_to_backdrop:
            # print("adding full image to backdrop")
            backdrop = overlay_image(backdrop.copy(), current_face_img, alpha)
            added_to_backdrop = True
            # print("frame = backdrop")
            frame = backdrop

        yield frame


def calculate_total_frames(settings, num_slides):
    total_duration = calculate_total_duration(settings, num_slides)
    total_frames = int(total_duration * settings.fps / 1000)
    return total_frames


def calculate_total_duration(settings, num_slides):
    slide_duration = calculate_slide_duration(settings)
    total_duration = (num_slides * slide_duration) + settings.fade_out_millis
    return total_duration


def calculate_slide_duration(settings):
    return settings.fade_in_millis + settings.show_millis


def get_face_landmarks(face_mesh, img):
    for i in range(
            20):  # processing multiple times gives better resuts because sometimes the model needs a few tries to settle on the right solution
        face_mesh.process(img)
    return face_mesh.process(img)


def show_points(img, pts, color=(255, 255, 255)):
    cv2.circle(img, pts[0].astype(int), 2, color, 3)
    cv2.circle(img, pts[1].astype(int), 2, color, 3)
    cv2.circle(img, pts[2].astype(int), 2, color, 3)


def overlay_image(base, overlay, alpha=1.0):
    overlay_noalpha = overlay[:, :, :3]
    mask = overlay[:, :, 3:] / 255
    # print(base.shape)
    # print(overlay.shape)
    # print(mask.shape)

    base = (1.0 - (mask * alpha)) * base + (mask * alpha) * overlay_noalpha
    return base.astype(np.uint8)


def single():
    proj = project.Project.load('project.json')
    settings = Settings()
    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=10)
    slides = calculate_slide_data(proj, settings, face_mesh)

    backdrop = np.zeros((settings.target_size[0], settings.target_size[1], 3))

    slide1 = slides[0]
    print(slide1[0])
    print(slide1[1])
    print(slide1[2])

    img1 = cv2.imread(slide1[0])
    print("sss", img1.shape)

    face1 = orient_image(img1, slide1[1], slide1[2], settings.target_size)

    slide2 = slides[1]
    img2 = cv2.imread(slide2[0])
    face2 = orient_image(img2, slide2[1], slide2[2], settings.target_size)

    while True:

        cv2.imshow('backdrop', backdrop)
        cv2.imshow('face1', face1)
        cv2.imshow('face2', face2)
        key = cv2.waitKey(0)
        if key == ord('q'):
            return


def oriented_face(face_mesh, img):
    target_shape = 1024, 1280
    faces = get_face_landmarks(face_mesh, img)
    face_landmarks = faces.multi_face_landmarks[0]
    src = get_src_points(face_landmarks, img.shape)
    dst = get_target_points(src, target_shape)
    img, result = orient_image(img, src, dst, target_shape)
    return img, result


def orient_image(img, src, dst, target_shape):
    img_with_alpha = add_alpha_channel(img)
    M = cv2.getAffineTransform(src, dst)
    result = np.zeros((target_shape[0], target_shape[1], 4))
    dsize = (target_shape[1], target_shape[0])

    # Note: cv2.BORDER_TRANSPARENT didn't work as expected,
    # so manually set cv2.BORDER_CONSTANT with a transparent borderValue
    result = cv2.warpAffine(img_with_alpha, M, dsize, result, flags = cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    return result


def add_alpha_channel(img):
    b_channel, g_channel, r_channel = cv2.split(img)
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255
    return cv2.merge((b_channel, g_channel, r_channel, alpha_channel))


if __name__ == "__main__":
    # single()
    main()
