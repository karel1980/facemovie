import math
import sys
import time

import cv2
import mediapipe as mp
import numpy as np

import project

LEFT_EYE = 33
RIGHT_EYE = 263


class Slide:
    def __init__(self, img_path, input_eyes, output_eyes, border_color, border_thickness):
        self.img_path = img_path
        self.input_eyes = input_eyes
        self.output_eyes = output_eyes
        self.border_color = border_color
        self.border_thickness = border_thickness

    def get_left_eye_input(self):
        return self.input_eyes[0]

    def get_input_eye_angle(self):
        input_eye_vec = self.input_eyes[1] - self.input_eyes[0]
        target_eye_vec = np.array([100, 0])
        angle = calculate_angle(input_eye_vec, target_eye_vec)
        return angle

    def get_scale(self):
        return np.linalg.norm(self.output_eyes[1] - self.output_eyes[0]) / np.linalg.norm(
            self.input_eyes[1] - self.input_eyes[0])

    def __str__(self):
        return "Slide(img_path=%s, input_eyes=%s, output_eyes=%s, border_color=%s, border_thickness=%s)" % (
            self.img_path, self.input_eyes, self.output_eyes, self.border_color, self.border_thickness
        )


def main():
    if len(sys.argv) != 3:
        print("usage: %s <project.json> <output.mov>")
        return

    project_file = sys.argv[1]
    output_file = sys.argv[2]

    proj = project.Project.load(project_file)
    proj.settings.output_path = output_file

    generate_facemovie(proj, ProgressWriter().on_progress)


class ProgressWriter:
    def __init__(self, print_interval=1):
        self.next_print_time = time.time()
        self.print_interval = print_interval

    def on_progress(self, *args):
        state = args[0]
        params = args[1:]
        if state == 'generating':
            now = time.time()
            if self.next_print_time < now:
                self.next_print_time += self.print_interval
                print("frame %s / %s" % params)
        else:
            print(state)


def generate_facemovie(project, progress_callback=None):
    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=10)
    settings = project.settings
    slides = calculate_slides(project, face_mesh)
    w, h = (settings.target_size[1], settings.target_size[0])
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    vid = cv2.VideoWriter(settings.output_path, fourcc, settings.fps, (w, h))
    frame_num = 0
    total_frames = calculate_total_frames(settings, len(slides))
    for frame in generate_frames(settings, slides):
        frame_num += 1
        if progress_callback is not None:
            progress_callback('generating', frame_num, total_frames)

        vid.write(frame)
    vid.release()
    if progress_callback != None:
        progress_callback('finished')


def calculate_slides(project, face_mesh):
    """ returns a list of Slides """

    result = []
    for input_slide in project.slides:
        if input_slide.face_rect is None:
            continue
        img = cv2.imread(input_slide.path)
        if img is None:
            print("Couldn't load ", input_slide.path)
            continue
        mesh_result = get_face_landmarks(face_mesh, img)

        if not mesh_result:
            print("no face mesh result")
            continue

        if not mesh_result.multi_face_landmarks:
            print("No faces found in %s. Skipped." % input_slide.path)
            continue

        src = get_src_points(find_nearest_face(img.shape, input_slide, mesh_result), img.shape)
        dst = get_target_points(src, project.settings)
        result.append(Slide(input_slide.path, src, dst, (255, 255, 255, 255), 10))

    return result


def find_nearest_face(input_shape, input_slide, mesh_result):
    faces = mesh_result.multi_face_landmarks

    nearest_face = faces[0]
    nearest_dist = find_distance_from_input_slide(input_shape, input_slide, nearest_face)

    for face in faces[1:]:
        dist = find_distance_from_input_slide(input_shape, input_slide, face)
        if dist < nearest_dist:
            nearest_dist = dist
            nearest_face = face

    return nearest_face


def find_distance_from_input_slide(input_shape, input_slide, face):
    w, h = input_shape[1], input_shape[0]
    input_slide_center = np.array([(input_slide.face_rect.pt1.x + input_slide.face_rect.pt2.x) / 2,
                                   (input_slide.face_rect.pt1.y + input_slide.face_rect.pt2.y) / 2])

    top = min([lm.y for lm in face.landmark])
    bottom = max([lm.y for lm in face.landmark])
    left = min([lm.x for lm in face.landmark])
    right = min([lm.x for lm in face.landmark])

    face_center = np.array((w * (right + left) / 2, h * (top + bottom) / 2))

    return np.linalg.norm(face_center - input_slide_center)


def get_transformation_points(img, face, target_shape):
    src = get_src_points(face, img.shape)
    dst = get_target_points(src, target_shape)

    return src, dst


def get_src_points(face_landmarks, input_shape):
    h, w = input_shape[0], input_shape[1]
    left_eye = np.array([face_landmarks.landmark[LEFT_EYE].x * w, face_landmarks.landmark[LEFT_EYE].y * h])
    right_eye = np.array([face_landmarks.landmark[RIGHT_EYE].x * w, face_landmarks.landmark[RIGHT_EYE].y * h])
    return np.float32([left_eye, right_eye, (0, 0)])


def get_target_points(src, settings):
    target_height, target_width = settings.target_size[0], settings.target_size[1]
    left_eye = src[0]
    right_eye = src[1]
    target_left_eye = np.array([(target_width / 2) - 30, (target_height * 2 / 5)])
    target_right_eye = np.array([(target_width / 2) + 30, (target_height * 2 / 5)])
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
            img = cv2.imread(current_slide.img_path)
            current_face_img = orient_image(img, current_slide, settings.target_size)

            added_to_backdrop = False

        fade_start = slide_duration * slide_idx
        fade_end = slide_duration * slide_idx + settings.fade_in_millis

        alpha = (frame_time_millis - fade_start) / (fade_end - fade_start)
        alpha = max(0, min(1, alpha))

        if alpha < 1:
            frame = overlay_image(backdrop.copy(), current_face_img, alpha)

        if alpha == 1 and not added_to_backdrop:
            backdrop = overlay_image(backdrop.copy(), current_face_img, alpha)
            added_to_backdrop = True
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

    base = (1.0 - (mask * alpha)) * base + (mask * alpha) * overlay_noalpha
    return base.astype(np.uint8)


def orient_image(img, slide, target_shape):
    scale = slide.get_scale()
    border_thickness = int(10 / scale)  # todo: 20 = border thickness in output; make this a setting
    img = cv2.copyMakeBorder(img, border_thickness, border_thickness, border_thickness, border_thickness,
                             cv2.BORDER_CONSTANT, value=(255, 255, 255, 255))

    input_eye = slide.input_eyes[0] + border_thickness
    img_with_alpha = add_alpha_channel(img)
    affine_transformation_matrix = cv2.getRotationMatrix2D(input_eye, slide.get_input_eye_angle() * 180 / math.pi,
                                                           scale)
    affine_transformation_matrix[:, 2] += slide.output_eyes[0] - input_eye

    dsize = (target_shape[1], target_shape[0])
    result = np.zeros((target_shape[0], target_shape[1], 4))

    # Note: cv2.BORDER_TRANSPARENT didn't work as expected,
    # so manually set cv2.BORDER_CONSTANT with a transparent borderValue
    result = cv2.warpAffine(img_with_alpha, affine_transformation_matrix, dsize, result, flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    return result


def add_alpha_channel(img):
    b_channel, g_channel, r_channel = cv2.split(img)
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255
    return cv2.merge((b_channel, g_channel, r_channel, alpha_channel))


if __name__ == "__main__":
    main()
