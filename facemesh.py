import mediapipe as mp

face_mesh = mp.solutxions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=2)


def get_face_landmarks(face_mesh, img):
    # TODO: mediapipe results not stable, even with static_image_mode?
    for i in range(20):
        mesh_result = face_mesh.process(img)
    return mesh_result
