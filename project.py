import json
import sys

import cv2
import face_recognition


class Image:
    def __init__(self, id, path, locations):
        self.id = id
        self.path = path
        self.locations = locations


class Project:
    def __init__(self, playlist=None):
        if playlist is None:
            playlist = []
        self.playlist = playlist

    def add_image(self, path, location = None):
        img = cv2.imread(path)
        if location is None:
            locations = face_recognition.face_locations(img)
            self.add_slide(path, None if len(locations) == 0 else locations[0])

    def add_slide(self, image_path, location):
        self.playlist.append((image_path, location))

    def save(self, path):
        data = dict()
        data["playlist"] = self.playlist
        if path is None:
            print(json.dumps(data))
        else:
            with open(path, 'w') as outfile:
                outfile.write(json.dumps(data))

    @staticmethod
    def load(path):
        with open(path) as infile:
            data = json.load(infile)
            playlist = data["playlist"]
            return Project(playlist=playlist)


def main():
    image_paths = sys.argv[1:]

    project = Project()
    for img_path in image_paths:
        project.add_image(img_path)


if __name__ == "__main__":
    main()
