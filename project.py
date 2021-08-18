import json


class Slide:
    def __init__(self, path, face_rect):
        self.path = path
        self.face_rect = face_rect


class Rect:
    def __init__(self, pt1, pt2):
        self.pt1 = pt1
        self.pt2 = pt2

    def to_coordinates(self):
        return [self.pt1.x, self.pt1.y, self.pt2.x, self.pt2.y]

    def __str__(self):
        return "Rect[ %s -> %s ]" % (self.pt1, self.pt2)

    def __eq__(self, other):
        return self.pt1 == other.pt1 and self.pt2 == other.pt2


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return "Point[%s,%s]" % (self.x, self.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def to_tuple(self):
        return (self.x, self.y)


class Project:
    def __init__(self, playlist=None, slides=None):
        if playlist is None:
            playlist = []
        if slides is None:
            slides = []
        self.playlist = playlist
        self.slides = slides

    def add_slide(self, slide):
        self.playlist.append((slide.path, None if slide.face_rect is None else slide.face_rect.to_coordinates()))
        self.slides.append(slide)

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
            return Project(playlist=playlist, slides=[create_slide(item[0], item[1]) for item in playlist])


def create_slide(path, face_coordinates):
    rect = create_rect(face_coordinates)
    return Slide(path, rect)


def create_rect(coordinates):
    if coordinates is None or len(coordinates) != 4:
        return None
    point1 = Point(coordinates[0], coordinates[1])
    point2 = Point(coordinates[2], coordinates[3])
    return Rect(point1, point2)
