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
    def __init__(self, playlist=None, slides=None, settings=None):
        if playlist is None:
            playlist = []
        if slides is None:
            slides = []
        if settings is None:
            settings = Settings()

        self.playlist = playlist
        self.slides = slides
        self.settings = settings

    def add_slide(self, slide):
        self.playlist.append((slide.path, None if slide.face_rect is None else slide.face_rect.to_coordinates()))
        self.slides.append(slide)

    def save(self, path):
        data = dict()
        data["playlist"] = self.playlist
        data["settings"] = dict(
            fade_in_millis=self.settings.fade_in_millis,
            fade_out_millis=self.settings.fade_out_millis,
            target_size=self.settings.target_size,
            fps=self.settings.fps,
            output_path=self.settings.output_path,
            show_millis=self.settings.show_millis,
        )
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
            settings = load_settings(data.get("settings", None))
            return Project(playlist=playlist, slides=[create_slide(item[0], item[1]) for item in playlist],
                           settings=settings)


def load_settings(data=None):
    if data is None:
        return Settings()

    fade_in_millis = data.get("fade_in_millis", 500)
    show_millis = data.get("show_millis", 2000)
    fade_out_millis = data.get("fade_out_millis", 1000)
    target_size = tuple(data.get("target_size", [768, 1024]))
    fps = data.get("fps", 30)
    output_path = data.get("output_path", "out.mov")

    return Settings(
        fade_in_millis=fade_in_millis,
        show_millis=show_millis,
        fade_out_millis=fade_out_millis,
        target_size=target_size,
        fps=fps,
        output_path=output_path
    )


def create_slide(path, face_coordinates):
    rect = create_rect(face_coordinates)
    return Slide(path, rect)


def create_rect(coordinates):
    if coordinates is None or len(coordinates) != 4:
        return None
    point1 = Point(coordinates[0], coordinates[1])
    point2 = Point(coordinates[2], coordinates[3])
    return Rect(point1, point2)


class Settings:
    def __init__(self, fade_in_millis=500, show_millis=2000, fade_out_millis=1000, target_size=(768, 1024),
                 fps=25, output_path="out.mov"):
        self.fade_in_millis = fade_in_millis
        self.show_millis = show_millis
        self.fade_out_millis = fade_out_millis
        self.target_size = target_size
        self.fps = fps
        self.output_path = output_path
