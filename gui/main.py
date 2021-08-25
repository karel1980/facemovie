import glob
import os
import sys
import threading

import cv2
import face_recognition
import numpy as np
import yaml
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QImage, QPixmap, QWindow
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QFileDialog, QToolBar, QListView, \
    QHBoxLayout, QWidget, QVBoxLayout, QAbstractItemView, QAction

import project
from generate_facemovie import generate_facemovie, ProgressWriter
from gui.components.output_settings import OutputSettingsWindow
from project import Settings


def face_to_rect(face):
    return project.Rect(project.Point(face[1], face[0]), project.Point(face[3], face[2]))


class QueueViewer(QWindow):
    def __init__(self):
        super().__init__()

        self.set

class MainWindow(QMainWindow):
    """Main Window."""

    def __init__(self, parent=None):
        """Initializer."""
        super().__init__(parent)
        self.selected_image = QLabel("selected image")
        self.image_list_model = None
        self.image_list = None

        self.generator_thread = None
        self.work_queue = []

        self._create_actions()
        self.settings = Settings()
        self.create_imagelist()

        self.current_project_path = None
        self.current_image_faces = []

        self.setWindowTitle("Facemovie builder")
        self.resize(800, 480)

        self.create_main_widget()

        toolbar = self.create_toolbar()
        self.addToolBar(Qt.TopToolBarArea, toolbar)

        rc = self.load_rcfile()
        if rc.get('last_opened', None) is not None:
            self.load_project(rc['last_opened'])
        else:
            self.start_new_project()

    def _create_actions(self):
        self.action_new = QAction("New", self)
        self.action_new.triggered.connect(self.handle_action_new)

        self.action_open = QAction("Open", self)
        self.action_open.triggered.connect(self.handle_action_open)

        self.action_save = QAction("Save", self)
        self.action_save.triggered.connect(self.handle_action_save_project)

        self.action_add_image_directory = QAction("Add images from dir", self)
        self.action_add_image_directory.triggered.connect(self.handle_action_add_image_directory)

        self.action_generate_facemovie = QAction("Generate facemovie", self)
        self.action_generate_facemovie.triggered.connect(self.handle_action_generate_facemovie)

        self.action_clear_face = QAction("clear", self)
        self.action_clear_face.triggered.connect(self.handle_action_clear)

        self.action_next_image = QAction("next_image", self)
        self.action_next_image.triggered.connect(self.handle_action_next_image)

        self.action_previous_image = QAction("previous_image", self)
        self.action_previous_image.triggered.connect(self.handle_action_previous_image)

        self.action_previous_face = QAction("previous_face", self)
        self.action_previous_face.triggered.connect(self.handle_action_previous_face)

        self.action_next_face = QAction("next_face", self)
        self.action_next_face.triggered.connect(self.handle_action_next_face)

    def create_imagelist(self):
        image_list = QListView()
        image_list.setEditTriggers(QAbstractItemView.NoEditTriggers)

        self.image_list_model = QStandardItemModel(image_list)
        image_list.setModel(self.image_list_model)

        def on_image_selected():
            if self.image_list_model.rowCount() == 0:
                return

            item = self.get_selected_item()
            if item is not None:
                self.set_selected_slide(item.data(5))

        image_list.setMaximumWidth(200)
        image_list.selectionModel().selectionChanged.connect(on_image_selected)
        self.image_list = image_list

    def set_selected_slide(self, slide):
        if slide is None:
            img = create_placeholder_image((240, 320))
            self.selected_image.setPixmap(image_to_pixmap(img))
            return

        img = cv2.imread(slide.path)
        if img is None:
            img = create_placeholder_image((240, 320))

        self.current_image_faces = [face_to_rect(face) for face in face_recognition.face_locations(img)]

        for face in self.current_image_faces:
            cv2.rectangle(img, face.pt1.to_tuple(), face.pt2.to_tuple(), (200, 200, 200))

        if slide.face_rect is not None:
            pt1, pt2 = slide.face_rect.pt1, slide.face_rect.pt2
            cv2.rectangle(img, (pt1.x, pt1.y), (pt2.x, pt2.y), (0, 255, 0))

        pixmap = image_to_pixmap(img)

        item = self.get_selected_item()
        if item is not None:
            item.setData(slide, 5)

        self.selected_image.setPixmap(pixmap)

    def clear_selected_face(self):
        item = self.get_selected_item()
        if item is None:
            return
        slide = item.data(5)
        self.set_selected_slide(project.InputSlide(slide.path, None))

    def get_selected_item(self):
        indexes = self.image_list.selectedIndexes()
        if len(indexes) == 0:
            return None
        return self.image_list_model.itemFromIndex(indexes[0])

    def handle_action_clear(self):
        self.clear_selected_face()

    def handle_action_next_image(self):
        selected = self.image_list.selectedIndexes()
        if len(selected) == 0:
            self.image_list.setCurrentIndex(self.image_list_model.index(0, 0))
        else:
            row, col = (selected[0].row() + 1) % self.image_list_model.rowCount(), selected[0].column()
            self.image_list.setCurrentIndex(self.image_list_model.index(row, col))

    def handle_action_previous_image(self):
        selected = self.image_list.selectedIndexes()
        list_size = self.image_list_model.rowCount()
        if len(selected) == 0:
            self.image_list.setCurrentIndex(self.image_list_model.index(list_size - 1, 0))
        else:
            row, col = (selected[0].row() + list_size - 1) % self.image_list_model.rowCount(), selected[0].column()
            self.image_list.setCurrentIndex(self.image_list_model.index(row, col))

    def handle_action_previous_face(self):
        self.select_face(-1)

    def handle_action_next_face(self):
        self.select_face(1)

    def select_face(self, offset):
        item = self.get_selected_item()
        if item is None:
            return  # no image selected
        slide = item.data(5)
        if slide is None:
            return  # no image selected

        if len(self.current_image_faces) == 0:
            return  # no faces in current image

        if slide.face_rect is None or slide.face_rect not in self.current_image_faces:
            self.set_selected_slide(project.InputSlide(slide.path, self.current_image_faces[0]))
            return

        current_face_index = self.current_image_faces.index(slide.face_rect)
        if current_face_index < 0:
            self.set_selected_slide(project.InputSlide(slide.path, self.current_image_faces[0]))
        else:
            face = self.current_image_faces[
                (current_face_index + offset + len(self.current_image_faces)) % len(self.current_image_faces)]
            self.set_selected_slide(project.InputSlide(slide.path, face))

    def create_main_widget(self):
        right_panel = QWidget()
        right_panel_layout = QVBoxLayout()
        right_panel_layout.setAlignment(Qt.AlignTop)
        slide_tool_bar = QToolBar()

        def set_face_from_mouseclick(a):
            item = self.get_selected_item()
            if item is None:
                return  # no image selected
            slide = item.data(5)
            if slide is None:
                return  # data for selected image
            face_rect = project.Rect(project.Point(a.x() - 5, a.y() - 5), project.Point(a.x() + 5, a.y() + 5))
            self.set_selected_slide(project.InputSlide(slide.path, face_rect))

        slide_tool_bar.addAction(self.action_previous_image)
        slide_tool_bar.addAction(self.action_next_image)
        slide_tool_bar.addAction(self.action_clear_face)
        slide_tool_bar.addAction(self.action_previous_face)
        slide_tool_bar.addAction(self.action_next_face)

        right_panel_layout.addWidget(slide_tool_bar)
        self.selected_image.mousePressEvent = set_face_from_mouseclick

        right_panel_layout.addWidget(self.selected_image)
        right_panel.setLayout(right_panel_layout)

        main_widget = QWidget()
        main_layout = QHBoxLayout()
        main_layout.setAlignment(Qt.AlignVCenter)
        main_layout.addWidget(self.image_list)
        main_layout.addWidget(right_panel)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def add_image(self, path):
        img = cv2.imread(path)
        if img is None:
            return
        faces = face_recognition.face_locations(img)
        if len(faces) == 0:
            self.add_slide(project.InputSlide(path, None))
        else:
            face = faces[0]
            self.add_slide(
                project.InputSlide(path,
                                   project.Rect(project.Point(face[1], face[0]), project.Point(face[3], face[2]))))

    @staticmethod
    def calculate_relative_image_path(image_path, project_path):
        if project_path is None:
            # TODO: given no project path, fall back to cwd?
            return image_path

        project_dir = os.path.dirname(project_path)
        if image_path.startswith(project_dir + '/'):  # TODO: deal with path separator differences across OS
            return image_path[len(project_dir) + 1:]
        return image_path

    def add_slide(self, slide):
        shortened_path = self.calculate_relative_image_path(slide.path, self.current_project_path)
        item = QStandardItem(shortened_path)

        item.setData(slide, 5)
        self.image_list_model.appendRow(item)

    def handle_action_open(self):
        file = QFileDialog.getOpenFileName(self, "Select project", filter="*.json")
        if file == ('', ''):
            return
        self.load_project(file[0])

    def handle_action_new(self):
        self.start_new_project()

    def handle_action_add_image_directory(self):
        file = str(QFileDialog.getExistingDirectory(self, "Select directory"))
        if len(file) == 0:
            return
        self.add_images_from_dir(file)

    def handle_action_save_project(self):
        if self.current_project_path is None:
            file = QFileDialog.getSaveFileName(self, "Save project")[0]
            if file == "":
                return
            self.save_project(file)
        else:
            self.save_project(self.current_project_path)

    def process_next_work_queue_element(self):
        if self.generator_thread is not None:
            print("Generator thread still working")
            return
        if len(self.work_queue) == 0:
            print("Queue empty")
            return

        next_project = self.work_queue[0]
        self.work_queue = self.work_queue[1:]

        def do_generate(proj):
            try:
                generate_facemovie(proj, progress_callback=ProgressWriter().on_progress)
            except Exception as ex:
                print("Facemovie generation failed", ex)
            self.generator_thread = None
            self.process_next_work_queue_element()

        self.generator_thread = threading.Thread(target=do_generate, args=[next_project])
        self.generator_thread.start()

    def handle_action_generate_facemovie(self):
        def on_generate_dialog_confirmed(settings):
            self.work_queue.append(self.create_project_from_state())
            self.settings = settings
            self.process_next_work_queue_element()

        dialog = OutputSettingsWindow(settings=self.settings, generate_callback=on_generate_dialog_confirmed)
        dialog.setAttribute(Qt.WA_DeleteOnClose)
        dialog.exec_()

    def create_toolbar(self):
        toolbar = QToolBar(self)
        toolbar.addAction(self.action_new)
        toolbar.addAction(self.action_open)
        toolbar.addAction(self.action_save)
        toolbar.addAction(self.action_add_image_directory)
        toolbar.addAction(self.action_generate_facemovie)
        return toolbar

    def save_project(self, file):
        proj = self.create_project_from_state()
        proj.save(file)
        self.save_last_opened(file)

    def create_project_from_state(self):
        proj = project.Project()
        for i in range(self.image_list_model.rowCount()):
            item = self.image_list_model.item(i)
            slide = item.data(5)
            if slide is None:
                continue
            proj.add_slide(project.InputSlide(slide.path, slide.face_rect))
        proj.settings = self.settings
        return proj

    def save_last_opened(self, path):
        rc = self.load_rcfile()
        rc['last_opened'] = path
        self.write_rcfile(rc)

    @staticmethod
    def load_rcfile():
        expanded = os.path.expanduser("~/.facemovierc")
        rc = None
        if os.path.exists(expanded):
            try:
                rc = yaml.load(open(expanded, "r"), Loader=yaml.SafeLoader)
            except Exception:
                print("could not read ~/.facemovierc")
        return dict() if rc is None else rc

    @staticmethod
    def write_rcfile(rc):
        expanded = os.path.expanduser("~/.facemovierc")
        try:
            with open(expanded, "w") as rcfile:
                yaml.dump(rc, rcfile)
        except Exception as e:
            print("could not write ~/.facemovierc")
            print(e)

    def load_project(self, project_file):
        self.start_new_project()
        proj = project.Project.load(project_file)

        self.settings = proj.settings

        self.current_project_path = project_file
        self.save_last_opened(project_file)
        for slide in proj.slides:
            self.add_slide(slide)

    def add_images_from_dir(self, dirname):
        for path in glob.glob(dirname + "/**", recursive=True):
            if not os.path.isfile(path):
                continue
            self.add_image(path)

    def start_new_project(self):
        self.save_last_opened(None)
        self.settings = Settings()
        self.image_list_model.clear()
        self.current_image_faces = []
        self.set_selected_slide(None)


def image_to_pixmap(img):
    return QPixmap(QImage(img, img.shape[1], img.shape[0], img.shape[1] * 3, QImage.Format_RGB888).rgbSwapped())


def create_placeholder_image(shape):
    h, w = shape
    img = np.zeros([h, w, 3], dtype=np.uint8)
    img[:] = 128
    cv2.rectangle(img, (10, 10), (310, 230), (255, 255, 255), 5)
    return img


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
