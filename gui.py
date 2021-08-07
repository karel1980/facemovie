import glob
import os
import sys

import cv2
import face_recognition
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QFileDialog, QToolBar, QPushButton, QListView, \
    QHBoxLayout, QWidget, QVBoxLayout, QAbstractItemView

import project


class Window(QMainWindow):
    """Main Window."""

    def __init__(self, parent=None):
        """Initializer."""
        super().__init__(parent)
        self.create_imagelist()

        self.current_project_path = None
        self.current_image_faces = []

        self.setWindowTitle("Facemovie builder")
        self.resize(640, 480)

        self.create_main_widget()

        toolbar = self.create_toolbar()
        self.addToolBar(Qt.TopToolBarArea, toolbar)

        # self.start_new_project()
        self.load_project("project.json")

    def create_imagelist(self):
        image_list = QListView()
        image_list.setEditTriggers(QAbstractItemView.NoEditTriggers)

        self.image_list_model = QStandardItemModel(image_list)
        image_list.setModel(self.image_list_model)

        def on_image_selected(sel):
            if self.image_list_model.rowCount() == 0:
                return

            item = self.get_selected_item()
            if item is not None:
                self.set_selected_slide(item.data(1))

        image_list.setMaximumWidth(200)
        image_list.selectionModel().selectionChanged.connect(on_image_selected)
        self.image_list = image_list

    def set_selected_slide(self, slide):
        print("setting selected slide", slide)
        path, loc = slide
        img = cv2.imread(path)
        self.current_image_faces = face_recognition.face_locations(img)

        for face in self.current_image_faces:
            cv2.rectangle(img, (face[1], face[0]), (face[3], face[2]), (200, 200, 200))

        if loc is not None:
            cv2.rectangle(img, (loc[1], loc[0]), (loc[3], loc[2]), (0, 255, 0))

        pixmap = QPixmap(QImage(img, img.shape[1], img.shape[0], img.shape[1] * 3, QImage.Format_RGB888).rgbSwapped())

        item = self.get_selected_item()
        if item is not None:
            item.setData(slide, 1)

        self.selected_image.setPixmap(pixmap)

    def clear_selected_face(self):
        item = self.get_selected_item()
        if item is None:
            return
        slide = item.data(1)
        self.set_selected_slide((slide[0], None))

    def get_selected_item(self):
        indexes = self.image_list.selectedIndexes()
        if len(indexes) == 0:
            return None
        return self.image_list_model.itemFromIndex(indexes[0])

    def create_main_widget(self):
        right_panel = QWidget()
        right_panel_layout = QVBoxLayout()
        right_panel_layout.setAlignment(Qt.AlignTop)
        slide_tool_bar = QToolBar()

        def on_clear():
            self.clear_selected_face()

        def go_to_next():
            selected = self.image_list.selectedIndexes()
            if len(selected) == 0:
                self.image_list.setCurrentIndex(self.image_list_model.index(0, 0))
            else:
                row, col = (selected[0].row() + 1) % self.image_list_model.rowCount(), selected[0].column()
                self.image_list.setCurrentIndex(self.image_list_model.index(row, col))

        def go_to_prev():
            selected = self.image_list.selectedIndexes()
            list_size = self.image_list_model.rowCount()
            if len(selected) == 0:
                self.image_list.setCurrentIndex(self.image_list_model.index(list_size - 1, 0))
            else:
                row, col = (selected[0].row() + list_size - 1) % self.image_list_model.rowCount(), selected[0].column()
                self.image_list.setCurrentIndex(self.image_list_model.index(row, col))

        btn_prev = QPushButton("prev")
        btn_prev.clicked.connect(go_to_prev)
        btn_next = QPushButton("next")
        btn_next.clicked.connect(go_to_next)
        btn_clear = QPushButton("clear")
        btn_clear.clicked.connect(on_clear)

        slide_tool_bar.addWidget(btn_prev)
        slide_tool_bar.addWidget(btn_next)
        slide_tool_bar.addWidget(btn_clear)

        right_panel_layout.addWidget(slide_tool_bar)
        self.selected_image = QLabel("selected image")
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
        face_locations = face_recognition.face_locations(cv2.imread(path))
        self.add_slide(path, None if len(face_locations) == 0 else face_locations[0])

    def add_slide(self, path, location):
        slide = path, location
        item = QStandardItem(path)
        item.setData(slide, 1)
        item.setData(path)
        self.image_list_model.appendRow(item)

    def create_toolbar(self):
        def start_new_project():
            self.start_new_project()

        def load_project():
            file = QFileDialog.getOpenFileName(self, "Select project", filter="*.json")
            if file == ('', ''):
                return
            self.load_project(file[0])

        def select_image_directory():
            file = str(QFileDialog.getExistingDirectory(self, "Select directory"))
            if len(file) == 0:
                return
            self.start_new_project()
            self.add_images_from_dir(file)

        def save_project():
            if self.current_project_path is None:
                file = QFileDialog.getSaveFileName(self, "Save project")[0]
                if file == "":
                    return
                proj = project.Project()
                for i in range(self.image_list_model.rowCount()):
                    item = self.image_list_model.item(i)
                    slide = item.data()
                    proj.add_slide(slide[0], slide[1])
                proj.save(file)
            else:
                self.project.save(self.current_project_path)

        toolbar = QToolBar()
        btn_new_project = QPushButton("New project")
        btn_new_project.clicked.connect(start_new_project)

        btn_load_project = QPushButton("Load project")
        btn_load_project.clicked.connect(load_project)

        btn_save_project = QPushButton("Save project")
        btn_save_project.clicked.connect(save_project)

        btn_add_images_from_dir = QPushButton("Add images from dir")
        btn_add_images_from_dir.clicked.connect(select_image_directory)

        toolbar.addWidget(btn_new_project)
        toolbar.addWidget(btn_load_project)
        toolbar.addWidget(btn_save_project)
        toolbar.addWidget(btn_add_images_from_dir)
        return toolbar

    def load_project(self, project_file):
        self.start_new_project()
        proj = project.Project.load(project_file)
        for slide in proj.playlist:
            print("opened slide:", slide)
            self.add_slide(slide[0], slide[1])

    def add_images_from_dir(self, dirname):
        for path in glob.glob(dirname + "/**", recursive=True):
            if not os.path.isfile(path):
                continue
            self.add_image(path)

    def start_new_project(self):
        self.image_list_model.clear()


def main():
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
