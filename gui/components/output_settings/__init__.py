from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import QDialog, QLineEdit, QPushButton, QFileDialog, QGridLayout, QLabel, QComboBox

from gui.constants import RESOLUTIONS
from project import Settings


class OutputSettingsWindow(QDialog):
    def __init__(self, parent=None, settings=None, generate_callback=None):
        super().__init__(parent)
        print("created output settings dialog with settings", settings.fps)
        self.generate_callback = generate_callback

        def create_int_edit(min_value, max_value, initial_value):
            edit = QLineEdit()
            validator = QIntValidator()
            validator.setRange(min_value, max_value)
            edit.setValidator(validator)
            edit.setText(str(initial_value))
            return edit

        def create_file_input():
            edit = QLineEdit()
            return edit

        edit_fade_in_millis = create_int_edit(0, 10000, settings.fade_in_millis)
        edit_show_millis = create_int_edit(0, 10000, settings.show_millis)
        combo_resolution = self.create_resolutions_combobox(RESOLUTIONS)
        edit_fps = create_int_edit(12, 120, settings.fps)
        edit_filename = create_file_input()
        edit_filename.setText(settings.output_path)
        btn_select_output_file = QPushButton("Select file")

        def generate_clicked():
            if self.generate_callback is not None:
                resolution = RESOLUTIONS[combo_resolution.currentIndex()]

                settings = Settings(int(edit_fade_in_millis.text()), int(edit_show_millis.text()), 1000,
                                    (resolution[2], resolution[1]),
                                    int(edit_fps.text()),
                                    edit_filename.text())
                self.generate_callback(settings)
            self.close()

        def select_file_clicked():
            file = QFileDialog.getSaveFileName(self, "Select output filename", initialFilter="*.mov")[0]
            if file == "":
                return
            edit_filename.setText(file)

        btn_select_output_file.clicked.connect(select_file_clicked)

        btn_generate = QPushButton("Generate facemovie")
        btn_generate.clicked.connect(generate_clicked)

        layout = QGridLayout()
        layout.addWidget(QLabel("fade_in_millis"), 0, 0)
        layout.addWidget(edit_fade_in_millis, 0, 1)
        layout.addWidget(QLabel("show_millis"), 1, 0)
        layout.addWidget(edit_show_millis, 1, 1)
        # layout.addWidget(QLabel("fade_out_millis"), 2, 0)
        # layout.addWidget(create_int_edit(0, 10000), 2, 1)
        layout.addWidget(QLabel("target_size"), 2, 0)
        layout.addWidget(combo_resolution, 2, 1)
        layout.addWidget(QLabel("fps"), 3, 0)
        layout.addWidget(edit_fps, 3, 1)
        layout.addWidget(QLabel("Output filename"), 4, 0)
        layout.addWidget(edit_filename, 4, 1)
        layout.addWidget(btn_select_output_file, 4, 2)
        layout.addWidget(btn_generate, 5, 0)

        self.setLayout(layout)

    def create_resolutions_combobox(self, resolutions):
        combo_resolution = QComboBox()
        for i, r in enumerate(resolutions):
            combo_resolution.addItem("%s (%s x %s)" % r)
        return combo_resolution