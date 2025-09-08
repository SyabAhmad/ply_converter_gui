import os
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QProgressBar,
                             QMessageBox, QFrame)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from conversion_worker import ConversionWorker

class PLYConverterGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.input_file = None
        self.output_dir = None
        self.worker = None
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("PLY Converter")
        self.setGeometry(600, 400, 700, 500)
        self.setMinimumSize(600, 400)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 30, 30, 30)

        title = QLabel("PLY Converter")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color:#2c3e50;")
        main_layout.addWidget(title)

        desc = QLabel("Convert 3D Gaussian Splatting PLY files to STL / GLB / 3MF / DXF")
        desc.setAlignment(Qt.AlignCenter)
        desc.setStyleSheet("color:#7f8c8d;")
        main_layout.addWidget(desc)

        file_frame = QFrame()
        file_layout = QVBoxLayout(file_frame)

        # Input
        in_layout = QHBoxLayout()
        in_label = QLabel("Input PLY:")
        in_label.setFont(QFont("Arial",10,QFont.Bold))
        self.input_path_label = QLabel("No file selected")
        self.input_path_label.setStyleSheet("color:#7f8c8d; padding:6px; background:#ecf0f1; border-radius:4px;")
        self.input_path_label.setFixedWidth(260)
        browse_in = QPushButton("Browse...")
        browse_in.clicked.connect(self.browse_input_file)
        in_layout.addWidget(in_label)
        in_layout.addWidget(self.input_path_label,1)
        in_layout.addWidget(browse_in)
        file_layout.addLayout(in_layout)

        # Output
        out_layout = QHBoxLayout()
        out_label = QLabel("Output Dir:")
        out_label.setFont(QFont("Arial",10,QFont.Bold))
        self.output_path_label = QLabel("No directory selected")
        self.output_path_label.setStyleSheet("color:#7f8c8d; padding:6px; background:#ecf0f1; border-radius:4px;")
        self.output_path_label.setFixedWidth(260)
        browse_out = QPushButton("Browse...")
        browse_out.clicked.connect(self.browse_output_dir)
        out_layout.addWidget(out_label)
        out_layout.addWidget(self.output_path_label,1)
        out_layout.addWidget(browse_out)
        file_layout.addLayout(out_layout)

        main_layout.addWidget(file_frame)

        self.convert_button = QPushButton("Convert")
        self.convert_button.setEnabled(False)
        self.convert_button.clicked.connect(self.start_conversion)
        self.convert_button.setMinimumHeight(48)
        self.convert_button.setStyleSheet("""
            QPushButton { background:#27ae60; color:#fff; border:none; border-radius:6px; font-weight:bold;}
            QPushButton:disabled { background:#bdc3c7; color:#7f8c8d;}
        """)
        main_layout.addWidget(self.convert_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        self.statusBar().showMessage("Ready")

    def browse_input_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select PLY", "", "PLY Files (*.ply)")
        if path:
            self.input_file = path
            self.input_path_label.setText(os.path.basename(path))
            self.input_path_label.setStyleSheet("color:#27ae60; padding:6px; background:#d5f4e6; border-radius:4px;")
            self.check_ready()

    def browse_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.output_dir = path
            self.output_path_label.setText(path)
            self.output_path_label.setStyleSheet("color:#27ae60; padding:6px; background:#d5f4e6; border-radius:4px;")
            self.check_ready()

    def check_ready(self):
        self.convert_button.setEnabled(bool(self.input_file and self.output_dir))

    def start_conversion(self):
        if not (self.input_file and self.output_dir):
            return
        self.convert_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0,100)
        self.progress_bar.setValue(0)
        self.worker = ConversionWorker(
            self.input_file,
            self.output_dir,
            keep_all_clusters=True,
            disable_plane_removal=True,   # ensure planes NOT removed
            disable_z_trim=True,
            high_detail=True,
            preserve_thin_features=True
        )
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.conversion_finished)
        self.worker.start()
        self.statusBar().showMessage("Converting...")

    def update_progress(self, msg):
        self.statusBar().showMessage(msg)
        m = msg.lower()
        if "loading" in m: self.progress_bar.setValue(10)
        elif "pre" in m: self.progress_bar.setValue(25)
        elif "processing and enhancing colors" in m: self.progress_bar.setValue(35)
        elif "stl" in m: self.progress_bar.setValue(50)
        elif "glb" in m: self.progress_bar.setValue(65)
        elif "3mf" in m: self.progress_bar.setValue(80)
        elif "dxf" in m: self.progress_bar.setValue(90)
        elif "complete" in m: self.progress_bar.setValue(100)

    def conversion_finished(self, ok, msg):
        self.convert_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        if ok:
            self.statusBar().showMessage("Done")
            QMessageBox.information(self,"Success",f"Done.\nFiles in:\n{self.output_dir}")
        else:
            self.statusBar().showMessage("Failed")
            QMessageBox.critical(self,"Error",msg)