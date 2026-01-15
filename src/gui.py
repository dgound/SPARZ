#!/usr/bin/env python
"""SPARZ Graphical User Interface (PyQt6)"""

import os
import sys
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QLineEdit, QPushButton, QComboBox,
    QSpinBox, QCheckBox, QProgressBar, QFileDialog, QMessageBox,
    QGroupBox, QFormLayout
)
from PyQt6.QtCore import QThread, pyqtSignal


class CompressWorker(QThread):
    """Worker thread for compression."""
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        try:
            from SPARZ import SPARZIP

            p = self.params
            os.makedirs(p['output_path'], exist_ok=True)
            stem = Path(p['input_path'].replace('*', '').replace('?', '')).stem or 'output'

            z = SPARZIP(
                path_image_files1=p['input_path'],
                stem=stem,
                output_path=p['output_path'],
                path_image_files2=p['bp2_path'],
                find_peaks=not p['no_roi'],
                extract_metadata=p['metadata'],
                create_single_file=p['single_file'],
                num_workers=p['workers'],
            )
            z.run(codec=p['codec'], compression_level=p['level'])
            self.finished.emit("Compression complete!")
        except Exception as e:
            self.error.emit(str(e))


class DecompressWorker(QThread):
    """Worker thread for decompression."""
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        try:
            from SPARZ import UNSPARZ

            p = self.params
            os.makedirs(p['output_path'], exist_ok=True)
            stem = Path(p['video_path'].replace('*', '').replace('?', '')).stem or 'output'

            npz_path = p['npz_path']
            if npz_path is None and not p['no_roi']:
                video_dir = str(Path(p['video_path']).parent)
                npz_path = f"{video_dir}/*.npz"

            u = UNSPARZ(
                path_sparse_bp1=npz_path,
                path_encoded_bp1=p['video_path'],
                stem=stem,
                output_path=p['output_path'],
                use_roi=not p['no_roi'],
                num_workers=p['workers'],
                output_format=p['format'],
            )
            u.run()
            self.finished.emit("Decompression complete!")
        except Exception as e:
            self.error.emit(str(e))


class SparzGUI(QMainWindow):
    """Main window for SPARZ GUI."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SPARZ")
        self.setMinimumSize(550, 500)
        self.worker = None

        self.init_ui()

    def init_ui(self):
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Tab widget
        tabs = QTabWidget()
        layout.addWidget(tabs)

        # Create tabs
        compress_tab = self.create_compress_tab()
        decompress_tab = self.create_decompress_tab()

        tabs.addTab(compress_tab, "  Compress  ")
        tabs.addTab(decompress_tab, "  Decompress  ")

        # Status bar
        status_layout = QHBoxLayout()
        status_layout.addWidget(QLabel("Status:"))
        self.status_label = QLabel("Ready")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        layout.addLayout(status_layout)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # Indeterminate
        self.progress.hide()
        layout.addWidget(self.progress)

    def create_compress_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Input files
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Input files:"))
        self.c_input = QLineEdit()
        input_layout.addWidget(self.c_input)
        c_input_btn = QPushButton("Browse")
        c_input_btn.clicked.connect(self.browse_c_input)
        input_layout.addWidget(c_input_btn)
        layout.addLayout(input_layout)

        # Output directory
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output dir:"))
        self.c_output = QLineEdit("./output/")
        output_layout.addWidget(self.c_output)
        c_output_btn = QPushButton("Browse")
        c_output_btn.clicked.connect(self.browse_c_output)
        output_layout.addWidget(c_output_btn)
        layout.addLayout(output_layout)

        # Biplane (optional)
        bp2_layout = QHBoxLayout()
        bp2_layout.addWidget(QLabel("Biplane (opt):"))
        self.c_bp2 = QLineEdit()
        bp2_layout.addWidget(self.c_bp2)
        c_bp2_btn = QPushButton("Browse")
        c_bp2_btn.clicked.connect(self.browse_c_bp2)
        bp2_layout.addWidget(c_bp2_btn)
        layout.addLayout(bp2_layout)

        # Options group
        options = QGroupBox("Options")
        options_layout = QFormLayout(options)

        # Codec and Level row
        codec_level = QHBoxLayout()
        self.c_codec = QComboBox()
        self.c_codec.addItems(['x265', 'av1', 'x264', 'ffv1', 'prores', 'zstd'])
        codec_level.addWidget(QLabel("Codec:"))
        codec_level.addWidget(self.c_codec)
        codec_level.addSpacing(20)
        self.c_level = QComboBox()
        self.c_level.addItems(['0', '1', '2', '3'])
        codec_level.addWidget(QLabel("Level:"))
        codec_level.addWidget(self.c_level)
        codec_level.addStretch()
        options_layout.addRow(codec_level)

        # Workers
        workers_layout = QHBoxLayout()
        self.c_workers = QSpinBox()
        self.c_workers.setRange(1, 16)
        self.c_workers.setValue(4)
        workers_layout.addWidget(QLabel("Workers:"))
        workers_layout.addWidget(self.c_workers)
        workers_layout.addStretch()
        options_layout.addRow(workers_layout)

        # Checkboxes
        self.c_metadata = QCheckBox("Extract metadata")
        options_layout.addRow(self.c_metadata)

        self.c_single_file = QCheckBox("Single MKV file")
        options_layout.addRow(self.c_single_file)

        self.c_no_roi = QCheckBox("Disable ROI detection")
        options_layout.addRow(self.c_no_roi)

        layout.addWidget(options)

        # Run button
        self.c_run_btn = QPushButton("Compress")
        self.c_run_btn.setMinimumHeight(40)
        self.c_run_btn.clicked.connect(self.run_compress)
        layout.addWidget(self.c_run_btn)

        layout.addStretch()
        return tab

    def create_decompress_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Video files
        video_layout = QHBoxLayout()
        video_layout.addWidget(QLabel("Video files:"))
        self.d_video = QLineEdit()
        video_layout.addWidget(self.d_video)
        d_video_btn = QPushButton("Browse")
        d_video_btn.clicked.connect(self.browse_d_video)
        video_layout.addWidget(d_video_btn)
        layout.addLayout(video_layout)

        # NPZ files
        npz_layout = QHBoxLayout()
        npz_layout.addWidget(QLabel("NPZ files:"))
        self.d_npz = QLineEdit()
        npz_layout.addWidget(self.d_npz)
        d_npz_btn = QPushButton("Browse")
        d_npz_btn.clicked.connect(self.browse_d_npz)
        npz_layout.addWidget(d_npz_btn)
        layout.addLayout(npz_layout)

        hint = QLabel("(auto-detected if empty)")
        hint.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(hint)

        # Output directory
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output dir:"))
        self.d_output = QLineEdit("./reconstructed/")
        output_layout.addWidget(self.d_output)
        d_output_btn = QPushButton("Browse")
        d_output_btn.clicked.connect(self.browse_d_output)
        output_layout.addWidget(d_output_btn)
        layout.addLayout(output_layout)

        # Options group
        options = QGroupBox("Options")
        options_layout = QFormLayout(options)

        # Format and Workers row
        format_workers = QHBoxLayout()
        self.d_format = QComboBox()
        self.d_format.addItems(['tiff', 'dat'])
        format_workers.addWidget(QLabel("Format:"))
        format_workers.addWidget(self.d_format)
        format_workers.addSpacing(20)
        self.d_workers = QSpinBox()
        self.d_workers.setRange(1, 16)
        self.d_workers.setValue(4)
        format_workers.addWidget(QLabel("Workers:"))
        format_workers.addWidget(self.d_workers)
        format_workers.addStretch()
        options_layout.addRow(format_workers)

        # Checkboxes
        self.d_no_roi = QCheckBox("Skip ROI patching (for lossless)")
        options_layout.addRow(self.d_no_roi)

        layout.addWidget(options)

        # Run button
        self.d_run_btn = QPushButton("Decompress")
        self.d_run_btn.setMinimumHeight(40)
        self.d_run_btn.clicked.connect(self.run_decompress)
        layout.addWidget(self.d_run_btn)

        layout.addStretch()
        return tab

    # Browse handlers for compress tab
    def browse_c_input(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select input files", "",
            "TIFF files (*.tiff *.tif);;DAT files (*.dat);;All files (*.*)"
        )
        if files:
            if len(files) > 1:
                dir_path = os.path.dirname(files[0])
                ext = os.path.splitext(files[0])[1]
                self.c_input.setText(f"{dir_path}/*{ext}")
            else:
                self.c_input.setText(files[0])

    def browse_c_output(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select output directory")
        if dir_path:
            self.c_output.setText(dir_path + "/")

    def browse_c_bp2(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select biplane files", "",
            "TIFF files (*.tiff *.tif);;DAT files (*.dat);;All files (*.*)"
        )
        if files:
            if len(files) > 1:
                dir_path = os.path.dirname(files[0])
                ext = os.path.splitext(files[0])[1]
                self.c_bp2.setText(f"{dir_path}/*{ext}")
            else:
                self.c_bp2.setText(files[0])

    # Browse handlers for decompress tab
    def browse_d_video(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select video files", "",
            "Video files (*.mp4 *.avi *.mov *.mkv);;Zstd files (*.zst);;All files (*.*)"
        )
        if files:
            if len(files) > 1:
                dir_path = os.path.dirname(files[0])
                ext = os.path.splitext(files[0])[1]
                self.d_video.setText(f"{dir_path}/*{ext}")
            else:
                self.d_video.setText(files[0])

    def browse_d_npz(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select NPZ files", "",
            "NPZ files (*.npz);;All files (*.*)"
        )
        if files:
            if len(files) > 1:
                dir_path = os.path.dirname(files[0])
                self.d_npz.setText(f"{dir_path}/*.npz")
            else:
                self.d_npz.setText(files[0])

    def browse_d_output(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select output directory")
        if dir_path:
            self.d_output.setText(dir_path + "/")

    # Run handlers
    def run_compress(self):
        if self.worker is not None and self.worker.isRunning():
            return

        if not self.c_input.text():
            QMessageBox.critical(self, "Error", "Please select input files")
            return

        params = {
            'input_path': self.c_input.text(),
            'output_path': self.c_output.text(),
            'bp2_path': self.c_bp2.text() or None,
            'codec': self.c_codec.currentText(),
            'level': int(self.c_level.currentText()),
            'workers': self.c_workers.value(),
            'metadata': self.c_metadata.isChecked(),
            'single_file': self.c_single_file.isChecked(),
            'no_roi': self.c_no_roi.isChecked(),
        }

        self.set_running(True)
        self.worker = CompressWorker(params)
        self.worker.finished.connect(self.on_complete)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def run_decompress(self):
        if self.worker is not None and self.worker.isRunning():
            return

        if not self.d_video.text():
            QMessageBox.critical(self, "Error", "Please select video files")
            return

        params = {
            'video_path': self.d_video.text(),
            'output_path': self.d_output.text(),
            'npz_path': self.d_npz.text() or None,
            'format': self.d_format.currentText(),
            'workers': self.d_workers.value(),
            'no_roi': self.d_no_roi.isChecked(),
        }

        self.set_running(True)
        self.worker = DecompressWorker(params)
        self.worker.finished.connect(self.on_complete)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def set_running(self, running):
        self.c_run_btn.setEnabled(not running)
        self.d_run_btn.setEnabled(not running)
        if running:
            self.progress.show()
            self.status_label.setText("Processing...")
        else:
            self.progress.hide()

    def on_complete(self, message):
        self.set_running(False)
        self.status_label.setText(message)
        QMessageBox.information(self, "Success", message)

    def on_error(self, error):
        self.set_running(False)
        self.status_label.setText("Error")
        QMessageBox.critical(self, "Error", error)


def main():
    """Entry point for sparz-gui command."""
    app = QApplication(sys.argv)
    window = SparzGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
