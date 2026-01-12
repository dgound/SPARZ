#!/usr/bin/env python
"""SPARZ Graphical User Interface (tkinter)"""

import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path


class SparzGUI:
    """Unified GUI for SPARZ compression and decompression."""

    def __init__(self, root):
        self.root = root
        self.root.title("SPARZ")
        self.root.geometry("520x480")
        self.root.resizable(True, True)

        self.running = False
        self.create_widgets()

    def create_widgets(self):
        # Main frame
        main = ttk.Frame(self.root, padding="10")
        main.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Notebook (tabs)
        self.notebook = ttk.Notebook(main)
        self.notebook.grid(row=0, column=0, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)

        # Create tabs
        self.compress_frame = ttk.Frame(self.notebook, padding="10")
        self.decompress_frame = ttk.Frame(self.notebook, padding="10")

        self.notebook.add(self.compress_frame, text="  Compress  ")
        self.notebook.add(self.decompress_frame, text="  Decompress  ")

        self.create_compress_tab()
        self.create_decompress_tab()

        # Status bar at bottom
        status_frame = ttk.Frame(main)
        status_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        status_frame.columnconfigure(1, weight=1)

        ttk.Label(status_frame, text="Status:").grid(row=0, column=0, sticky="w")
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(status_frame, textvariable=self.status_var).grid(row=0, column=1, sticky="w", padx=(5, 0))

        self.progress = ttk.Progressbar(main, mode='indeterminate')
        self.progress.grid(row=2, column=0, sticky="ew", pady=(5, 0))

    def create_compress_tab(self):
        frame = self.compress_frame
        frame.columnconfigure(1, weight=1)

        row = 0

        # Input files
        ttk.Label(frame, text="Input files:").grid(row=row, column=0, sticky="w", pady=2)
        self.c_input_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.c_input_var).grid(row=row, column=1, sticky="ew", pady=2, padx=(5, 0))
        ttk.Button(frame, text="Browse", command=self.browse_c_input).grid(row=row, column=2, pady=2, padx=(5, 0))
        row += 1

        # Output directory
        ttk.Label(frame, text="Output dir:").grid(row=row, column=0, sticky="w", pady=2)
        self.c_output_var = tk.StringVar(value="./output/")
        ttk.Entry(frame, textvariable=self.c_output_var).grid(row=row, column=1, sticky="ew", pady=2, padx=(5, 0))
        ttk.Button(frame, text="Browse", command=self.browse_c_output).grid(row=row, column=2, pady=2, padx=(5, 0))
        row += 1

        # Biplane (optional)
        ttk.Label(frame, text="Biplane (opt):").grid(row=row, column=0, sticky="w", pady=2)
        self.c_bp2_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.c_bp2_var).grid(row=row, column=1, sticky="ew", pady=2, padx=(5, 0))
        ttk.Button(frame, text="Browse", command=self.browse_c_bp2).grid(row=row, column=2, pady=2, padx=(5, 0))
        row += 1

        # Separator
        ttk.Separator(frame, orient="horizontal").grid(row=row, column=0, columnspan=3, sticky="ew", pady=10)
        row += 1

        # Options frame
        opts = ttk.LabelFrame(frame, text="Options", padding="5")
        opts.grid(row=row, column=0, columnspan=3, sticky="ew", pady=5)
        opts.columnconfigure(1, weight=1)
        row += 1

        # Codec
        ttk.Label(opts, text="Codec:").grid(row=0, column=0, sticky="w", pady=2)
        self.c_codec_var = tk.StringVar(value="x265")
        codec_combo = ttk.Combobox(opts, textvariable=self.c_codec_var, state="readonly", width=12)
        codec_combo['values'] = ('x265', 'av1', 'x264', 'ffv1', 'prores', 'zstd')
        codec_combo.grid(row=0, column=1, sticky="w", pady=2, padx=(5, 0))

        # Level
        ttk.Label(opts, text="Level:").grid(row=0, column=2, sticky="w", pady=2, padx=(15, 0))
        self.c_level_var = tk.StringVar(value="0")
        level_combo = ttk.Combobox(opts, textvariable=self.c_level_var, state="readonly", width=5)
        level_combo['values'] = ('0', '1', '2', '3')
        level_combo.grid(row=0, column=3, sticky="w", pady=2, padx=(5, 0))

        # Workers
        ttk.Label(opts, text="Workers:").grid(row=1, column=0, sticky="w", pady=2)
        self.c_workers_var = tk.StringVar(value="4")
        ttk.Spinbox(opts, textvariable=self.c_workers_var, from_=1, to=16, width=5).grid(row=1, column=1, sticky="w", pady=2, padx=(5, 0))

        # Checkboxes
        self.c_metadata_var = tk.BooleanVar()
        ttk.Checkbutton(opts, text="Extract metadata", variable=self.c_metadata_var).grid(row=2, column=0, columnspan=2, sticky="w", pady=2)

        self.c_single_file_var = tk.BooleanVar()
        ttk.Checkbutton(opts, text="Single MKV file", variable=self.c_single_file_var).grid(row=2, column=2, columnspan=2, sticky="w", pady=2)

        self.c_no_roi_var = tk.BooleanVar()
        ttk.Checkbutton(opts, text="Disable ROI detection", variable=self.c_no_roi_var).grid(row=3, column=0, columnspan=2, sticky="w", pady=2)

        # Run button
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=row, column=0, columnspan=3, pady=15)
        self.c_run_btn = ttk.Button(btn_frame, text="Compress", command=self.run_compress, width=20)
        self.c_run_btn.pack()

    def create_decompress_tab(self):
        frame = self.decompress_frame
        frame.columnconfigure(1, weight=1)

        row = 0

        # Video files
        ttk.Label(frame, text="Video files:").grid(row=row, column=0, sticky="w", pady=2)
        self.d_video_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.d_video_var).grid(row=row, column=1, sticky="ew", pady=2, padx=(5, 0))
        ttk.Button(frame, text="Browse", command=self.browse_d_video).grid(row=row, column=2, pady=2, padx=(5, 0))
        row += 1

        # NPZ files (optional)
        ttk.Label(frame, text="NPZ files:").grid(row=row, column=0, sticky="w", pady=2)
        self.d_npz_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.d_npz_var).grid(row=row, column=1, sticky="ew", pady=2, padx=(5, 0))
        ttk.Button(frame, text="Browse", command=self.browse_d_npz).grid(row=row, column=2, pady=2, padx=(5, 0))
        row += 1
        ttk.Label(frame, text="(auto-detected if empty)", font=("", 8)).grid(row=row, column=1, sticky="w", padx=(5, 0))
        row += 1

        # Output directory
        ttk.Label(frame, text="Output dir:").grid(row=row, column=0, sticky="w", pady=2)
        self.d_output_var = tk.StringVar(value="./reconstructed/")
        ttk.Entry(frame, textvariable=self.d_output_var).grid(row=row, column=1, sticky="ew", pady=2, padx=(5, 0))
        ttk.Button(frame, text="Browse", command=self.browse_d_output).grid(row=row, column=2, pady=2, padx=(5, 0))
        row += 1

        # Separator
        ttk.Separator(frame, orient="horizontal").grid(row=row, column=0, columnspan=3, sticky="ew", pady=10)
        row += 1

        # Options frame
        opts = ttk.LabelFrame(frame, text="Options", padding="5")
        opts.grid(row=row, column=0, columnspan=3, sticky="ew", pady=5)
        row += 1

        # Format
        ttk.Label(opts, text="Output format:").grid(row=0, column=0, sticky="w", pady=2)
        self.d_format_var = tk.StringVar(value="tiff")
        format_combo = ttk.Combobox(opts, textvariable=self.d_format_var, state="readonly", width=10)
        format_combo['values'] = ('tiff', 'dat')
        format_combo.grid(row=0, column=1, sticky="w", pady=2, padx=(5, 0))

        # Workers
        ttk.Label(opts, text="Workers:").grid(row=0, column=2, sticky="w", pady=2, padx=(15, 0))
        self.d_workers_var = tk.StringVar(value="4")
        ttk.Spinbox(opts, textvariable=self.d_workers_var, from_=1, to=16, width=5).grid(row=0, column=3, sticky="w", pady=2, padx=(5, 0))

        # Checkboxes
        self.d_no_roi_var = tk.BooleanVar()
        ttk.Checkbutton(opts, text="Skip ROI patching (for lossless)", variable=self.d_no_roi_var).grid(row=1, column=0, columnspan=4, sticky="w", pady=2)

        # Run button
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=row, column=0, columnspan=3, pady=15)
        self.d_run_btn = ttk.Button(btn_frame, text="Decompress", command=self.run_decompress, width=20)
        self.d_run_btn.pack()

    # Browse handlers for compress tab
    def browse_c_input(self):
        files = filedialog.askopenfilenames(
            title="Select input files",
            filetypes=[("TIFF files", "*.tiff *.tif"), ("DAT files", "*.dat"), ("All files", "*.*")]
        )
        if files:
            if len(files) > 1:
                dir_path = os.path.dirname(files[0])
                ext = os.path.splitext(files[0])[1]
                self.c_input_var.set(f"{dir_path}/*{ext}")
            else:
                self.c_input_var.set(files[0])

    def browse_c_output(self):
        dir_path = filedialog.askdirectory(title="Select output directory")
        if dir_path:
            self.c_output_var.set(dir_path + "/")

    def browse_c_bp2(self):
        files = filedialog.askopenfilenames(
            title="Select biplane files",
            filetypes=[("TIFF files", "*.tiff *.tif"), ("DAT files", "*.dat"), ("All files", "*.*")]
        )
        if files:
            if len(files) > 1:
                dir_path = os.path.dirname(files[0])
                ext = os.path.splitext(files[0])[1]
                self.c_bp2_var.set(f"{dir_path}/*{ext}")
            else:
                self.c_bp2_var.set(files[0])

    # Browse handlers for decompress tab
    def browse_d_video(self):
        files = filedialog.askopenfilenames(
            title="Select video files",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("Zstd files", "*.zst"), ("All files", "*.*")]
        )
        if files:
            if len(files) > 1:
                dir_path = os.path.dirname(files[0])
                ext = os.path.splitext(files[0])[1]
                self.d_video_var.set(f"{dir_path}/*{ext}")
            else:
                self.d_video_var.set(files[0])

    def browse_d_npz(self):
        files = filedialog.askopenfilenames(
            title="Select NPZ files",
            filetypes=[("NPZ files", "*.npz"), ("All files", "*.*")]
        )
        if files:
            if len(files) > 1:
                dir_path = os.path.dirname(files[0])
                self.d_npz_var.set(f"{dir_path}/*.npz")
            else:
                self.d_npz_var.set(files[0])

    def browse_d_output(self):
        dir_path = filedialog.askdirectory(title="Select output directory")
        if dir_path:
            self.d_output_var.set(dir_path + "/")

    # Run handlers
    def run_compress(self):
        if self.running:
            return

        if not self.c_input_var.get():
            messagebox.showerror("Error", "Please select input files")
            return

        self.running = True
        self.c_run_btn.config(state="disabled")
        self.d_run_btn.config(state="disabled")
        self.progress.start()
        self.status_var.set("Compressing...")

        thread = threading.Thread(target=self._do_compress)
        thread.daemon = True
        thread.start()

    def _do_compress(self):
        try:
            from SPARZ import SPARZIP

            input_path = self.c_input_var.get()
            output_path = self.c_output_var.get()
            bp2_path = self.c_bp2_var.get() or None

            os.makedirs(output_path, exist_ok=True)
            stem = Path(input_path.replace('*', '').replace('?', '')).stem or 'output'

            z = SPARZIP(
                path_image_files1=input_path,
                stem=stem,
                output_path=output_path,
                path_image_files2=bp2_path,
                find_peaks=not self.c_no_roi_var.get(),
                extract_metadata=self.c_metadata_var.get(),
                create_single_file=self.c_single_file_var.get(),
                num_workers=int(self.c_workers_var.get()),
            )
            z.run(codec=self.c_codec_var.get(), compression_level=int(self.c_level_var.get()))

            self.root.after(0, lambda: self._on_complete("Compression complete!"))
        except Exception as e:
            self.root.after(0, lambda: self._on_error(str(e)))

    def run_decompress(self):
        if self.running:
            return

        if not self.d_video_var.get():
            messagebox.showerror("Error", "Please select video files")
            return

        self.running = True
        self.c_run_btn.config(state="disabled")
        self.d_run_btn.config(state="disabled")
        self.progress.start()
        self.status_var.set("Decompressing...")

        thread = threading.Thread(target=self._do_decompress)
        thread.daemon = True
        thread.start()

    def _do_decompress(self):
        try:
            from SPARZ import UNSPARZ

            video_path = self.d_video_var.get()
            output_path = self.d_output_var.get()
            npz_path = self.d_npz_var.get() or None

            if npz_path is None and not self.d_no_roi_var.get():
                video_dir = str(Path(video_path).parent)
                npz_path = f"{video_dir}/*.npz"

            os.makedirs(output_path, exist_ok=True)
            stem = Path(video_path.replace('*', '').replace('?', '')).stem or 'output'

            u = UNSPARZ(
                path_sparse_bp1=npz_path,
                path_encoded_bp1=video_path,
                stem=stem,
                output_path=output_path,
                use_roi=not self.d_no_roi_var.get(),
                num_workers=int(self.d_workers_var.get()),
                output_format=self.d_format_var.get(),
            )
            u.run()

            self.root.after(0, lambda: self._on_complete("Decompression complete!"))
        except Exception as e:
            self.root.after(0, lambda: self._on_error(str(e)))

    def _on_complete(self, message):
        self.progress.stop()
        self.status_var.set(message)
        self.c_run_btn.config(state="normal")
        self.d_run_btn.config(state="normal")
        self.running = False
        messagebox.showinfo("Success", message)

    def _on_error(self, error):
        self.progress.stop()
        self.status_var.set("Error")
        self.c_run_btn.config(state="normal")
        self.d_run_btn.config(state="normal")
        self.running = False
        messagebox.showerror("Error", error)


def main():
    """Entry point for sparz-gui command."""
    root = tk.Tk()
    app = SparzGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
