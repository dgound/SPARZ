#!/usr/bin/env python
"""SPARZ Graphical User Interface (tkinter)"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path


class SparzGUI:
    """GUI for SPARZ compression."""

    def __init__(self, root):
        self.root = root
        self.root.title("SPARZ - Compress")
        self.root.geometry("500x400")
        self.root.resizable(True, True)

        self.running = False
        self.create_widgets()

    def create_widgets(self):
        # Main frame with padding
        main = ttk.Frame(self.root, padding="10")
        main.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)

        row = 0

        # Input files
        ttk.Label(main, text="Input files:").grid(row=row, column=0, sticky="w", pady=2)
        self.input_var = tk.StringVar()
        ttk.Entry(main, textvariable=self.input_var).grid(row=row, column=1, sticky="ew", pady=2, padx=(5,0))
        ttk.Button(main, text="Browse", command=self.browse_input).grid(row=row, column=2, pady=2, padx=(5,0))
        row += 1

        # Output directory
        ttk.Label(main, text="Output dir:").grid(row=row, column=0, sticky="w", pady=2)
        self.output_var = tk.StringVar(value="./output/")
        ttk.Entry(main, textvariable=self.output_var).grid(row=row, column=1, sticky="ew", pady=2, padx=(5,0))
        ttk.Button(main, text="Browse", command=self.browse_output).grid(row=row, column=2, pady=2, padx=(5,0))
        row += 1

        # Biplane (optional)
        ttk.Label(main, text="Biplane (opt):").grid(row=row, column=0, sticky="w", pady=2)
        self.bp2_var = tk.StringVar()
        ttk.Entry(main, textvariable=self.bp2_var).grid(row=row, column=1, sticky="ew", pady=2, padx=(5,0))
        ttk.Button(main, text="Browse", command=self.browse_bp2).grid(row=row, column=2, pady=2, padx=(5,0))
        row += 1

        # Separator
        ttk.Separator(main, orient="horizontal").grid(row=row, column=0, columnspan=3, sticky="ew", pady=10)
        row += 1

        # Options frame
        opts = ttk.LabelFrame(main, text="Options", padding="5")
        opts.grid(row=row, column=0, columnspan=3, sticky="ew", pady=5)
        opts.columnconfigure(1, weight=1)
        row += 1

        # Codec
        ttk.Label(opts, text="Codec:").grid(row=0, column=0, sticky="w", pady=2)
        self.codec_var = tk.StringVar(value="x265")
        codec_combo = ttk.Combobox(opts, textvariable=self.codec_var, state="readonly", width=15)
        codec_combo['values'] = ('x265', 'av1', 'x264', 'ffv1', 'prores', 'zstd')
        codec_combo.grid(row=0, column=1, sticky="w", pady=2, padx=(5,0))

        # Level
        ttk.Label(opts, text="Level:").grid(row=0, column=2, sticky="w", pady=2, padx=(15,0))
        self.level_var = tk.StringVar(value="0")
        level_combo = ttk.Combobox(opts, textvariable=self.level_var, state="readonly", width=5)
        level_combo['values'] = ('0', '1', '2', '3')
        level_combo.grid(row=0, column=3, sticky="w", pady=2, padx=(5,0))

        # Workers
        ttk.Label(opts, text="Workers:").grid(row=1, column=0, sticky="w", pady=2)
        self.workers_var = tk.StringVar(value="4")
        ttk.Spinbox(opts, textvariable=self.workers_var, from_=1, to=16, width=5).grid(row=1, column=1, sticky="w", pady=2, padx=(5,0))

        # Checkboxes
        self.metadata_var = tk.BooleanVar()
        ttk.Checkbutton(opts, text="Extract metadata", variable=self.metadata_var).grid(row=2, column=0, columnspan=2, sticky="w", pady=2)

        self.single_file_var = tk.BooleanVar()
        ttk.Checkbutton(opts, text="Single MKV file", variable=self.single_file_var).grid(row=2, column=2, columnspan=2, sticky="w", pady=2)

        self.no_roi_var = tk.BooleanVar()
        ttk.Checkbutton(opts, text="Disable ROI detection", variable=self.no_roi_var).grid(row=3, column=0, columnspan=2, sticky="w", pady=2)

        # Progress
        ttk.Label(main, text="Status:").grid(row=row, column=0, sticky="w", pady=(10,2))
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(main, textvariable=self.status_var).grid(row=row, column=1, columnspan=2, sticky="w", pady=(10,2), padx=(5,0))
        row += 1

        self.progress = ttk.Progressbar(main, mode='indeterminate')
        self.progress.grid(row=row, column=0, columnspan=3, sticky="ew", pady=5)
        row += 1

        # Buttons
        btn_frame = ttk.Frame(main)
        btn_frame.grid(row=row, column=0, columnspan=3, pady=10)

        self.run_btn = ttk.Button(btn_frame, text="Compress", command=self.run_compress)
        self.run_btn.pack(side="left", padx=5)

        ttk.Button(btn_frame, text="Quit", command=self.root.quit).pack(side="left", padx=5)

    def browse_input(self):
        files = filedialog.askopenfilenames(
            title="Select input files",
            filetypes=[("TIFF files", "*.tiff *.tif"), ("DAT files", "*.dat"), ("All files", "*.*")]
        )
        if files:
            # Convert to glob pattern if multiple files in same dir
            if len(files) > 1:
                dir_path = os.path.dirname(files[0])
                ext = os.path.splitext(files[0])[1]
                self.input_var.set(f"{dir_path}/*{ext}")
            else:
                self.input_var.set(files[0])

    def browse_output(self):
        dir_path = filedialog.askdirectory(title="Select output directory")
        if dir_path:
            self.output_var.set(dir_path + "/")

    def browse_bp2(self):
        files = filedialog.askopenfilenames(
            title="Select biplane files",
            filetypes=[("TIFF files", "*.tiff *.tif"), ("DAT files", "*.dat"), ("All files", "*.*")]
        )
        if files:
            if len(files) > 1:
                dir_path = os.path.dirname(files[0])
                ext = os.path.splitext(files[0])[1]
                self.bp2_var.set(f"{dir_path}/*{ext}")
            else:
                self.bp2_var.set(files[0])

    def run_compress(self):
        if self.running:
            return

        if not self.input_var.get():
            messagebox.showerror("Error", "Please select input files")
            return

        self.running = True
        self.run_btn.config(state="disabled")
        self.progress.start()
        self.status_var.set("Compressing...")

        # Run in thread to keep UI responsive
        thread = threading.Thread(target=self._do_compress)
        thread.daemon = True
        thread.start()

    def _do_compress(self):
        try:
            from SPARZ import SPARZIP

            input_path = self.input_var.get()
            output_path = self.output_var.get()
            bp2_path = self.bp2_var.get() or None

            # Create output dir
            os.makedirs(output_path, exist_ok=True)

            # Get stem from input
            stem = Path(input_path.replace('*', '').replace('?', '')).stem or 'output'

            z = SPARZIP(
                path_image_files1=input_path,
                stem=stem,
                output_path=output_path,
                path_image_files2=bp2_path,
                find_peaks=not self.no_roi_var.get(),
                extract_metadata=self.metadata_var.get(),
                create_single_file=self.single_file_var.get(),
                num_workers=int(self.workers_var.get()),
            )
            z.run(codec=self.codec_var.get(), compression_level=int(self.level_var.get()))

            self.root.after(0, lambda: self._on_complete("Compression complete!"))
        except Exception as e:
            self.root.after(0, lambda: self._on_error(str(e)))

    def _on_complete(self, message):
        self.progress.stop()
        self.status_var.set(message)
        self.run_btn.config(state="normal")
        self.running = False
        messagebox.showinfo("Success", message)

    def _on_error(self, error):
        self.progress.stop()
        self.status_var.set("Error")
        self.run_btn.config(state="normal")
        self.running = False
        messagebox.showerror("Error", error)


class UnsparzGUI:
    """GUI for UNSPARZ decompression."""

    def __init__(self, root):
        self.root = root
        self.root.title("UNSPARZ - Decompress")
        self.root.geometry("500x350")
        self.root.resizable(True, True)

        self.running = False
        self.create_widgets()

    def create_widgets(self):
        # Main frame with padding
        main = ttk.Frame(self.root, padding="10")
        main.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)

        row = 0

        # Video files
        ttk.Label(main, text="Video files:").grid(row=row, column=0, sticky="w", pady=2)
        self.video_var = tk.StringVar()
        ttk.Entry(main, textvariable=self.video_var).grid(row=row, column=1, sticky="ew", pady=2, padx=(5,0))
        ttk.Button(main, text="Browse", command=self.browse_video).grid(row=row, column=2, pady=2, padx=(5,0))
        row += 1

        # NPZ files (optional)
        ttk.Label(main, text="NPZ files:").grid(row=row, column=0, sticky="w", pady=2)
        self.npz_var = tk.StringVar()
        ttk.Entry(main, textvariable=self.npz_var).grid(row=row, column=1, sticky="ew", pady=2, padx=(5,0))
        ttk.Button(main, text="Browse", command=self.browse_npz).grid(row=row, column=2, pady=2, padx=(5,0))
        ttk.Label(main, text="(auto-detected if empty)", font=("", 8)).grid(row=row+1, column=1, sticky="w", padx=(5,0))
        row += 2

        # Output directory
        ttk.Label(main, text="Output dir:").grid(row=row, column=0, sticky="w", pady=2)
        self.output_var = tk.StringVar(value="./reconstructed/")
        ttk.Entry(main, textvariable=self.output_var).grid(row=row, column=1, sticky="ew", pady=2, padx=(5,0))
        ttk.Button(main, text="Browse", command=self.browse_output).grid(row=row, column=2, pady=2, padx=(5,0))
        row += 1

        # Separator
        ttk.Separator(main, orient="horizontal").grid(row=row, column=0, columnspan=3, sticky="ew", pady=10)
        row += 1

        # Options frame
        opts = ttk.LabelFrame(main, text="Options", padding="5")
        opts.grid(row=row, column=0, columnspan=3, sticky="ew", pady=5)
        row += 1

        # Format
        ttk.Label(opts, text="Output format:").grid(row=0, column=0, sticky="w", pady=2)
        self.format_var = tk.StringVar(value="tiff")
        format_combo = ttk.Combobox(opts, textvariable=self.format_var, state="readonly", width=10)
        format_combo['values'] = ('tiff', 'dat')
        format_combo.grid(row=0, column=1, sticky="w", pady=2, padx=(5,0))

        # Workers
        ttk.Label(opts, text="Workers:").grid(row=0, column=2, sticky="w", pady=2, padx=(15,0))
        self.workers_var = tk.StringVar(value="4")
        ttk.Spinbox(opts, textvariable=self.workers_var, from_=1, to=16, width=5).grid(row=0, column=3, sticky="w", pady=2, padx=(5,0))

        # Checkboxes
        self.no_roi_var = tk.BooleanVar()
        ttk.Checkbutton(opts, text="Skip ROI patching (for lossless)", variable=self.no_roi_var).grid(row=1, column=0, columnspan=4, sticky="w", pady=2)

        # Progress
        ttk.Label(main, text="Status:").grid(row=row, column=0, sticky="w", pady=(10,2))
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(main, textvariable=self.status_var).grid(row=row, column=1, columnspan=2, sticky="w", pady=(10,2), padx=(5,0))
        row += 1

        self.progress = ttk.Progressbar(main, mode='indeterminate')
        self.progress.grid(row=row, column=0, columnspan=3, sticky="ew", pady=5)
        row += 1

        # Buttons
        btn_frame = ttk.Frame(main)
        btn_frame.grid(row=row, column=0, columnspan=3, pady=10)

        self.run_btn = ttk.Button(btn_frame, text="Decompress", command=self.run_decompress)
        self.run_btn.pack(side="left", padx=5)

        ttk.Button(btn_frame, text="Quit", command=self.root.quit).pack(side="left", padx=5)

    def browse_video(self):
        files = filedialog.askopenfilenames(
            title="Select video files",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("Zstd files", "*.zst"), ("All files", "*.*")]
        )
        if files:
            if len(files) > 1:
                dir_path = os.path.dirname(files[0])
                ext = os.path.splitext(files[0])[1]
                self.video_var.set(f"{dir_path}/*{ext}")
            else:
                self.video_var.set(files[0])

    def browse_npz(self):
        files = filedialog.askopenfilenames(
            title="Select NPZ files",
            filetypes=[("NPZ files", "*.npz"), ("All files", "*.*")]
        )
        if files:
            if len(files) > 1:
                dir_path = os.path.dirname(files[0])
                self.npz_var.set(f"{dir_path}/*.npz")
            else:
                self.npz_var.set(files[0])

    def browse_output(self):
        dir_path = filedialog.askdirectory(title="Select output directory")
        if dir_path:
            self.output_var.set(dir_path + "/")

    def run_decompress(self):
        if self.running:
            return

        if not self.video_var.get():
            messagebox.showerror("Error", "Please select video files")
            return

        self.running = True
        self.run_btn.config(state="disabled")
        self.progress.start()
        self.status_var.set("Decompressing...")

        thread = threading.Thread(target=self._do_decompress)
        thread.daemon = True
        thread.start()

    def _do_decompress(self):
        try:
            from SPARZ import UNSPARZ

            video_path = self.video_var.get()
            output_path = self.output_var.get()
            npz_path = self.npz_var.get() or None

            # Auto-detect NPZ if not provided
            if npz_path is None and not self.no_roi_var.get():
                video_dir = str(Path(video_path).parent)
                npz_path = f"{video_dir}/*.npz"

            # Create output dir
            os.makedirs(output_path, exist_ok=True)

            # Get stem from input
            stem = Path(video_path.replace('*', '').replace('?', '')).stem or 'output'

            u = UNSPARZ(
                path_sparse_bp1=npz_path,
                path_encoded_bp1=video_path,
                stem=stem,
                output_path=output_path,
                use_roi=not self.no_roi_var.get(),
                num_workers=int(self.workers_var.get()),
                output_format=self.format_var.get(),
            )
            u.run()

            self.root.after(0, lambda: self._on_complete("Decompression complete!"))
        except Exception as e:
            self.root.after(0, lambda: self._on_error(str(e)))

    def _on_complete(self, message):
        self.progress.stop()
        self.status_var.set(message)
        self.run_btn.config(state="normal")
        self.running = False
        messagebox.showinfo("Success", message)

    def _on_error(self, error):
        self.progress.stop()
        self.status_var.set("Error")
        self.run_btn.config(state="normal")
        self.running = False
        messagebox.showerror("Error", error)


def main_sparz_gui():
    """Entry point for sparz-gui command."""
    root = tk.Tk()
    app = SparzGUI(root)
    root.mainloop()


def main_unsparz_gui():
    """Entry point for unsparz-gui command."""
    root = tk.Tk()
    app = UnsparzGUI(root)
    root.mainloop()


if __name__ == '__main__':
    # Default to compress GUI
    main_sparz_gui()
