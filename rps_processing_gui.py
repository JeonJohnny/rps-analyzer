"""GUI for processing resistive pulse sensing WAV recordings."""

import os
import queue
import re
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, medfilt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# Median-filter window in seconds used for the reference processing.
MEDIAN_WINDOW_S = 0.5
# Gaussian smoothing width in samples used for the reference processing.
GAUSSIAN_SIGMA_SAMPLES = 500
# Peak-height threshold in estimated noise standard deviations.
DETECTION_THRESHOLD_SIGMA = 5
# Minimum distance between peaks in seconds.
MIN_PEAK_SEPARATION_S = 0.05
# Minimum peak prominence in estimated noise standard deviations.
PEAK_PROMINENCE_SIGMA = 1.0
# Peak width crossing level in estimated noise standard deviations.
WIDTH_LEVEL_SIGMA = 3
# Reverse the recorded signal by default to make negative events positive.
REVERSE_SIGNAL = True
# Resolution used when writing diagnostic plots.
PLOT_DPI = 300


def process_wav(wav_path, params, output_dir, start_time=None, end_time=None, save_outputs=True):
    """Process one WAV while preserving the original numerical operations."""
    # ---- 1. Read WAV ----
    sample_rate, original_data = wavfile.read(wav_path)
    fs = sample_rate

    # ---- 2. Select channel 0 for stereo recordings ----
    if original_data.ndim == 2:
        original_data = original_data[:, 0]
    original_data = original_data.copy()

    # ---- 3. Apply an optional analysis range to the raw samples ----
    start_index = 0
    end_index = len(original_data)
    time_offset = 0.0
    range_note = None
    if start_time is not None or end_time is not None:
        requested_start = 0.0 if start_time is None else start_time
        requested_end = len(original_data) / fs if end_time is None else end_time
        start_index = max(0, min(len(original_data), int(requested_start * fs)))
        requested_end_index = int(requested_end * fs)
        end_index = max(0, min(len(original_data), requested_end_index))
        if requested_end_index > len(original_data):
            range_note = "Analysis range end was clipped to the WAV record duration."
        original_data = original_data[start_index:end_index].copy()
        time_offset = start_index / fs

    # ---- 4. Optionally invert the selected samples ----
    # Unary negation deliberately keeps the WAV's native integer dtype.
    data = -original_data if params["reverse_signal"] else original_data.copy()

    # ---- 5. Estimate the median baseline ----
    # Keep int(fs * window), then make an even kernel odd as in the original.
    window_size = int(fs * params["median_window"])
    if window_size % 2 == 0:
        window_size += 1
    # medfilt zero-pads both edges, and this edge behavior is part of the result.
    median_filtered = medfilt(data, kernel_size=window_size)

    # ---- 6. Smooth the median-filtered baseline ----
    # Sigma is in samples. gaussian_filter1d preserves the integer input dtype,
    # truncating its floating-point convolution back to ADC counts.
    filtered_data = gaussian_filter1d(median_filtered, sigma=params["gaussian_sigma"])

    # ---- 7. Subtract the baseline ----
    # This subtraction also retains NumPy's native integer arithmetic/dtype.
    corrected = data - filtered_data
    time = np.arange(len(data)) / fs + time_offset

    # ---- 8. Estimate noise from the MAD mask ----
    m = np.median(corrected)
    mad = np.median(np.abs(corrected - m))
    sigma_mad = mad / 0.6745
    # Preserve the original asymmetric expression: abs(x-m) < (m + 3*sigma_MAD).
    # It differs from a symmetric 3*sigma_MAD mask and is kept for reproducibility.
    noise_mask = np.abs(corrected - m) < (m + 3 * sigma_mad)
    noise_only = corrected[noise_mask]
    noise_mean = float(np.mean(noise_only)) if len(noise_only) else float("nan")
    noise_std = float(np.std(noise_only)) if len(noise_only) else float("nan")
    thresholds = {level: noise_mean + level * noise_std for level in (3, 4, 5)}
    detection_line = noise_mean + params["detection_threshold"] * noise_std

    # ---- 9. Detect peaks ----
    # Keep integer truncation of separation and guard against a zero distance.
    peak_distance = max(1, int(params["peak_distance"] * fs))
    peak_prominence = params["peak_prominence"] * noise_std
    peaks, _ = find_peaks(
        corrected,
        height=detection_line,
        distance=peak_distance,
        prominence=peak_prominence,
    )

    # ---- 10. Measure widths at the 3-sigma noise level ----
    peak_rows = []
    for peak_number, peak_idx in enumerate(peaks, start=1):
        left = int(peak_idx)
        while left > 0 and corrected[left] > thresholds[WIDTH_LEVEL_SIGMA]:
            left -= 1
        right = int(peak_idx)
        while right < len(corrected) - 1 and corrected[right] > thresholds[WIDTH_LEVEL_SIGMA]:
            right += 1
        peak_rows.append({
            "Peak Index": peak_number,
            "Time (s)": time[peak_idx],
            "Height": corrected[peak_idx],
            "Width (s)": (right - left) / fs,
            "Start (s)": time[left],
            "End (s)": time[right],
            "_left_index": left,
            "_right_index": right,
        })
    peaks_frame = pd.DataFrame(peak_rows, columns=[
        "Peak Index", "Time (s)", "Height", "Width (s)", "Start (s)", "End (s)",
        "_left_index", "_right_index",
    ])

    # ---- 11. Summarize detected peaks ----
    peak_count = len(peaks_frame)
    duration = len(data) / fs
    peak_times = peaks_frame["Time (s)"].to_numpy(dtype=float)
    intervals = np.diff(peak_times)
    heights = peaks_frame["Height"].to_numpy(dtype=float)
    widths = peaks_frame["Width (s)"].to_numpy(dtype=float)
    summary = {
        "Peak count": peak_count,
        "Analysed duration (s)": duration,
        "Peak rate (1/s)": peak_count / duration if duration else float("nan"),
        "Mean height (ADC counts)": float(np.mean(heights)) if peak_count else float("nan"),
        "Median height (ADC counts)": float(np.median(heights)) if peak_count else float("nan"),
        "Mean width (s)": float(np.mean(widths)) if peak_count else float("nan"),
        "Median width (s)": float(np.median(widths)) if peak_count else float("nan"),
        "Mean inter-peak interval (s)": float(np.mean(intervals)) if len(intervals) else float("nan"),
        "SD inter-peak interval (s)": float(np.std(intervals)) if len(intervals) else float("nan"),
    }

    result = {
        "sample_rate": fs, "data": data, "original_data": original_data,
        "time": time, "median_filtered": median_filtered, "baseline": filtered_data,
        "corrected": corrected, "noise_mean": noise_mean, "noise_std": noise_std,
        "sigma_MAD": sigma_mad, "noise_mask": noise_mask, "thresholds": thresholds,
        "detection_line": detection_line, "peaks": peaks,
        "peak_table": peaks_frame, "summary": summary, "duration": duration,
        "window_size": window_size, "time_offset": time_offset, "range_note": range_note,
        "reverse_signal": params["reverse_signal"],
    }

    # ---- 12. Save per-file tables and readable summaries ----
    if save_outputs:
        if output_dir is None:
            file_stem = re.sub(r'[\\/:*?"<>|]', "_", os.path.splitext(os.path.basename(wav_path))[0])
            output_dir = os.path.join(os.path.dirname(wav_path), file_stem + "_processed_files")
        os.makedirs(output_dir, exist_ok=True)
        output_name = os.path.basename(wav_path)
        stem = re.sub(r'[\\/:*?"<>|]', "_", os.path.splitext(output_name)[0])
        pd.DataFrame({"Time (s)": time, "Amplitude": result["original_data"]}).to_csv(
            os.path.join(output_dir, f"{stem}_rawdata.csv"), index=False)
        pd.DataFrame({"Time (s)": time, "Baseline Removed Amplitude": corrected,
                      "Baseline": filtered_data}).to_csv(
            os.path.join(output_dir, f"{stem}_baseline_removed.csv"), index=False)
        peaks_frame[["Peak Index", "Time (s)", "Height", "Width (s)", "Start (s)", "End (s)"]].to_csv(
            os.path.join(output_dir, f"{stem}_05_peak.csv"), index=False)
        pd.DataFrame([summary]).to_csv(os.path.join(output_dir, f"{stem}_06_peak_summary.csv"), index=False)
        with open(os.path.join(output_dir, f"{stem}_06_peak_summary.txt"), "w", encoding="utf-8") as summary_file:
            for key, value in summary.items():
                if isinstance(value, (int, np.integer)):
                    formatted_value = str(value)
                else:
                    formatted_value = f"{value:.6g}"
                summary_file.write(f"{key}: {formatted_value}\n")
        with open(os.path.join(output_dir, f"{stem}_info.txt"), "w", encoding="utf-8") as info_file:
            info_file.write(f"File: {output_name}\nSample rate: {fs} Hz\n")
            info_file.write(f"Samples: {len(data)}\nDuration: {duration:.6g} s\n")
            info_file.write(f"Signal direction: {'Reversed' if params['reverse_signal'] else 'Forward'}\n")
            info_file.write(f"Median window: {params['median_window']:.6g} s ({window_size} samples)\n")
            info_file.write(f"Gaussian sigma: {params['gaussian_sigma']:.6g} samples\n")
            info_file.write(f"Detection threshold: {params['detection_threshold']:.6g} sigma\n")
            info_file.write(f"Minimum separation: {params['peak_distance']:.6g} s\n")
            info_file.write(f"Prominence: {params['peak_prominence']:.6g} sigma\n")
            info_file.write(f"Analysis range: {time_offset:.6g} to {time_offset + duration:.6g} s\n")
            info_file.write(f"Noise mean: {noise_mean:.6g} ADC counts\n")
            info_file.write(f"Noise standard deviation: {noise_std:.6g} ADC counts\n")
            info_file.write(f"Sigma_MAD: {sigma_mad:.6g} ADC counts\n")
            info_file.write(f"Detection level: {detection_line:.6g} ADC counts\n")
            if range_note:
                info_file.write(f"Note: {range_note}\n")

    # ---- 13. Save diagnostic plots ----
    if save_outputs:
        _save_processing_plots(result, output_dir, stem)
    return result


def _save_processing_plots(result, output_dir, stem):
    """Write the six diagnostic plots with Matplotlib's object-oriented API."""
    time = result["time"]
    data = result["data"]
    original_data = result["original_data"]
    baseline = result["baseline"]
    median = result["median_filtered"]
    corrected = result["corrected"]
    peaks = result["peaks"]
    thresholds = result["thresholds"]
    signal_label = "Signal (inverted)" if result["reverse_signal"] else "Signal"

    fig = Figure(figsize=(14, 5))
    ax = fig.subplots()
    ax.plot(time, original_data, label="Raw signal (as recorded)", linewidth=1)
    ax.set_title("Raw Data")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (ADC counts)")
    ax.grid(True)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{stem}_00_raw.png"), dpi=PLOT_DPI)

    fig = Figure(figsize=(14, 5))
    ax = fig.subplots()
    ax.plot(time, data, label=signal_label, linewidth=1)
    ax.plot(time, median, label="Median baseline", linewidth=1)
    ax.set_title("Raw Signal and Median Filter")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (ADC counts)")
    ax.grid(True)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{stem}_01_raw_and_median.png"), dpi=PLOT_DPI)

    fig = Figure(figsize=(14, 5))
    ax = fig.subplots()
    ax.plot(time, data, label=signal_label, linewidth=1)
    ax.plot(time, median, label="Median baseline", linewidth=1)
    ax.plot(time, baseline, label="Gaussian-smoothed baseline", linewidth=1)
    ax.set_title("Gaussian-Filtered Baseline")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (ADC counts)")
    ax.grid(True)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{stem}_02_gaussian_filtered.png"), dpi=PLOT_DPI)

    fig = Figure(figsize=(14, 5))
    ax = fig.subplots()
    ax.plot(time, corrected, label=signal_label, linewidth=1)
    ax.set_title("Baseline-Corrected Signal")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (ADC counts)")
    ax.grid(True)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{stem}_03_baseline.png"), dpi=PLOT_DPI)

    fig = Figure(figsize=(14, 5))
    ax = fig.subplots()
    ax.plot(time, corrected, label=signal_label, linewidth=1)
    ax.plot(time[peaks], corrected[peaks], "rx", label="Detected peaks")
    for level, value in thresholds.items():
        ax.axhline(value, linestyle="--", alpha=0.45, label=f"{level} sigma")
    ax.axhline(result["detection_line"], color="red", label="Detection threshold")
    ax.set_title("Peak Detection")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (ADC counts)")
    ax.grid(True)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{stem}_04_peak_picking.png"), dpi=PLOT_DPI)

    fig = Figure(figsize=(14, 5))
    ax = fig.subplots()
    ax.plot(time, corrected, label=signal_label, linewidth=1)
    ax.plot(time[peaks], corrected[peaks], "rx", label="Detected peaks")
    for peak in result["peak_table"].itertuples(index=False):
        ax.axvline(peak[4], color="gray", linestyle=":", alpha=0.5)
        ax.axvline(peak[5], color="gray", linestyle=":", alpha=0.5)
    ax.set_title("Detected Peaks and 3 Sigma Width Boundaries")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (ADC counts)")
    ax.grid(True)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{stem}_05_peak.png"), dpi=PLOT_DPI)


class App:
    """Single-window tkinter application for selecting and processing WAVs."""

    def __init__(self, root):
        self.root = root
        self.root.title("RPS Processing")
        self.files = []
        self.folder = ""
        self.messages = queue.Queue()
        self.running = False
        self.direction = tk.StringVar(value="Reversed")
        self.time_mode = tk.StringVar(value="Full record")
        self.root.after(100, self._poll_messages)
        self._build_ui()

    def _build_ui(self):
        frame = ttk.Frame(self.root, padding=10)
        frame.pack(fill="both", expand=True)

        select_folder = ttk.Button(frame, text="Select folder", command=self.select_folder)
        select_folder.grid(row=0, column=0, sticky="ew")
        select_file = ttk.Button(frame, text="Select WAV", command=self.select_file)
        select_file.grid(row=0, column=1, sticky="ew")
        self.status = ttk.Label(frame, text="Choose a WAV file or folder")
        self.status.grid(row=0, column=2, columnspan=3, sticky="w")

        ttk.Label(frame, text="Signal direction").grid(row=1, column=0, sticky="w")
        ttk.Radiobutton(frame, text="Forward", variable=self.direction, value="Forward").grid(row=1, column=1, sticky="w")
        ttk.Radiobutton(frame, text="Reversed", variable=self.direction, value="Reversed").grid(row=1, column=2, sticky="w")

        ttk.Label(frame, text="Median window (s)").grid(row=2, column=0, sticky="w")
        self.median_entry = ttk.Entry(frame, width=14)
        self.median_entry.insert(0, str(MEDIAN_WINDOW_S))
        self.median_entry.grid(row=2, column=1, sticky="w")

        ttk.Label(frame, text="Gaussian sigma (samples)").grid(row=3, column=0, sticky="w")
        self.gaussian_entry = ttk.Entry(frame, width=14)
        self.gaussian_entry.insert(0, str(GAUSSIAN_SIGMA_SAMPLES))
        self.gaussian_entry.grid(row=3, column=1, sticky="w")

        ttk.Label(frame, text="Detection threshold (sigma)").grid(row=4, column=0, sticky="w")
        self.threshold_entry = ttk.Entry(frame, width=14)
        self.threshold_entry.insert(0, str(DETECTION_THRESHOLD_SIGMA))
        self.threshold_entry.grid(row=4, column=1, sticky="w")

        ttk.Label(frame, text="Minimum separation (s)").grid(row=5, column=0, sticky="w")
        self.separation_entry = ttk.Entry(frame, width=14)
        self.separation_entry.insert(0, str(MIN_PEAK_SEPARATION_S))
        self.separation_entry.grid(row=5, column=1, sticky="w")

        ttk.Label(frame, text="Prominence (sigma)").grid(row=6, column=0, sticky="w")
        self.prominence_entry = ttk.Entry(frame, width=14)
        self.prominence_entry.insert(0, str(PEAK_PROMINENCE_SIGMA))
        self.prominence_entry.grid(row=6, column=1, sticky="w")

        ttk.Radiobutton(frame, text="Full record", variable=self.time_mode, value="Full record").grid(row=7, column=0, sticky="w")
        ttk.Radiobutton(frame, text="Custom range", variable=self.time_mode, value="Custom range").grid(row=7, column=1, sticky="w")
        ttk.Label(frame, text="Start (s)").grid(row=8, column=0, sticky="w")
        self.start_entry = ttk.Entry(frame, width=12)
        self.start_entry.insert(0, "0")
        self.start_entry.grid(row=8, column=1, sticky="w")
        ttk.Label(frame, text="End (s)").grid(row=8, column=2, sticky="w")
        self.end_entry = ttk.Entry(frame, width=12)
        self.end_entry.insert(0, "60")
        self.end_entry.grid(row=8, column=3, sticky="w")

        self.select_folder_button = select_folder
        self.select_file_button = select_file
        self.preview_button = ttk.Button(frame, text="Preview", command=self.preview)
        self.preview_button.grid(row=9, column=0, sticky="ew")
        self.start_button = ttk.Button(frame, text="Start", command=self.start)
        self.start_button.grid(row=9, column=1, sticky="ew")
        self.buttons = [self.select_folder_button, self.select_file_button, self.preview_button, self.start_button]

        self.log = tk.Text(frame, height=12, width=90)
        self.log.grid(row=10, column=0, columnspan=5, sticky="nsew", pady=(8, 0))
        scrollbar = ttk.Scrollbar(frame, command=self.log.yview)
        scrollbar.grid(row=10, column=5, sticky="ns")
        self.log.configure(yscrollcommand=scrollbar.set)
        frame.columnconfigure(4, weight=1)
        frame.rowconfigure(10, weight=1)

    def _read_parameters(self):
        try:
            median_window = float(self.median_entry.get())
            if not np.isfinite(median_window) or median_window <= 0:
                raise ValueError
        except ValueError:
            median_window = MEDIAN_WINDOW_S
            self.messages.put(f"Invalid Median window (s), using reference value {MEDIAN_WINDOW_S}.")

        try:
            gaussian_sigma = float(self.gaussian_entry.get())
            if not np.isfinite(gaussian_sigma) or gaussian_sigma <= 0:
                raise ValueError
        except ValueError:
            gaussian_sigma = GAUSSIAN_SIGMA_SAMPLES
            self.messages.put(f"Invalid Gaussian sigma (samples), using reference value {GAUSSIAN_SIGMA_SAMPLES}.")

        try:
            detection_threshold = float(self.threshold_entry.get())
            if not np.isfinite(detection_threshold) or detection_threshold <= 0:
                raise ValueError
        except ValueError:
            detection_threshold = DETECTION_THRESHOLD_SIGMA
            self.messages.put(f"Invalid Detection threshold (sigma), using reference value {DETECTION_THRESHOLD_SIGMA}.")

        try:
            peak_distance = float(self.separation_entry.get())
            if not np.isfinite(peak_distance) or peak_distance <= 0:
                raise ValueError
        except ValueError:
            peak_distance = MIN_PEAK_SEPARATION_S
            self.messages.put(f"Invalid Minimum separation (s), using reference value {MIN_PEAK_SEPARATION_S}.")

        try:
            peak_prominence = float(self.prominence_entry.get())
            if not np.isfinite(peak_prominence) or peak_prominence <= 0:
                raise ValueError
        except ValueError:
            peak_prominence = PEAK_PROMINENCE_SIGMA
            self.messages.put(f"Invalid Prominence (sigma), using reference value {PEAK_PROMINENCE_SIGMA}.")

        return {
            "median_window": median_window,
            "gaussian_sigma": gaussian_sigma,
            "detection_threshold": detection_threshold,
            "peak_distance": peak_distance,
            "peak_prominence": peak_prominence,
            "reverse_signal": self.direction.get() == "Reversed",
        }

    def _read_analysis_range(self):
        if self.time_mode.get() == "Full record":
            return None, None
        try:
            start_time = float(self.start_entry.get())
            end_time = float(self.end_entry.get())
            if not np.isfinite(start_time) or not np.isfinite(end_time) or start_time < 0 or end_time <= start_time:
                raise ValueError
        except ValueError:
            self.messages.put("Invalid analysis range, using the full record.")
            return None, None
        return start_time, end_time

    def select_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.folder = folder
            self.files = sorted(name for name in os.listdir(folder) if name.lower().endswith(".wav"))
            self.status.configure(text=f"{len(self.files)} WAV file(s) selected")

    def select_file(self):
        path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if path:
            self.folder, name = os.path.split(path)
            self.files = [name]
            self.status.configure(text=f"Selected {name}")

    def _run_batch(self, files, output_parent, params, start_time, end_time):
        for index, name in enumerate(files, start=1):
            path = os.path.join(self.folder, name)
            self.messages.put(f"Processing {index}/{len(files)}: {name}")
            try:
                output_stem = re.sub(r'[\\/:*?"<>|]', "_", os.path.splitext(name)[0])
                output_dir = os.path.join(output_parent, output_stem + "_processed_files")
                result = process_wav(path, params, output_dir, start_time, end_time)
                if result["range_note"]:
                    self.messages.put(f"{name}: {result['range_note']}")
                self.messages.put(f"Done {name}: {result['summary']['Peak count']} peaks")
            except Exception as exc:
                self.messages.put(f"Failed {name}: {exc}")
        self.messages.put(("__BATCH_DONE__", len(files)))

    def start(self):
        if not self.files:
            messagebox.showinfo("RPS Processing", "Select a WAV file or folder first.")
            return
        params = self._read_parameters()
        start_time, end_time = self._read_analysis_range()
        self._set_running(True)
        worker = threading.Thread(
            target=self._run_batch,
            args=(self.files[:], self.folder, params, start_time, end_time),
            daemon=True,
        )
        worker.start()

    def preview(self):
        if not self.files:
            messagebox.showinfo("RPS Processing", "Select a WAV file or folder first.")
            return
        path = os.path.join(self.folder, self.files[0])
        params = self._read_parameters()
        start_time, end_time = self._read_analysis_range()
        try:
            result = process_wav(path, params, None, start_time, end_time, save_outputs=False)
            if result["range_note"]:
                self.messages.put(result["range_note"])
            signal_label = "Signal (inverted)" if params["reverse_signal"] else "Signal"
            window = tk.Toplevel(self.root)
            window.title("Processing preview")
            fig = Figure(figsize=(12, 8))
            axes = fig.subplots(4, 1)

            axes[0].plot(result["time"], result["data"], label=signal_label)
            axes[0].set_title("Signal")
            axes[0].set_xlabel("Time (s)")
            axes[0].set_ylabel("Amplitude (ADC counts)")
            axes[0].legend(loc="best")

            axes[1].plot(result["time"], result["data"], label=signal_label, alpha=0.5)
            axes[1].plot(result["time"], result["baseline"], label="Gaussian-smoothed baseline")
            axes[1].set_title("Signal and baseline")
            axes[1].set_xlabel("Time (s)")
            axes[1].set_ylabel("Amplitude (ADC counts)")
            axes[1].legend(loc="best")

            axes[2].plot(result["time"], result["corrected"], label=signal_label)
            axes[2].axhline(result["thresholds"][3], linestyle="--", label="3 sigma")
            axes[2].axhline(result["thresholds"][5], linestyle="--", label="5 sigma")
            axes[2].axhline(result["detection_line"], color="red", label="Detection threshold")
            axes[2].set_title("Baseline-corrected signal and thresholds")
            axes[2].set_xlabel("Time (s)")
            axes[2].set_ylabel("Amplitude (ADC counts)")
            axes[2].legend(loc="best")

            axes[3].plot(result["time"], result["corrected"], label=signal_label)
            axes[3].plot(result["time"][result["peaks"]], result["corrected"][result["peaks"]], "rx", label="Detected peaks")
            for peak in result["peak_table"].itertuples(index=False):
                axes[3].axvline(peak[4], color="gray", linestyle=":", alpha=0.5)
                axes[3].axvline(peak[5], color="gray", linestyle=":", alpha=0.5)
            axes[3].set_title("Detected peaks and width boundaries")
            axes[3].set_xlabel("Time (s)")
            axes[3].set_ylabel("Amplitude (ADC counts)")
            axes[3].legend(loc="best")

            for axis in axes:
                axis.grid(True)
            fig.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
        except Exception as exc:
            messagebox.showerror("Preview failed", str(exc))

    def _set_running(self, running):
        self.running = running
        for button in self.buttons:
            button.configure(state="disabled" if running else "normal")

    def _poll_messages(self):
        try:
            while True:
                message = self.messages.get_nowait()
                if isinstance(message, tuple) and message[0] == "__BATCH_DONE__":
                    self._set_running(False)
                    self.status.configure(text=f"Processing complete: {message[1]} file(s)")
                else:
                    self.log.insert("end", message + "\n")
                    self.log.see("end")
                    self.status.configure(text=message)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_messages)


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
