# RPS signal processing

## Overview

This repository accompanies **“A Tunable Micro-/Nanogap-Based Resistive Pulse Sensing for Particle and Biomaterial Detection.”** It contains the tkinter analysis program and one representative WAV recording (RPS signal). The program estimates a baseline, detects translocation peaks, measures their heights and widths, and saves diagnostic plots and tables. 

## Requirements and installation

Python 3.10 or newer is required. Install the Python packages with:

```bash
pip install -r requirements.txt
```

Tkinter is included with Python. On Linux, install the system package `python3-tk` if needed. 

The software was tested with Python 3.13.9, NumPy 2.3.5, SciPy 1.16.3, pandas 2.3.3, and Matplotlib 3.10.6.

## Usage

1. Start the interface with `python rps_processing_gui.py`.
2. Select **Select WAV** and choose `data/8V_1.wav`, or select its containing folder.
3. Keep **Reversed** selected for this recording, since its translocation peaks are negative-going.
4. Keep **Full record** selected or choose a custom `[start, end)` time interval in seconds. The analysis range trims the raw samples to `[start, end)` before processing, equivalent to analysing a WAV file containing only those samples.
5. Review parameters, optionally click **Preview**, then click **Start**.
6. Results are saved beside each input WAV in a `<name>_processed_files/` directory.

## Parameters

| Name | Reference value | Unit | Meaning |
|---|---:|---|---|
| Median window | 0.5 | s | Window used to estimate the slow baseline (2,001 samples at 4 kHz). |
| Gaussian sigma | 500 | samples | Gaussian smoothing sigma applied after the median filter (0.125 s at 4 kHz). |
| Detection threshold | 5 | σ | Peak-height threshold above the estimated noise mean. |
| Minimum peak separation | 0.05 | s | Minimum distance between local maxima. |
| Prominence | 1 | σ | Minimum peak prominence relative to the estimated noise standard deviation. |
| Signal direction | Reversed | — | Invert the recorded signal before processing; this recording has negative-going peaks. |
| Width level | 3 | σ | Width boundaries are where the corrected trace crosses the mean + 3σ noise level. |

Invalid or nonpositive entries fall back to the reference value and are logged.

## Processing pipeline

1. Read the WAV and use channel 0 for stereo files. Optionally invert the signal.
2. Apply a zero-padded median filter using `int(fs × window)`, incremented by one if even.
3. Apply `gaussian_filter1d` with sigma in samples.
4. Subtract the smoothed baseline from the signal.
5. Estimate noise using `m = median(corrected)`, `MAD = median(abs(corrected - m))`, and `sigma_MAD = MAD / 0.6745`. The coded mask is `abs(corrected - m) < (m + 3 * sigma_MAD)`.
6. Detect peaks at `noise_mean + threshold × noise_std`, with the configured minimum separation and prominence.
7. Measure width by walking left and right from each peak while the corrected signal is above `noise_mean + 3 × noise_std`.

## Outputs

Each input produces `<name>_processed_files/` containing:

- `<name>_info.txt`: sample rate, record length, parameters, analysis range, noise statistics, and thresholds.
- `<name>_rawdata.csv`: `Time (s)`, `Amplitude` (ADC counts).
- `<name>_baseline_removed.csv`: `Time (s)`, `Baseline Removed Amplitude`, and `Baseline` (ADC counts).
- `<name>_05_peak.csv`: `Peak Index`, `Time (s)`, `Height` (ADC counts), `Width (s)`, `Start (s)`, and `End (s)`.
- `<name>_06_peak_summary.txt` and `.csv`: peak count, analyzed duration, peak rate, height and width summaries, and inter-peak interval summaries.
- `_00_raw.png`, `_01_raw_and_median.png`, `_02_gaussian_filtered.png`, `_03_baseline.png`, `_04_peak_picking.png`, and `_05_peak.png`: processing diagnostics.

## Implementation notes kept for reproducibility

The WAV's native integer dtype is retained during inversion, median filtering, Gaussian filtering, and baseline subtraction. In particular, Gaussian filtering truncates back to integer ADC counts. The noise mask above is intentionally the original asymmetric expression, including its dependence on `m`. The median filter uses SciPy's zero-padded edge behavior. A custom time range trims the raw samples to `[start, end)` before processing, matching processing of a WAV containing those same samples.

## Changes from the program version used in the paper

The interface uses tkinter. Preview and batch processing use the same pipeline; analysis ranges trim the data before processing; width markers use the measured left and right crossings; plots use automatic y-limits; minimum peak distance has a one-sample guard; recordings with no peaks still produce outputs with NaN summary statistics; one summary file pair replaces duplicate summary files; the default direction is **Reversed**; and numeric entries accept floating-point values with fallback to reference values. These interface and output changes do not alter the processing results at the reference settings, as checked against the original 73-peak output.

## Example data

`data/8V_1.wav` is a representative recording from a device with a 10 μm-high bridge channel (gap ≈2 μm), measured with 1.0 μm polystyrene particles at 0.01‰ in 1× PBS containing 0.1% Tween 20 under an 8 V bias. Signals were amplified (gain 1,000) and digitized at 4 kHz with a 24-bit DAQ (DT9837A); the file is mono, 60.03 s long, and stores the 24-bit samples in a 32-bit integer container (raw ADC counts). Translocation peaks are negative-going in this recording, so use **Reversed**.

## License

The code is provided under the MIT License; see [LICENSE](LICENSE). 