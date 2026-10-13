"""
Pure-Python High-Resolution PNG Chart Generator
Creates results/f1_comparison.png without third-party dependencies using zlib and struct.
"""

import os
import sys
import zlib
import struct
import json
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
raw_json_path = os.path.join(RESULTS_DIR, "raw_benchmark_results.json")
output_png = os.path.join(RESULTS_DIR, "f1_comparison.png")

with open(raw_json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

summary = data["summary_stats"]

# Models and values
models = [
    ("Federated GraphSAGE\n(G.O.A.T. Shield)", summary["federated_graphsage"]["f1"]["mean_pct"], summary["federated_graphsage"]["f1"]["std_pct"], (37, 99, 235)),    # Blue
    ("Isolated Silos\n(Per-Bank Avg)", summary["isolated_silos"]["f1"]["mean_pct"], summary["isolated_silos"]["f1"]["std_pct"], (245, 158, 11)),                   # Amber
    ("Centralized GraphSAGE\n(Upper Bound)", summary["centralized_graphsage"]["f1"]["mean_pct"], summary["centralized_graphsage"]["f1"]["std_pct"], (16, 185, 129)), # Green
    ("Rule-Based Baseline\n(R1-R6 Heuristic)", summary["rule_baseline"]["f1"]["mean_pct"], summary["rule_baseline"]["f1"]["std_pct"], (220, 38, 38))                # Red
]

# Canvas dimensions
WIDTH = 900
HEIGHT = 560
canvas = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255  # White background

# Simple bitmap 5x7 font for rendering clean text
FONT_5X7 = {
    '0': ["01110","10001","10011","10101","11001","10001","01110"],
    '1': ["00100","01100","00100","00100","00100","00100","01110"],
    '2': ["01110","10001","00001","00010","00100","01000","11111"],
    '3': ["01110","10001","00001","00110","00001","10001","01110"],
    '4': ["00010","00110","01010","10010","11111","00010","00010"],
    '5': ["11111","10000","11110","00001","00001","10001","01110"],
    '6': ["01110","10000","11110","10001","10001","10001","01110"],
    '7': ["11111","00001","00010","00100","01000","01000","01000"],
    '8': ["01110","10001","10001","01110","10001","10001","01110"],
    '9': ["01110","10001","10001","01111","00001","00001","01110"],
    '.': ["00000","00000","00000","00000","00000","01100","01100"],
    '%': ["11001","11010","00100","01000","01011","10011","00000"],
    '+': ["00000","00100","00100","11111","00100","00100","00000"],
    '-': ["00000","00000","00000","11111","00000","00000","00000"],
    '/': ["00001","00010","00010","00100","01000","01000","10000"],
    ':': ["00000","01100","01100","00000","01100","01100","00000"],
    '(': ["00010","00100","01000","01000","01000","00100","00010"],
    ')': ["01000","00100","00010","00010","00010","00100","01000"],
    ' ': ["00000","00000","00000","00000","00000","00000","00000"],
    'A': ["01110","10001","10001","11111","10001","10001","10001"],
    'B': ["11110","10001","10001","11110","10001","10001","11110"],
    'C': ["01110","10001","10000","10000","10000","10001","01110"],
    'D': ["11110","10001","10001","10001","10001","10001","11110"],
    'E': ["11111","10000","10000","11110","10000","10000","11111"],
    'F': ["11111","10000","10000","11110","10000","10000","10000"],
    'G': ["01110","10001","10000","10111","10001","10001","01110"],
    'H': ["10001","10001","10001","11111","10001","10001","10001"],
    'I': ["01110","00100","00100","00100","00100","00100","01110"],
    'L': ["10000","10000","10000","10000","10000","10000","11111"],
    'M': ["10001","11011","10101","10001","10001","10001","10001"],
    'N': ["10001","11001","10101","10011","10001","10001","10001"],
    'O': ["01110","10001","10001","10001","10001","10001","01110"],
    'P': ["11110","10001","10001","11110","10000","10000","10000"],
    'R': ["11110","10001","10001","11110","10100","10010","10001"],
    'S': ["01110","10001","10000","01110","00001","10001","01110"],
    'T': ["11111","00100","00100","00100","00100","00100","00100"],
    'U': ["10001","10001","10001","10001","10001","10001","01110"],
    'V': ["10001","10001","10001","10001","10001","01010","00100"],
    'W': ["10001","10001","10001","10101","10101","11011","10001"],
    'X': ["10001","10001","01010","00100","01010","10001","10001"],
    'Y': ["10001","10001","01010","00100","00100","00100","00100"],
    'a': ["00000","00000","01110","00001","01111","10001","01111"],
    'b': ["10000","10000","11110","10001","10001","10001","11110"],
    'c': ["00000","00000","01110","10000","10000","10001","01110"],
    'd': ["00001","00001","01111","10001","10001","10001","01111"],
    'e': ["00000","00000","01110","10001","11111","10000","01110"],
    'f': ["00110","01000","11110","01000","01000","01000","01000"],
    'g': ["00000","00000","01111","10001","01111","00001","01110"],
    'h': ["10000","10000","11110","10001","10001","10001","10001"],
    'i': ["00100","00000","01100","00100","00100","00100","01110"],
    'k': ["10000","10000","10010","10100","11000","10100","10010"],
    'l': ["01100","00100","00100","00100","00100","00100","01110"],
    'm': ["00000","00000","11010","10101","10101","10001","10001"],
    'n': ["00000","00000","11110","10001","10001","10001","10001"],
    'o': ["00000","00000","01110","10001","10001","10001","01110"],
    'p': ["00000","00000","11110","10001","11110","10000","10000"],
    'r': ["00000","00000","10110","11001","10000","10000","10000"],
    's': ["00000","00000","01111","10000","01110","00001","11110"],
    't': ["01000","01000","11110","01000","01000","01001","00110"],
    'u': ["00000","00000","10001","10001","10001","10011","01101"],
    'v': ["00000","00000","10001","10001","01010","01010","00100"],
    'w': ["00000","00000","10001","10101","10101","11011","01010"],
    'x': ["00000","00000","10001","01010","00100","01010","10001"],
    'y': ["00000","00000","10001","10001","01111","00001","01110"]
}

def draw_text(canvas, text, x0, y0, scale=2, color=(30, 41, 59)):
    cur_x = x0
    for ch in text:
        bitmap = FONT_5X7.get(ch, FONT_5X7.get(' '))
        for r_idx, row in enumerate(bitmap):
            for c_idx, pixel in enumerate(row):
                if pixel == '1':
                    px = cur_x + c_idx * scale
                    py = y0 + r_idx * scale
                    canvas[py:py+scale, px:px+scale] = color
        cur_x += 6 * scale

# Margins and plot area
plot_left = 100
plot_right = 850
plot_top = 80
plot_bottom = 440
plot_h = plot_bottom - plot_top

# Draw Background & Grid lines
for y_pct in range(0, 101, 20):
    y_pos = int(plot_bottom - (y_pct / 100.0) * plot_h)
    canvas[y_pos, plot_left:plot_right] = (226, 232, 240)  # Light gridline
    draw_text(canvas, f"{y_pct:3d}%", plot_left - 65, y_pos - 7, scale=2, color=(100, 116, 139))

# Draw Axes
canvas[plot_bottom:plot_bottom+2, plot_left:plot_right] = (30, 41, 59)
canvas[plot_top:plot_bottom, plot_left:plot_left+2] = (30, 41, 59)

# Draw Title & Subtitle
draw_text(canvas, "G.O.A.T. SHIELD: 2,000 ACCOUNTS F1 BENCHMARK", 170, 25, scale=3, color=(15, 23, 42))
draw_text(canvas, "Mean +/- Std across 5 Random Seeds (0, 1, 2, 3, 4) on 400 Held-Out Test Accounts", 120, 55, scale=2, color=(71, 85, 105))

# Draw Bars & Error Bars
bar_w = 90
gap = 75
start_x = plot_left + 50

for idx, (label, f1_mean, f1_std, color) in enumerate(models):
    bar_x = start_x + idx * (bar_w + gap)
    bar_height = int((f1_mean / 100.0) * plot_h)
    bar_top = plot_bottom - bar_height

    # Bar rectangle
    canvas[bar_top:plot_bottom, bar_x:bar_x+bar_w] = color
    # Border
    canvas[bar_top:plot_bottom, bar_x:bar_x+2] = (15, 23, 42)
    canvas[bar_top:plot_bottom, bar_x+bar_w-2:bar_x+bar_w] = (15, 23, 42)
    canvas[bar_top:bar_top+2, bar_x:bar_x+bar_w] = (15, 23, 42)

    # Error bar
    err_h = int((f1_std / 100.0) * plot_h)
    err_top = max(plot_top, bar_top - err_h)
    err_bottom = min(plot_bottom, bar_top + err_h)
    mid_x = bar_x + bar_w // 2

    canvas[err_top:err_bottom, mid_x-1:mid_x+2] = (15, 23, 42)
    canvas[err_top:err_top+2, mid_x-12:mid_x+13] = (15, 23, 42)
    canvas[err_bottom-2:err_bottom, mid_x-12:mid_x+13] = (15, 23, 42)

    # Value label on top
    val_str = f"{f1_mean:.1f}%"
    draw_text(canvas, val_str, bar_x + 12, err_top - 20, scale=2, color=(15, 23, 42))

    # X-axis label (split lines)
    lines = label.split("\n")
    for l_idx, line in enumerate(lines):
        line_w = len(line) * 12
        lbl_x = max(plot_left, mid_x - line_w // 2)
        draw_text(canvas, line, lbl_x, plot_bottom + 15 + l_idx * 18, scale=2, color=(51, 65, 85))

# Pure Python PNG Writer
def write_png(buf, width, height, filepath):
    header = b'\x89PNG\r\n\x1a\n'
    ihdr_data = struct.pack("!2I5B", width, height, 8, 2, 0, 0, 0)
    ihdr_crc = zlib.crc32(b'IHDR' + ihdr_data)
    ihdr = struct.pack("!I", len(ihdr_data)) + b'IHDR' + ihdr_data + struct.pack("!I", ihdr_crc)

    raw_data = bytearray()
    for y in range(height):
        raw_data.append(0)  # filter type 0
        raw_data.extend(buf[y].tobytes())

    compressed = zlib.compress(bytes(raw_data), 9)
    idat_crc = zlib.crc32(b'IDAT' + compressed)
    idat = struct.pack("!I", len(compressed)) + b'IDAT' + compressed + struct.pack("!I", idat_crc)

    iend_crc = zlib.crc32(b'IEND')
    iend = struct.pack("!I", 0) + b'IEND' + struct.pack("!I", iend_crc)

    with open(filepath, "wb") as f:
        f.write(header + ihdr + idat + iend)

write_png(canvas, WIDTH, HEIGHT, output_png)
print(f"Successfully generated pure-Python PNG chart at {output_png} ({os.path.getsize(output_png)} bytes)")
