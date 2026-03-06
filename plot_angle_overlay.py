import os
import glob
import csv
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Optional

HEADER_MARKERS = {
    "波長(nm)",
    "Wavelength(nm)",
    "Wavelength (nm)",
}

def read_spectrum(csv_path: str) -> Tuple[List[float], List[float], Optional[str]]:
    """讀取光譜儀輸出的 CSV 檔案 (380-780 nm)"""
    wavelengths, values = [], []
    unit_label = None

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        in_data = False
        for row in reader:
            if not row: continue
            if not in_data:
                first = row[0].strip()
                if first in HEADER_MARKERS:
                    in_data = True
                    if len(row) > 1:
                        unit_label = row[1].strip() or unit_label
                continue

            if len(row) < 2: continue
            try:
                wl = float(row[0].strip())
                val = float(row[1].strip())
            except ValueError:
                continue

            if 380 <= wl <= 780:
                wavelengths.append(wl)
                values.append(val)

    if not wavelengths:
        print(f"警告：在 {csv_path} 中找不到光譜資料。")
    return wavelengths, values, unit_label

def main():
    # 設定資料夾路徑與輸出路徑
    data_dir = "data"
    out_dir = "image_overlay"
    os.makedirs(out_dir, exist_ok=True)

    # 建立字典來對檔案進行分組
    # 資料結構: groups[(color, device)] = {angle: filepath}
    groups = defaultdict(dict)

    # 用正規表達式自動解析檔名
    # 範例: yellow0mini-222500.csv -> color: yellow, angle: 0, device: mini
    # pattern 解釋: (非數字字串) + (0或30或60) + (mini或scr) + (-後面的任意字元)
    filename_pattern = re.compile(r'^(.*?)(0|30|60)(mini|scr)-.*\.csv$', re.IGNORECASE)

    # 1. 遍歷資料夾，自動分組
    for path in glob.glob(os.path.join(data_dir, "*.csv")):
        basename = os.path.basename(path)
        match = filename_pattern.match(basename)
        
        if match:
            color = match.group(1).lower()
            angle = match.group(2)
            device = match.group(3).lower()
            groups[(color, device)][angle] = path

    if not groups:
        print("沒有找到符合命名規則的檔案。請確認檔名是否包含角度(0,30,60)與設備(mini,scr)。")
        return

    # 繪圖設定: 替不同角度指定清楚的顏色
    angle_colors = {'0': "#5FDD99", '30': '#1f77b4', '60': '#d62728'}  # 黑、藍、紅
    angle_labels = {'0': '0 deg', '30': '30 deg', '60': '60 deg'}
    
    
    line_styles = {'0': '-', '30': '-.', '60': '--'}
    line_widths = {'0': 1.8, '30': 2.0, '60': 2.0}

    # 2. 為每個群組繪製疊圖
    for (color, device), angle_files in groups.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        
        has_data = False
        # 確保按照 0 -> 30 -> 60 的順序畫圖
        for angle in ['0', '30', '60']:
            if angle in angle_files:
                filepath = angle_files[angle]
                wls, vals, _ = read_spectrum(filepath)
                
                if wls and vals:
                    # 【核心】將光譜數值歸一化 (除以最大值)，讓峰值都對齊在 1.0
                    vals_np = np.array(vals)
                    max_val = np.max(vals_np)
                    if max_val > 0:
                        vals_norm = vals_np / max_val
                        
                        ax.plot(wls, vals_norm, 
                                color=angle_colors[angle], 
                                linestyle=line_styles[angle], 
                                linewidth=line_widths[angle], 
                                label=angle_labels[angle])
                        has_data = True

        if has_data:
            device_name = "Mini-LED" if device == "mini" else "Screen"
            ax.set_title(f"Normalized Spectra: {color.capitalize()} Light on {device_name}")
            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("Normalized Intensity (a.u.)")
            ax.set_xlim(515, 565)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # 儲存圖片
            out_filename = f"{color}_{device}_overlay_515-565nm.png"
            out_path = os.path.join(out_dir, out_filename)
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            print(f"已儲存疊圖: {out_path}")
            
        plt.close(fig)

if __name__ == "__main__":
    main()