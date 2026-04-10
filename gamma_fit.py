"""
This script is used to determine the gamma fit from camera taken images. 
"""

import os 
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Setting
# =========================
IMG_DIR = r"C:\Users\zengr\Desktop\school\Display\Project6_Luminance_Gamma"
IMG_GLOB = "gray_*.png"

# Center ROI
Y1, Y2 = 1150, 1550
X1, X2 = 1680, 2280

OUT_DIR = os.path.join(IMG_DIR,"gamma_out")
os.makedirs(OUT_DIR,exist_ok = True)

# =========================
# Helpers
# =========================
def extract_gray_level(fname:str) -> int:
    """
    gray_031.png -> 31
    """
    base = os.path.basename(fname)            # gray_031.png
    name,_ = os.path.splitext(base)           # gray_031
    gray_level = int(name.split("_")[-1])    # 31
    return gray_level
    
def load_rgb(path:str) -> np.ndarray: 
    """
    OpenCV read BGR -> convert RGB
    """
    bgr = cv2.imread(path,cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    rgb = rgb.astype(np.float32)
    return rgb
    
def rgb_to_luminance_Y(rgb:np.ndarray) -> np.ndarray:
    """
    RGB -> Luminance Y (relative, not cd/m^2)
    """
    R = rgb[..., 0]
    G = rgb[..., 1]
    B = rgb[..., 2]
    Y = 0.2126 * R + 0.7152 * G + 0.0722 * B
    return Y
    
def roi_mean (arr:np.ndarray) -> float:
    """
    ROI mean
    """
    roi = arr[Y1:Y2, X1:X2]
    return float(np.mean(roi))
    
# =========================
# Main
# =========================
def main():
    paths = sorted(glob.glob(os.path.join(IMG_DIR, IMG_GLOB)))
    if not paths:
        raise RuntimeError("No images found")
        
    gray_levels = []
    luminances = []
    
    for p in paths:
        gray_level = extract_gray_level(p)
        rgb = load_rgb(p)
        Y = rgb_to_luminance_Y(rgb)
        mean_Y = roi_mean(Y)
        
        gray_levels.append(gray_level)
        luminances.append(mean_Y)
        
        print(f"gray level = {gray_level:3d}, ROI mean luminance = {mean_Y:.2f}")
        
    gray_levels = np.array(gray_levels, dtype = np.float32)
    luminances = np.array(luminances,dtype = np.float32)
    
    # -------------------------
    # Luminance Normalization 
    # -------------------------
    black = luminances[gray_levels == 0][0]
    white = luminances[gray_levels == 255][0]
    
    L_norm = (luminances - black)/(white - black)
    delta_L = luminances - black
    delta_V = gray_levels - 0
    
    # Remove gray level 0 from log Gamma fitting 
    mask = (gray_levels > 0)
    x_fit = delta_V[mask]
    y_fit = delta_L[mask]

    
    # Linear fitting log(y) = log(k) + gamma*log(x)
    logx = np.log(x_fit)
    logy = np.log(y_fit)
    b, a = np.polyfit(logx,logy, 1)  # slope=b=gamma, intercept=a=log(k)
    gamma = float(b)
    k = float(np.exp(a))
    y_fitted = gamma*logx + float(a)
 
    # Linear fitting by normalized gray level and luminances
    # x = gray_levels/255.0
    # x_fit = x[mask]
    # y_fit = L_norm[mask]
    # logx = np.log(x_fit)
    # logy = np.log(y_fit)
    # b,a = np. polyfit(logx,logy,1)   # slope=b=gamma, intercept=a=log(k) 
    # gamma = float(b)
    # k = float(np.exp(a))    
    
    print(f"\nFitted gamma = {gamma:.3f}")
    print(f"\nFitted k = {k:.3f}")
    
    
    # Plot & save
    # Plot 1: Normalized Luminance vs gray level 
    plt.figure()
    plt.plot(gray_levels, L_norm, marker = "o")
    fig_path_1 = os.path.join(OUT_DIR, "normalized_illuminance_vs_gray.png")
    plt.xlabel("Gray Level V")
    plt.ylabel("ROI Mean Normalized Luminance(relative)")
    plt.title("ROI Relative Luminance vs. Gray Level")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fig_path_1, dpi=200)
    plt.close()
    print(f"Saved plot: {fig_path_1}")
    
    # Plot 2: log-log fit
    plt.figure()
    plt.plot(logx, logy, label = "Measured Data")
    plt.plot(logx, y_fitted, "-", label=f"Fit: gamma = {gamma:.3f}, k = {k:.3f}")
    fig_path_2 = os.path.join(OUT_DIR, "gamma_fitted.png")
    plt.xlabel("log(gray)")
    plt.ylabel("log(Delta Luminance)")
    plt.title("log-log plot of luminance vs. gray level")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path_2, dpi=200)
    plt.close()
    print(f"Saved plot: {fig_path_2}")
    
    
    print(f"\nAll outputs in: {OUT_DIR}")


       
if __name__ == "__main__":
    main()

    
    
    