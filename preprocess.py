import sys
import re
import os
import shutil
import scipy.io as io
import math
import csv
import numpy as np
from math import *
from scipy import signal
import scipy.io as scio
from scipy import interpolate
from scipy import signal
import cv2 as cv
import os
import pandas as pd
from multiprocessing import Pool

def getValue(img, patch_sz = 5, channels='RGB'):
    Value = []
    # if c == 1:
    #     channels == 'NIR'
    if channels == 'NIR':
        img = img[:, :, np.newaxis]
    elif channels == 'YUV':
        img = cv.cvtColor(img.astype(np.float32), cv.COLOR_RGB2YUV)
    # print(img.shape)
    h, w, c = img.shape
    w_num = int(w / patch_sz)
    h_num = int(h / patch_sz)
    for w_index in range(w_num):
        for h_index in range(h_num):
            temp = img[h_index * patch_sz: (h_index + 1) * patch_sz, w_index * patch_sz:(w_index + 1) * patch_sz, :] # 16, 16, 3
            temp1 = np.nanmean(np.nanmean(temp, axis=0), axis=0)# 3
            Value.append(temp1) 
    return np.array(Value) # 196, 3

def mySTMap(img_dir, c='RGB'):    
    imgs = np.load(img_dir).astype('float32')
    if c == 'NIR':
        imgs = imgs[:, :, :, 0]
    num_frames = imgs.shape[0]
    STMap = []
    for i in range(num_frames):
        img = imgs[i]
        Value = getValue(img, channels = c)
        if np.isnan(Value).any():
            Value[:, :] = 100
        STMap.append(Value) # along the time
    STMap = np.array(STMap) # 900, 196
    # # CHROM
    # for w in range(STMap.shape[1]):
    #     STMap_CSI[:, w, 0] = np.squeeze(POS(STMap_CSI[:, w, :]))
    # Normal
    # T x ROI_C x C
    for c in range(STMap.shape[2]):
        for w in range(STMap.shape[1]):
            STMap[:, w, c] = 255 * ((STMap[:, w, c] - np.nanmin(STMap[:, w, c])) / (
                    0.001 + np.nanmax(STMap[:, w, c]) - np.nanmin(STMap[:, w, c])))
    STMap = np.swapaxes(STMap, 0, 1)
    STMap = np.rint(STMap)
    STMap = np.array(STMap, dtype='uint8')
    return STMap # 900, 196, 3

def single_process(rgb_file, nir_file, save_dir):
    if os.path.exists(save_dir):
        # print(f'{save_dir} exists')
        return
    print(f'start {save_dir} !')
    rgb_arr = mySTMap(rgb_file, 'RGB')
    yuv_arr = mySTMap(rgb_file, 'YUV')
    nir_arr = mySTMap(nir_file, 'NIR')
    MSTmap = np.concatenate([rgb_arr, yuv_arr, nir_arr], axis=2)
    np.save(save_dir,MSTmap)
    print(f'{save_dir} is saved !')
    return

def choose_windows(name='Hamming', N=20):
    # Rect/Hanning/Hamming
    if name == 'Hamming':
        window = np.array([0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1)) for n in range(N)])
    elif name == 'Hanning':
        window = np.array([0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) for n in range(N)])
    elif name == 'Rect':
        window = np.ones(N)
    return window

def CHROM(STMap_CSI):
    LPF = 0.7  # low cutoff frequency(Hz) - specified as 40bpm(~0.667Hz) in reference
    HPF = 2.5  # high cutoff frequency(Hz) - specified as 240bpm(~4.0Hz) in reference
    WinSec = 1.6  # (was a 48 frame window with 30 fps camera)
    NyquistF = 15  # 30fps
    FS = 30 # 30fps
    FN = STMap_CSI.shape[0]
    B, A = signal.butter(3, [LPF/NyquistF, HPF/NyquistF], 'bandpass')
    WinL = int(WinSec * FS)
    if (WinL % 2):  # force even window size for overlap, add of hanning windowed signals
        WinL = WinL + 1
    if WinL <= 18:
        WinL = 20
    NWin = int((FN - WinL / 2) / (WinL / 2))
    S = np.zeros(FN)
    WinS = 0  # Window Start Index
    WinM = WinS + WinL / 2  # Window Middle Index
    WinE = WinS + WinL   # Window End Index
    #T = np.linspace(0, FN, FN)
    BGRNorm = np.zeros((WinL, 3))
    for i in range(NWin):
        #TWin = T[WinS:WinE, :]
        for j in range(3):
            BGRBase = np.nanmean(STMap_CSI[WinS:WinE, j]) # temporal mean
            BGRNorm[:, j] = STMap_CSI[WinS:WinE, j]/(BGRBase+0.0001) - 1
        Xs = 3*BGRNorm[:, 2] - 2*BGRNorm[:, 1]  # 3Rn-2Gn
        Ys = 1.5*BGRNorm[:, 2] + BGRNorm[:, 1] - 1.5*BGRNorm[:, 0]  # 1.5Rn+Gn-1.5Bn

        Xf = signal.filtfilt(B, A, np.squeeze(Xs)) # bandpass filter
        Yf = signal.filtfilt(B, A, np.squeeze(Ys)) # filter

        Alpha = np.nanstd(Xf)/np.nanstd(Yf)
        SWin = Xf - Alpha*Yf
        SWin = choose_windows(name='Hanning', N=WinL)*SWin
        if i == 0:
            S[WinS:WinE] = SWin
            #TX[WinS:WinE] = TWin
        else:
            S[WinS: WinM - 1] = S[WinS: WinM - 1] + SWin[0: int(WinL/2) - 1] # overlap
            S[WinM: WinE] = SWin[int(WinL/2):]
            #TX[WinM: WinE] = TWin[WinL/2 + 1:]
        WinS = int(WinM)
        WinM = int(WinS + WinL / 2)
        WinE = int(WinS + WinL)
    return S

def POS(STMap_CSI):
    LPF = 0.7  # low cutoff frequency(Hz) - specified as 40bpm(~0.667Hz) in reference
    HPF = 2.5  # high cutoff frequency(Hz) - specified as 240bpm(~4.0Hz) in reference
    WinSec = 1.6  # (was a 48 frame window with 30 fps camera)
    NyquistF = 15  # 30fps
    FS = 30  # 30fps
    N = STMap_CSI.shape[0]
    l = int(WinSec * FS) # 48
    H = np.zeros(N)
    Cn = np.zeros((3, l))
    P = np.array([[0, 1, -1], [-2, 1, 1]])
    for n in range(N-1):
        m = n - l
        if m >= 0:
            Cn[0, :] = STMap_CSI[m:n, 2]/np.nanmean(STMap_CSI[m:n, 2])
            Cn[1, :] = STMap_CSI[m:n, 1]/np.nanmean(STMap_CSI[m:n, 1])
            Cn[2, :] = STMap_CSI[m:n, 0]/np.nanmean(STMap_CSI[m:n, 0])
            # temporal normalization
            
            S = np.dot(P, Cn)
            h = S[0, :] + ((np.nanstd(S[0, :])/np.nanstd(S[1, :]))*S[1, :])
            H[m: n] = H[m: n] + (h - np.nanmean(h))
    return H

def newSTMap(path, save_path, method='POS'): # manually modification
    videos = np.load(path)
    # print(videos.shape)
    STMap = []
    total_time = videos.shape[0]
    for t in range(total_time):
        pic = videos[t]
        # if method == 'NIR':
        #     pic = cv.cvtColor(pic, cv.COLOR_RGB2BGR)
        # pic = cv.resize(pic, (224, 224))
        # now_path = os.path.join(imglist_root, imgPath_sub)
        # img = cv2.imread(now_path)
        if method == 'YUV':
            Value = getValue(pic, channels="YUV")   
        else: 
            Value = getValue(pic)
            
        if np.isnan(Value).any():
            Value[:, :] = 100
        STMap.append(Value)
    STMap = np.array(STMap) # 900, 25, 3
    STMap_CSI = STMap.copy()
    if method == 'CHROM':
        for w in range(STMap.shape[1]):
            STMap_CSI[:, w, 2] = np.squeeze(CHROM(STMap_CSI[:, w, :]))
    elif method == 'POS':
        for w in range(STMap.shape[1]):
            STMap_CSI[:, w, 2] = np.squeeze(POS(STMap_CSI[:, w, :]))
    # elif method == 'YUV'
    for c in range(STMap.shape[2]):
        for w in range(STMap.shape[1]):
            STMap_CSI[:, w, c] = 255 * ((STMap_CSI[:, w, c] - np.nanmin(STMap_CSI[:, w, c])) / (
                    0.001 + np.nanmax(STMap_CSI[:, w, c]) - np.nanmin(STMap_CSI[:, w, c])))
    STMap_CSI = np.swapaxes(STMap_CSI, 0, 1)
    STMap_CSI = np.rint(STMap_CSI)
    STMap_CSI = np.array(STMap_CSI, dtype='uint8')
    save_path = save_path.replace('npy', 'png')
    cv.imwrite(save_path, STMap_CSI)
    print(f"{save_path} has been saved")
    return 

if __name__ == '__main__':
    root = f'/data/PreprocessedData/UBFC_FULL_STMap/POS'    
    df = pd.read_csv('data/PreprocessedData/DataFileLists/UBFC_RAW_RAW_0.0_1.0.csv', index_col = 0)
    
    # file_paths = df['input_files'].tolist()
    # save_dirs = [os.path.join(root, os.path.basename(i)) for i in file_paths]
    
    # print(file_paths)
    
    # for path, save_dir in zip(file_paths, save_dirs):
    #     newSTMap(path, save_dir)
        
    
    with Pool(64) as p:
        file_paths = df['input_files'].tolist()
        save_dirs = [os.path.join(root, os.path.basename(i)) for i in file_paths]
        p.starmap(newSTMap, zip(file_paths, save_dirs))