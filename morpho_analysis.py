# 形態解析.
# 2022-06-29

# virtualenv -p python3.9 pyenv39
# source [venv]/bin/activate
# pip install numpy pandas matplotlib opencv-python scikit-image

# ライブラリ
import os, sys, math
import cv2
import csv
import glob
import re

import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from PIL import Image

fpath = sorted(glob.glob("./herbarium_raw/*"))
fpath = list(filter(lambda x: x.endswith('JPG'), fpath))

for j in range(len(fpath)):

   fname = fpath[j]
   fname_base = re.sub("_th[0-9]+", "", fname)
   
   fname_bi = fname_base.replace("RAW", "BI")
   fname_bi = fname_bi.replace("herbarium_raw", "herbarium_bi")
   
   fname_con = fname_base.replace("RAW", "CON")
   fname_con = fname_con.replace("herbarium_raw", "herbarium_con")
   
   fname_csv = fname_base.replace("_RAW.JPG", ".csv")
   fname_csv = fname_csv.replace("herbarium_raw", "herbarium_csv")
   
   fname_base = re.sub(".+/", "", fname_base)
   sp = re.sub("_.+", "", fname_base)
   threshold = int(re.sub(".+th|_RAW.JPG", "", fname))
   
   # 画像の輪郭線を抽出する. ------------------------------------------------------
   
   # 元画像
   img = cv2.imread(fname) 
   
   # 白黒に変換
   img_edit = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
   
   # 2値化
   ret, img_edit = cv2.threshold(img_edit, threshold, 255, cv2.THRESH_BINARY) 
   
   # 反転
   img_edit = cv2.bitwise_not(img_edit) 
   
   # 画像の保存
   im = Image.fromarray(cv2.cvtColor(img_edit, cv2.COLOR_BGR2RGB))
   im.save(fname_bi)
   
   area = cv2.countNonZero(img_edit) # 面積
   
   contours, hierarchy = cv2.findContours(img_edit, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # 境界抽出
   
   # 小さな輪郭は削除, ワカメの時は 5000 にします.
   contours = list(filter(lambda x: cv2.contourArea(x) >= 5, contours)) 
   
   # 画像に輪郭を描画
   line_color = (0, 255, 0)
   thickness = 2
   
   img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   img = cv2.drawContours(img, contours, -1, line_color, thickness)
   
   # 画像を保存
   im = Image.fromarray(img)
   im.save(fname_con)
   
   arclen = 0
   for i in range(len(contours)):
      arclen = arclen + cv2.arcLength(contours[i], True)

   data = [sp,arclen,area]
   
   f = open(fname_csv, 'w')
   writer = csv.writer(f)
   writer.writerow(data)
   f.close()
   

