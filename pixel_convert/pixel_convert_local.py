# coding:utf-8
import os
import datetime as dt
import cv2
import pixel_isdf
import numpy as np  # 画像の情報が収められたarrayをいじるためにnumpyを用いる。
from matplotlib import pyplot as plt  # 画像の表示に用いる。
img_path = 'serval_l.jpg'
img_def = cv2.imread(img_path)

pal_num = 8  # 色数
dot_size = 4  # ドットサイズ
blur_lv = 0  # 平滑化
erode_lv = 0  # 輪郭線拡張
dilate_lv = 0  # 輪郭線拡張
img_res, img_res_list, colors = pixel_isdf.make_dot2(img_path, pal_num, dot_size, blur_lv, erode_lv, dilate_lv)
#img_res_list = pixel.make_threshold(img_path)
#oldres, oldcolors = pixel.make_dot(img_path, pal_num, dot_size)
# img_res_list.append(oldres)
i = 0
for img in img_res_list:
    cv2.namedWindow(str(i))
    cv2.imshow(str(i), img)
    i += 1

cv2.imshow("output", img_res)

cv2.imwrite("output.png", img_res_list[-1])

# 水平連結
#imgAdd = cv2.hconcat(img_res_list)
# Window_name = 'serval'# + str(item)
# ウィンドウの作成 (ウィンドウ名、ウィンドウの表示形式)
#cv2.namedWindow(Window_name, cv2.WINDOW_AUTOSIZE)

# ウィンドウに画像表示 (ウィンドウ名、表示する画像の配列)
#cv2.imshow(Window_name, imgAdd)

# キーが押されるまで画像を表示したままにする
# 0→無限、0以上→指定ミリ秒表示
cv2.waitKey(0)

#ウィンドウの破棄 (ウィンドウ名)
# cv2.destroyWindow(Window_name)
cv2.destroyAllWindows()


# 定数定義
#ORG_WINDOW_NAME = "org"
#GRAY_WINDOW_NAME = "gray"
#CANNY_WINDOW_NAME = "canny"

#ORG_FILE_NAME = "serval.jpg"
#GRAY_FILE_NAME = "serval.png"
#CANNY_FILE_NAME = "serval.png"

# 元の画像を読み込む
#org_img = cv2.imread(ORG_FILE_NAME, cv2.IMREAD_UNCHANGED)
# グレースケールに変換
#gray_img = cv2.imread(ORG_FILE_NAME, cv2.IMREAD_GRAYSCALE)
# エッジ抽出
#canny_img = cv2.Canny(gray_img, 50, 110)

# ウィンドウに表示
# cv2.namedWindow(ORG_WINDOW_NAME)
# cv2.namedWindow(GRAY_WINDOW_NAME)
# cv2.namedWindow(CANNY_WINDOW_NAME)

#cv2.imshow(ORG_WINDOW_NAME, org_img)
#cv2.imshow(GRAY_WINDOW_NAME, gray_img)
#cv2.imshow(CANNY_WINDOW_NAME, canny_img)

#    #plt.imshow(image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
