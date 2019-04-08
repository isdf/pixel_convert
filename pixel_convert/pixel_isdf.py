# coding:utf-8
import sys
import cv2
from PIL import Image
import numpy as np
import math

n4 = np.array([[0, 1, 0], [1, 1, 1],  [0, 1, 0]], np.uint8)  # 4近傍の定義
n8 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], np.uint8)  # 8近傍の定義


# ガンマ変換
def gamma(img, lv):
    # ルックアップテーブルの生成
    look_up_table = np.ones((256, 1), dtype='uint8') * 0

    for i in range(256):

        look_up_table[i][0] = 255 * pow(float(i) / 255, 1.0 / lv)

    # ガンマ変換後の出力
    res = cv2.LUT(img, look_up_table)

    return res


# 収縮(erosion)
def erode(img, lv, ite):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3 + lv, 3 + lv))
    res = cv2.erode(img, kernel, iterations=ite)
    return res


# 膨張(dilate)
def dilate(img, lv, ite):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3 + lv, 3 + lv))
    res = cv2.dilate(img, kernel, iterations=ite)
    return res


# コントラストs
def contrast(img, a):
    lut = [np.uint8(255.0 / (1 + math.exp(-a * (i - 128.) / 255.))) for i in range(256)]
    res = np.array([lut[value] for value in img.flat], dtype=np.uint8)
    res = res.reshape(img.shape)
    return res


# 平滑化
def blur(img, b):
    res = cv2.bilateralFilter(img, 15, b, 20)
    return res


# シャープネス
def sharp(img, k):
    # シャープ化するためのオペレータ
    shape_operator = np.array([[0, -k, 0],
                               [-k, 1 + 4 * k, -k],
                               [0, -k, 0]])
    # 作成したオペレータを基にシャープ化
    res = cv2.filter2D(img, -1, shape_operator)
    # img_shape = cv2.convertScaleAbs(img_tmp)

    return res


# ゴミ除去
def cleardust(img, size):
    # img, contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    img, contours, hierarchy = cv2.findContours(img,  cv2.RETR_EXTERNAL | cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    new_contours = []
    FILL_COLOR = (255, 255, 255)
    for c in contours:
        s = abs(cv2.contourArea(c))
        if s <= size:
            new_contours.append(c)
    img2 = img.copy()
    return cv2.drawContours(img2, new_contours, -1, FILL_COLOR, -1)

# 大外の輪郭線を消す


def clear_outline(img):
    # 輪郭は黒地の白を取るらしいので反転
    img = 255 - img

    img, contours, hierarchy = cv2.findContours(img,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img = 255 - img
    FILL_COLOR = (255, 255, 255)

    new_contours = []

    for i, c in enumerate(contours):
        new_contours.append(c)

    # img2 = img + 255
    img2 = img.copy()
    # cv2.drawContours(img2, new_contours, -1, FILL_COLOR, -1)
    cv2.polylines(img2, new_contours, True, FILL_COLOR, 5)
    return img2

# 大外の輪郭線


def draw_outline(img):
    # 輪郭は黒地の白を取るらしいので反転
    FILL_COLOR = (0, 0, 0)

    img = 255 - img
    img, contours, hierarchy = cv2.findContours(img,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img = 255 - img

    new_contours = []
    for i, c in enumerate(contours):
        new_contours.append(c)

    # 無地に描く
    img2 = img + 255
    cv2.polylines(img2, new_contours, True, FILL_COLOR, 4)
    return img2


# モルフォロジー変換


def morphology_open(img, lv):
    # 楕円形カーネル
    # cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    # 十字型カーネル
    # cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
    # 矩形カーネル
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3 + lv, 3 + lv))
    res = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return res


# モルフォロジー変換
def morphology_close(img, lv):
    # 楕円形カーネル
    # cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    # 十字型カーネル
    # cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
    # 矩形カーネル
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3 + lv, 3 + lv))
    res = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return res


# クラスタリング
def kmeans(img, c, pal_num):
    # numpy.reshape(a, newshape, order=’C’)
    # 配列のshapeを指定する際に (n, -1) のように-1を指定すると要素数に合わせてn × mの2次元配列となります
    img_cp = img.reshape(-1, c)

    # データ型変換
    img_cp = img_cp.astype(np.float32)

    # kmeans
    # samples : np.float32 型のデータとして与えられ，各特徴ベクトルは一列に保存されていなければなりません．
    # nclusters(K) : 最終的に必要とされるクラスタの数．
    # attempts :
    # 異なる初期ラベリングを使ってアルゴリズムを実行する試行回数を表すフラグ．アルゴリズムは最良のコンパクトさをもたらすラベルを返します．このコンパクトさが出力として返されます．
    # flags : このフラグは重心の初期値を決める方法を指定します．普通は二つのフラグ cv2.KMEANS_PP_CENTERS と
    # cv2.KMEANS_RANDOM_CENTERS が使われます．
    # criteria
    #: 繰り返し処理の終了条件です．この条件が満たされた時，アルゴリズムの繰り返し計算が終了します．
    # 実際は3個のパラメータのtuple ( type, max_iter, epsilon ) として与えられます:
    # 3.a - 終了条件のtype: 以下に示す3つのフラグを持っています:
    # cv2.TERM_CRITERIA_EPS - 指定された精度(epsilon)に到達したら繰り返し計算を終了する．
    # cv2.TERM_CRITERIA_MAX_ITER - 指定された繰り返し回数(max_iter)に到達したら繰り返し計算を終了する．
    # cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER -
    # 上記のどちらかの条件が満たされた時に繰り返し計算を終了する．
    # 3.b - max_iter - 繰り返し計算の最大値を指定するための整数値．
    # 3.c - epsilon - 要求される精度．
    # output
    # compactness : 各点と対応する重心の距離の二乗和．
    # labels : 各要素に与えられたラベル(‘0’, ‘1’ ...)のarray (前チュートリアルにおける ‘code’ )．
    # centers : クラスタの重心のarray．
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(img_cp, pal_num, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    # データ型変換
    # 近似色のリスト（パレット）
    center = center.astype(np.uint8)
    # label.flatten()はインデックス
    result = center[label.flatten()]
    # 配列の変換img.shapeは縦ｘ横ｘRGB(3)が入ってる
    result = result.reshape(img.shape)

    return result, center


# 二値化
def threshold(img, thresh):
    MAXVALUE = 255
    retval, dst = cv2.threshold(img, thresh, MAXVALUE, cv2.THRESH_BINARY)
    return dst


# グレースケール
def grayscale(img):
    res = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # res = cv2.merge((gray, gray, gray))
    return res


# グレースケールからRGB
def gray2rgb(img):
    res = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return res


# 差分＆ネガポジ
def subtract(nowimg, baseimg):
    # 差分
    nega = cv2.subtract(nowimg, baseimg)
    # nega = cv2.add(nega,nega)
    res = 255 - nega
    return res


def make_col(img, pal_num, dotsize):

    res_List = []
    # ファイルオープン
    # res_List.append(img)
    h, w, c = img.shape  # 画像サイズと色数

    # 膨張（輪郭線消去）
    img = dilate(img, 0, 2)
    res_List.append(img)

    # 収縮（輪郭線膨張）
    img = erode(img, 0, 1)
    res_List.append(img)

    # # ガンマ変換
    # img = gamma(img, 0.5)
    # res_List.append(img)

    # # コントラスト
    # img = contrast(img, 7)
    # res_List.append(img)

    # 減色
    img, center = kmeans(img, c, pal_num)
    res_List.append(img)

    # 縮小と拡大
    dot_h = int(h / dotsize)
    dot_w = int(w / dotsize)
    interpolation = cv2.INTER_NEAREST
    small_img = cv2.resize(img, (dot_w, dot_h), interpolation=interpolation)
    res_List.append(small_img)

    small_img = 255 - small_img
    res_List.append(small_img)

    return small_img, res_List


def make_line(img, pal_num, dotsize):
    res_List = []
    h, w, c = img.shape  # 画像サイズと色数

    # シャープネス
    img = sharp(img, 2)
    res_List.append(img)

    # グレースケール
    img = grayscale(img)
    # res_List.append(img)

    # 膨張（輪郭線消去）
    dilation = dilate(img, 0, 2)
    # res_List.append(dilation)

    # 膨張前からの差分＆ネガポジ（線画抽出）
    img = subtract(dilation, img)
    res_List.append(img)

    # 二値化
    img = threshold(img, 150)
    res_List.append(img)

    # ゴミ除去
    img = cleardust(img, 10)
    res_List.append(img)

    # 収縮（輪郭線膨張）
    img = erode(img, 0, 1)
    res_List.append(img)

    # 輪郭線削除
    img = clear_outline(img)
    res_List.append(img)

    # 縮小と拡大
    dot_h = int(h / dotsize)
    dot_w = int(w / dotsize)
    # interpolation = cv2.INTER_NEAREST
    interpolation = cv2.INTER_LINEAR
    img = cv2.resize(img, (dot_w, dot_h), interpolation=interpolation)
    res_List.append(img)

    return img, res_List


def make_outline(img, pal_num, dotsize):
    res_List = []
    h, w, c = img.shape  # 画像サイズと色数

    # シャープネス
    img = sharp(img, 2)
    res_List.append(img)

    # グレースケール
    img = grayscale(img)
    res_List.append(img)

    # 膨張（輪郭線消去）
    dilation = dilate(img, 0, 2)
    res_List.append(dilation)

    # 膨張前からの差分＆ネガポジ（線画抽出）
    img = subtract(dilation, img)
    res_List.append(img)

    # 二値化
    img = threshold(img, 150)
    res_List.append(img)

    # ゴミ除去
    img = cleardust(img, 10)
    res_List.append(img)

    # 輪郭線
    img = draw_outline(img)
    res_List.append(img)

    # # 収縮（輪郭線膨張）
    # img = erode(img, 0, 1)
    # res_List.append(img)

    # 縮小と拡大
    dot_h = int(h / dotsize)
    dot_w = int(w / dotsize)
    interpolation = cv2.INTER_NEAREST
    # interpolation = cv2.INTER_LINEAR
    img = cv2.resize(img, (dot_w, dot_h), interpolation=interpolation)
    res_List.append(img)

    return img, res_List


def mix_col_line(img_col, img_line, img_outline):
    # img_col = 255 - img_col
    img_line = 255 - img_line
    img_outline = 255 - img_outline
    img_line = gray2rgb(img_line)
    img_outline = gray2rgb(img_outline)
    img = cv2.add(img_col, img_line)
    img = cv2.add(img, img_outline)
    img = 255 - img
    return img


def make_dot2(image_path, pal_num, dotsize, blur_lv, erode_lv, dilate_lv):
    res_List = []

    # ファイルオープン
    img = cv2.imread(image_path)

    h, w, c = img.shape  # 画像サイズと色数
    dot_h = int(h / dotsize)
    dot_w = int(w / dotsize)

    img_col, img_col_list = make_col(img, pal_num, dotsize)
    img_line, img_line_list = make_line(img, pal_num, dotsize)
    img_outline, img_outline_list = make_outline(img, pal_num, dotsize)

    # res_List.extend(img_col_list)
    res_List.extend(img_line_list)
    # res_List.extend(img_outline_list)

    img = mix_col_line(img_col, img_line, img_outline)

    # # コントラスト
    # img = contrast(img, 7)
    # res_List.append(img)

    # 拡大
    img = cv2.resize(img, (dot_w * dotsize, dot_h * dotsize), interpolation=cv2.INTER_NEAREST)

    res_List.append(img)

    # 使用されたカラーコード
    colors = []

    return res_List[-1], res_List, colors
