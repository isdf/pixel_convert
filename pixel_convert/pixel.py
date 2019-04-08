# coding:utf-8
import sys
import cv2
from PIL import Image
import numpy as np
import math

n4 = np.array([[0, 1, 0], [1, 1, 1],  [0, 1, 0]], np.uint8)  # 4近傍の定義
n8 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], np.uint8)  # 8近傍の定義
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

 # 平滑化


def blur(img, b):
    res = cv2.bilateralFilter(img, 15, b, 20)
    return res

# コントラスト


def contrast(img, a):
    lut = [np.uint8(255.0 / (1 + math.exp(-a * (i - 128.) / 255.))) for i in range(256)]
    res = np.array([lut[value] for value in img.flat], dtype=np.uint8)
    res = res.reshape(img.shape)
    return res

# シャープネス


def sharp(img, k):
    # シャープ化するためのオペレータ
    shape_operator = np.array([[0, -k, 0],
                               [-k, 1 + 4 * k, -k],
                               [0, -k, 0]])
    # 作成したオペレータを基にシャープ化
    res = cv2.filter2D(img, -1, shape_operator)
    #img_shape = cv2.convertScaleAbs(img_tmp)

    return res


# ゴミ除去
def cleardust(img, size):
    #img, contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(img,  cv2.RETR_EXTERNAL | cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    new_contours = []
    FILL_COLOR = (255, 255, 255)
    for c in contours:
        s = abs(cv2.contourArea(c))
        if s <= size:
            new_contours.append(c)
    img2 = img.copy()
    return cv2.drawContours(img2, new_contours, -1, FILL_COLOR, -1)

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
    #res = cv2.merge((gray, gray, gray))
    return res

# グレースケールから


def gray2rgb(img):
    res = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return res

# 差分＆ネガポジ


def subtract(nowimg, baseimg):
    # 差分
    nega = cv2.subtract(nowimg, baseimg)
    #nega = cv2.add(nega,nega)
    res = 255 - nega
    return res


def make_dotc(img, pal_num, dotsize):

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

    return res_List


def make_dot2(image_path, pal_num, dotsize, blur_lv, erode_lv, dilate_lv):
    res_List = []

    # ファイルオープン
    img = cv2.imread(image_path)
    imgc_res_List = make_dotc(img, pal_num, dotsize)
    res_List.extend(imgc_res_List)
    # res_List.append(img)
    h, w, c = img.shape  # 画像サイズと色数

    # シャープネス
    img = sharp(img, 2)
    # コントラスト
    #img = contrast(img,5)
    res_List.append(img)

    # 収縮（輪郭線強調）
    # if (erode_lv > 0):
    #    img = erode(img, erode_lv)
    #    res_List.append(img)

    # 膨張（輪郭線消失）
    # if (dilate_lv > 0):
    #    img = dilate(img, dilate_lv)
    #    res_List.append(img)

    # 平滑化
    # if (blur_lv > 0):
    #    img = blur(img, blur_lv)
    #    res_List.append(img)

    # グレースケール
    img = grayscale(img)
    # res_List.append(img)

    # 顔検出
    #cascade_file = "lbpcascade_animeface.xml"
    #cascade = cv2.CascadeClassifier(cascade_file)
    # faces = cascade.detectMultiScale(img, scaleFactor = 1.1, minNeighbors = 5,
    #faces = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3, minSize=(50, 50))
    # minSize = (24, 24))

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

    #imgray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    #ret,thresh = cv2.threshold(imgray,127,255,0)
    #print (img.shape)
    #print (thresh.shape)

    # モルフォロジー変換（ノイズ除去）
    ##img = morphology_open(img, 1)
    # res_List.append(img)

    # モルフォロジー変換（ノイズ除去）
    #img = morphology_close(img, 0)
    # res_List.append(img)

    # 収縮（輪郭線膨張）
    img = erode(img, 0, 1)
    res_List.append(img)

    # 減色
    ##img , center = kmeans(img, c, pal_num)
    # res_List.append(img)

    # 縮小と拡大
    dot_h = int(h / dotsize)
    dot_w = int(w / dotsize)
    interpolation = cv2.INTER_NEAREST
    small_img = cv2.resize(img, (dot_w, dot_h), interpolation=interpolation)
    res_List.append(small_img)

    small_img = 255 - small_img
    small_img = gray2rgb(small_img)
    small_img = cv2.add(imgc_res_List[-1], small_img)
    small_img = 255 - small_img
    # 輪郭
    #img, contours, hierarchy = cv2.findContours(small_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # res_List.append(img)
    #img = gray2rgb(img)
    #img = cv2.drawContours(img, contours, -1, (0,255,0), 1)
    # res_List.append(img)

    # 拡大
    img = cv2.resize(small_img, (dot_w * dotsize, dot_h * dotsize), interpolation=cv2.INTER_NEAREST)

    # 顔の枠表示
    # for (x, y, w, h) in faces: cv2.rectangle(img, (x, y), (x + w, y + h), (0,
    # 0, 255), 2)
    res_List.append(img)

    # 使用されたカラーコード
    colors = []
    # for res_c in center:
    #    color_code = '#{0:02x}{1:02x}{2:02x}'.format(res_c[2], res_c[1],
    #    res_c[0])
    #    colors.append(color_code)

    return res_List, colors


def make_dot(src, k=3, scale=2, color=True, blur=0, erode=0, alpha=False, to_tw=False):
    # ファイルオープン
    #img_pl = Image.open(src)
    # 画像フォーマット
    # if (img_pl.mode == 'RGBA' or img_pl.mode == 'P') and alpha:
    #    if img_pl.mode != 'RGBA':
    #        img_pl = img_pl.convert('RGBA')
    #    alpha_mode = True
    # elif img_pl.mode != 'RGB' and img_pl.mode != 'L':
    #    img_pl = img_pl.convert('RGB')
    #    alpha_mode = False
    # else:
    #    alpha_mode = False
    alpha_mode = False
    # 配列を作る
    #img = np.asarray(img_pl)
    img = cv2.imread(src)

    #    void cvtColor(const Mat& src, Mat& dst, int code, int dstCn=0)
    # 画像の色空間を変換します．
    # src – 8ビット符号なし整数型，16ビット符号なし整数型（ CV_16UC...  ），または単精度浮動小数型の入力画像
    # dst – src と同じサイズ，同じタイプの出力画像
    # code – 色空間の変換コード．説明を参照してください
    # dstCn – 出力画像のチャンネル数．この値が 0 の場合，チャンネル数は src と code から自動的に求められます
    if color and alpha_mode:
        a = img[:, :, 3]
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        h, w, c = img.shape
    elif color:
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, c = img.shape  # 画像サイズと色数
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        h, w = img.shape
        c = 0

    # 収縮(Erosion)
    if erode == 1:
        n4 = np.array([[0, 1, 0], [1, 1, 1],  [0, 1, 0]], np.uint8)
        img = cv2.erode(img, n4, iterations=1)
    elif erode == 2:
        n8 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], np.uint8)
        img = cv2.erode(img, n8, iterations=1)

    # 平滑化
    if blur:
        img = cv2.bilateralFilter(img, 15, blur, 20)

    # 縮小
    d_h = int(h / scale)
    d_w = int(w / scale)
    # INTER_NEAREST 最近傍補間
    # INTER_LINEAR バイリニア補間（デフォルト）
    # INTER_AREA
    # ピクセル領域の関係を利用したリサンプリング．画像を大幅に縮小する場合は，モアレを避けることができる良い手法です．しかし，画像を拡大する場合は，
    # INTER_NEAREST メソッドと同様になります
    # INTER_CUBIC 4x4 の近傍領域を利用するバイキュービック補間
    # INTER_LANCZOS4 8x8 の近傍領域を利用する Lanczos法の補間
    img = cv2.resize(img, (d_w, d_h), interpolation=cv2.INTER_NEAREST)

    # アルファ値ある場合はアルファ値の縮小
    if alpha_mode:
        a = cv2.resize(a, (d_w, d_h), interpolation=cv2.INTER_NEAREST)
        a = cv2.resize(a, (d_w * scale, d_h * scale), interpolation=cv2.INTER_NEAREST)
        a[a != 0] = 255
        if not 0 in a:
            a[0, 0] = 0

    # numpy.reshape(a, newshape, order=’C’)
    # 配列のshapeを指定する際に (n, -1) のように-1を指定すると要素数に合わせてn × mの2次元配列となります
    if color:
        img_cp = img.reshape(-1, c)
    else:
        img_cp = img.reshape(-1)

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
    ret, label, center = cv2.kmeans(img_cp, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    # データ型変換
    # 近似色のリスト（パレット）
    center = center.astype(np.uint8)
    # label.flatten()はインデックス
    result = center[label.flatten()]
    # 配列の変換img.shapeは縦ｘ横ｘRGB(3)が入ってる
    result = result.reshape(img.shape)
    # ドットサイズ拡大
    result = cv2.resize(result, (d_w * scale, d_h * scale), interpolation=cv2.INTER_NEAREST)

    # α値変化
    if alpha_mode:
        r, g, b = cv2.split(result)
        result = cv2.merge((r, g, b, a))
    elif to_tw:  # ツイッター用左上透過
        r, g, b = cv2.split(result)
        a = np.ones(r.shape, dtype=np.uint8) * 255
        a[0, 0] = 0
        result = cv2.merge((r, g, b, a))

    # 使用されたカラーコード
    colors = []
    for res_c in center:
        color_code = '#{0:02x}{1:02x}{2:02x}'.format(res_c[2], res_c[1], res_c[0])
        colors.append(color_code)

    return result, colors
