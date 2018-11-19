# -*- coding:utf-8 -*-

from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from werkzeug import secure_filename
import os
import detect
import cv2
import io

app = Flask(__name__)

app.config['DEBUG'] = True
# 投稿画像の保存先
UPLOAD_FOLDER = './static/images'
RESULT_FOLDER = './static/detected_img'
CUT_FOLDER = './static/cut_img'

# ルーティング。/にアクセス時
@app.route('/')
def index():
    return render_template('index_root.html')

# 画像投稿時のアクション
@app.route('/post', methods=['GET','POST'])
def post():
    if request.method == 'POST':
        if not request.files['file'].filename == '':
      # アップロードされたファイルを保存
            none_file = 0
            img_file = request.files['file']
            f = img_file.stream.read()
            bin_data = io.BytesIO(f)
            file_bytes = np.asarray(bytearray(bin_data.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            raw_img_url = os.path.join(UPLOAD_FOLDER, "raw_" + secure_filename(img_file.filename))
            cv2.imwrite(raw_img_url, img)
      # detect.pyへアップロードされた画像を渡す
            results, dstimg = detect.evaluation(raw_img_url)
            if dstimg != []:
                detected_img_url = os.path.join(RESULT_FOLDER, "detected_" + secure_filename(img_file.filename))
                cv2.imwrite(detected_img_url, dstimg)

            bottles = [
            "三ツ矢サイダー",
            "カルピスウォーター",
            "綾鷹"
            ]

            count = 0
            result_list = []

            if results != 0 :
                for result in results:
                    mydict = {}
                    ans, prob, cut_img = result
                    cut_img_url = os.path.join(CUT_FOLDER, str(count) + "_cut_" + secure_filename(img_file.filename))
                    cv2.imwrite(cut_img_url, cut_img)
                    ans = bottles[ans]
                    prob = [str(i) for i in prob]
                    mydict["ans"] = ans
                    mydict["prob"] = prob
                    mydict["cut_img_url"] = cut_img_url
                    result_list.append(mydict)
                    count += 1
            else :
                detected_img_url = raw_img_url

        else:
            none_file = 1
            result_list = []
            raw_img_url = ''
            detected_img_url = ''

        return render_template('index_bottle_ui.html',result_list = result_list, image=raw_img_url, dstimg=detected_img_url, none_file=none_file, bottles=bottles)
    else:
    # エラーなどでリダイレクトしたい場合
        return redirect(url_for('bottle_ui'))

if __name__ == "__main__":
    app.run()
