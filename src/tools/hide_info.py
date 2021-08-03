import os
import numpy as np
import pandas as pd
import cv2

def triming(path,outpath):
    # 動画ファイルを読み取り
    cap = cv2.VideoCapture(path)
    _,image = cap.read()

    fps    = cap.get(cv2.CAP_PROP_FPS)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    # 書き出し先　形式はMP4Vを指定
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(outpath,int(fourcc), fps, (int(width), int(height)))

    roi = cv2.selectROI(image, False) # x,y,w,h
    cv2.destroyAllWindows()

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # フレーム初期化

    for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        print("Frame: {}/{}".format(i,int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
        # 1フレーム分の画像を読み取り
        _,frame = cap.read()

        cv2.rectangle(frame,(roi[0],roi[1]),(roi[2]+roi[0],roi[3]+roi[1]), (255, 0, 0),thickness=-1)
        out.write(frame)
        # cv2.imshow('a',frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    out.release()
    cap.release()
    cv2.destroyAllWindows()