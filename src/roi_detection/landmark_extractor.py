"""
RoI領域の抽出
"""
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import cv2
from . import skintone_detector as sd

def FaceAreaRoI(df,filepath,skinpath,show=False):
    """
    openfaceのlandmarkを使って，顔領域を選択し平均化されたRGBを返す
    
    Parameters
    ------
    df: Dataframe 
        OpenFaceから出力したランドマーク
    filepath： str
        動画ファイル，あるいは画像フォルダの絶対パス
    skinpath : str, optional
        肌色検出の閾値が補間されているnpyフォルダの絶対パス
    show : bool,optional
        肌色検出の結果をopencvで出力する

	Returns
	-------
    rgb_components : array
        1フレームごとに平均化されたRGB信号
    """

    # Import landmark 
    pix_x_frames = df.loc[:, df.columns.str.contains('x_')].values.astype(np.int)
    pix_y_frames = df.loc[:, df.columns.str.contains('y_')].values.astype(np.int)

    # 動画ファイルor画像フォルダ一覧を読み取り
    if os.path.isfile(filepath):# 動画ファイルの場合
        cap = cv2.VideoCapture(filepath)
        format_video = True
        total_frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    else:# 画像フォルダの場合
        files = []
        for filename in os.listdir(filepath):
            if os.path.isfile(os.path.join(filepath, filename)): #ファイルのみ取得
                files.append(filename)
        total_frame_num = len(files)
        format_video = False

    # loop from first to last frame
    for i in range(total_frame_num):
        print("Frame: {}/{}".format(i,total_frame_num))

        # 1フレーム分のLandmarkを読み取り
        pix_x = pix_x_frames[i,:].reshape(-1, 1)
        pix_y = pix_y_frames[i,:].reshape(-1, 1)

        # 1フレーム分の画像を読み取り
        if format_video:
            ret, frame = cap.read()
        else:
            frame = cv2.imread(os.path.join(filepath, files[i]))
            
        # FaceMask by features point
        mask = RoIDetection(frame,pix_x,pix_y)
        face_img = cv2.bitwise_and(frame, frame, mask=mask)
        if show:
            cv2.imshow("RoI Detect", face_img)

        # skin area detection HSV & YCbCr
        if skinpath is not None:
            skin_mask = sd.SkinDetectTrack(face_img, skinpath)
            mask = cv2.bitwise_and(mask, skin_mask, skin_mask)
        else:
            skin_mask = sd.SkinDetect(face_img)
            mask = cv2.bitwise_and(mask, skin_mask, skin_mask)
        if show:
                cv2.imshow("Skin Mask", skin_mask)
        
        # merge the mask image
        # average bgr components
        mask_img = cv2.bitwise_and(frame, frame, mask=mask)
        ave_rgb = np.array(cv2.mean(frame, mask=mask)[::-1][1:]).reshape(1,-1)

        if i == 0:
            rgb_components = ave_rgb
        else:   
            rgb_components = np.concatenate([rgb_components, ave_rgb], axis=0)

        if show:
            cv2.imshow("Mask Img", mask_img)
        i = i + 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    return rgb_components

def MouseRoI(videopath,skinpath=None,start=0,fps=0):
    """
    マウスで選択したROI領域にて，平均化されたRGBを返す
    
    Parameters
    ------
    videoepath： str
        動画ファイルの絶対パス
    skinpath : str, optional
        肌色検出の閾値が補間されているnpyフォルダの絶対パス

	Returns
	-------
    rgb_components : array
        1フレームごとに平均化されたRGB信号
    """
    startframe = start*fps
    # 動画ファイルを読み取り
    cap = cv2.VideoCapture(videopath)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(startframe)) # フレーム初期化
    _,image = cap.read()

    # マウスからROIを選択
    roi = cv2.selectROI(image, False) # x,y,w,h
    cv2.destroyAllWindows()
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # フレーム初期化

    # loop from first to last frame
    for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        print("Frame: {}/{}".format(i,int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
        # 1フレーム分の画像を読み取り
        _,frame = cap.read()
        
        # 肌色検出を実行
        if skinpath is not None:
            skin_frame = sd.SkinDetectTrack(frame,skinpath)
        else:
            skin_frame = sd.SkinDetect(frame)
        
        # Merge処理
        mask = cv2.bitwise_and(frame, frame, mask=skin_frame)
        
        # 画像からRoI領域を切り取り
        img_roi = frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
        
        # RoI領域のRGB成分をそれぞれ平均化
        ave_rgb_roi = AveragedRGB(img_roi)
    
        # データ更新
        if i == 0:
            rgb_components = ave_rgb_roi
        else:   
            rgb_components = np.concatenate([rgb_components, ave_rgb_roi], axis=0)
        
        # RoI領域を画像で出力
        cv2.rectangle(frame,(roi[0],roi[1]),(roi[2]+roi[0],roi[3]+roi[1]), (255, 0, 0),thickness=3)
        cv2.imshow("frame", frame)
        cv2.imshow("mask", mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    return rgb_components


def RoIDetection(frame,pix_x,pix_y):
    height, width = frame.shape[:2]
    # roi segmentation
    landmarks = np.concatenate([pix_x, pix_y],axis=1)
    white_img = np.zeros((int(height),int(width)),np.uint8)
    # face
    points = np.concatenate([landmarks[:17,:],landmarks[17:27,:][::-1,:]],axis=0)
    face_mask = cv2.fillConvexPoly(white_img, points = points, color=(255, 255, 255))
    # mouse & eye
    white_img = np.zeros((int(height),int(width)),np.uint8)
    # mask mouse
    cv2.fillConvexPoly(white_img, points = landmarks[48:60,:], color=(255, 255, 255))
    # mask eye
    cv2.fillConvexPoly(white_img, points = landmarks[36:42,:], color=(255, 255, 255))
    outlier_mask = cv2.fillConvexPoly(white_img, points = landmarks[42:48,:], color=(255, 255, 255))
    # merge mask
    roi_mask = cv2.bitwise_xor(face_mask,outlier_mask)
    return roi_mask

def AveragedRGB(roi):
    """
    RoI領域のRGB信号を平均化して返す
    """
    B_value, G_value, R_value = roi.T
    rgb_component = np.array([[np.mean(R_value), np.mean(G_value),np.mean(B_value)]])
    return rgb_component

def ExportRGBComponents(df,cap,fpath):
    rgb_components = MultipleRoI(df,cap)
    columnslist = []
    for i in range(11):
        columnslist.append("camera{}_B".format(i+1))
        columnslist.append("camera{}_G".format(i+1))
        columnslist.append("camera{}_R".format(i+1))
    df = pd.DataFrame(rgb_components,columns=columnslist)
    df.to_csv(fpath)
    print("########################\n")
    print("ExportData:\n{}".format(fpath))
    print("########################\n")

if __name__ == "__main__":
    #動画の読み込み
    cap = cv2.VideoCapture(vpath)

    #Openfaceで取得したLandMark
    df = pd.read_csv(landmark_data,header = 0).rename(columns=lambda x: x.replace(' ', ''))
    print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(df.shape)

    data = FaceAreaRoI(df,cap)
    np.savetxt("./result/rgb_ucomp2_faceroi.csv",data,delimiter=",")