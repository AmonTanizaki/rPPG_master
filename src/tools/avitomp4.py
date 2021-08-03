import numpy as np
import cv2

def moviech(path,outpath):
    # 読み込み元　形式はMP4Vを指定
    cap = cv2.VideoCapture(path)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    # 書き出し先　形式はMP4Vを指定
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(outpath,int(fourcc), fps, (int(width), int(height)))

    for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        try:
            ret, frame = cap.read()
            out.write(frame)
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except:
            break

    out.release()
    cap.release()
    cv2.destroyAllWindows()
