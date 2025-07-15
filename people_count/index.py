import glob
import os
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import argparse
import cv2
import numpy as np
from typing import Tuple

def main(args):
    print("Starting people count...")
    video_stream = VideoStreamManager(args.get("input", 0))
    gui_show = not args.get("no_gui", False)  # デフォルトでGUIを表示

    try:
        while True:
            result, frame = video_stream.read()
            if not result:
                break
            
            if gui_show:
                # show the output frame
                cv2.imshow("Frame", frame)
                key = cv2.waitKey(1) & 0xFF

                # if the `q` key was pressed, break from the loop
                if key == ord("q"):
                    print("Exiting...")
                    break
    finally:
        # ウィンドウを適切に破棄
        if gui_show:
            cv2.destroyAllWindows()
        video_stream.release()

class VideoStreamManager:
    def __init__(self, input: str|int) -> None:
        self.videostream = cv2.VideoCapture(input)
        self.input = input
        self.videolength = int(self.videostream.get(cv2.CAP_PROP_FRAME_COUNT))
        self.videolength = -2 if self.videolength == 0 or self.videolength == -1 else self.videolength
        
        # ビデオキャプチャが開けない場合のエラーハンドリング
        if not self.videostream.isOpened():
            raise ValueError(f"Cannot open video source: {input}")
    
    def read(self) -> Tuple[bool, np.ndarray]:
        '''
        戻り値: 読み込みの成否, frame
        '''
        retval, frame = self.videostream.read()

        # ビデオファイルの終わりに来た際は終了する
        if int(self.videostream.get(cv2.CAP_PROP_POS_FRAMES)) == self.videolength:
            return False, None
        
        # 接続が切れたとき
        if not retval:
            return False, None
        
        return True, frame
    
    def release(self):
        '''ビデオストリームを解放'''
        if self.videostream is not None:
            self.videostream.release()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-a", "--area", required=False, help="video area to count people (sx, sy, gx, gy)")
    argparser.add_argument("-i", "--input", type=str, help="path to optional input video file")
    argparser.add_argument("--no-gui", action="store_true", help="disable GUI display")

    args = vars(argparser.parse_args())
    print(args)
    main(args)