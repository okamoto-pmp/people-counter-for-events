import glob
import os
import time
import dlib
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import argparse
import cv2
import numpy as np
from typing import Tuple
from imutils.video import FPS
from lib.tracker import Tracker
from lib.trackableObject import TrackableObject
from lib import utils

def main(args):
    print("Starting people count...")
    video_stream = VideoStreamManager(args.get("input", 0))
    gui_show = not args.get("no_gui", False)  # デフォルトでGUIを表示

    # line検知指定
    line_y = args.get("line")
    # エリア検知指定
    area = args.get("area")
    if area:
        # 文字列 "x1,y1,x2,y2" を整数のリストに変換
        area = [int(x) for x in area.split(',')]
        if len(area) != 4:
            raise ValueError("Area must be specified as 'x1,y1,x2,y2'")

    # フレームサイズの初期化
    W = None
    H = None

    net = cv2.dnn.readNetFromTensorflow(args["model"], args["prototxt"])

    # Trackerの初期化
    ct = Tracker(
        maxDisappeared=args["max_disappeared"], maxDistance=args["max_distance"])
    trackers = []
    trackableObjects = {}

    # カウンタの初期化
    totalFrames = 0
    totalDown = 0
    totalUp = 0
    totalEnter = 0
    totalLeave = 0
    nowEnter = 0

    # frame skipのための計測
    last_detect_time = time.time()

    # FPSの計測
    fps = FPS().start()

    # Stream clientの初期化（ダミー実装）
    class DummyStreamClient:
        def enter(self):
            pass
        def leave(self):
            pass
    
    stream_client = DummyStreamClient()

    try:
        while True:
            result, frame = video_stream.read()
            if not result:
                break
            if W is None or H is None:
                (H, W) = frame.shape[:2]
            
            # フレームをRGBに変換
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            status = "Waiting"
            rects = []
            status = "Detecting"
            trackers = []

            # 機械学習モデルでの検知
            blob = cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False)
            net.setInput(blob)
            detections = net.forward()

            # 検知結果についてループ
            for i in np.arange(0, detections.shape[2]):
                # 確信度を取得
                confidence = detections[0, 0, i, 2]

                # 確信度が一定値以上のものを取得
                if confidence > args["confidence"]:
                    idx = int(detections[0, 0, i, 1])

                    # if CLASSES[idx] != "person":
                    #     continue

                    # boxの座標取得
                    box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    (lx, ly, rx, ry) = box.astype("int")

                    # 検知された矩形をrectsに追加
                    rects.append((lx, ly, rx, ry))

                    # 類似度ベースのtracking
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(lx, ly, rx, ry)
                    tracker.start_track(rgb, rect)

                    trackers.append(tracker)

            last_detect_time = time.time()
            # 検知した物体ボックスの描画
            for rect in rects:
                cv2.rectangle(frame, rect[0:2], rect[2:4], color=(0, 0, 200))

            # 検知ラインの描画
            if line_y:
                cv2.line(frame, (0, line_y), (W, line_y), (0, 255, 255), 2)

            # 検知エリアの描画
            if area:
                cv2.rectangle(frame, area[0:2], area[2:4], color=(200, 0, 0))

            # Trackerを更新
            objects, deregisters = ct.update(rects)

            nowEnter = 0

            # 領域内で物体が消失したときにtotalLeaveを+1
            if area:
                for (objectID, centroid) in deregisters.items():
                    if utils.is_in_area(area, centroid):
                        totalLeave += 1
                        stream_client.leave()
            # 追跡対象のオブジェクトを取得
            for (objectID, centroid) in objects.items():
                trackable_object = trackableObjects.get(objectID, None)

                if trackable_object is None:
                    trackable_object = TrackableObject(objectID, centroid)

                # 追跡物体のこれまでのy座標のリスト
                xs = [c[0] for c in trackable_object.centroids]
                ys = [c[1] for c in trackable_object.centroids]

                # 領域検知
                if area:
                    if utils.is_in_area(area, centroid):
                        nowEnter += 1

                    # 過去5フレームのx,y座標の平均
                    xmean, ymean = np.mean(xs[:-5]), np.mean(ys[:-5])
                    # 新たに領域内に入った場合、enterイベントを送信する
                    if not trackable_object.enter_counted and utils.is_in_area(area, centroid):
                        totalEnter += 1
                        trackable_object.enter_counted = True
                        stream_client.enter()

                    # 領域から出た場合、leaveイベントを送信する
                    if not trackable_object.leave_counted and not utils.is_in_area(area, centroid) and utils.is_in_area(area, [xmean, ymean]):
                        totalLeave += 1
                        trackable_object.leave_counted = True
                        stream_client.leave()
                direction = centroid[1] - np.mean(ys)

                trackable_object.centroids.append(centroid)
                
                if not trackable_object.counted and line_y:
                    # 検知ラインを上向きにクロスした場合、退出としてカウント
                    if direction < 0 and centroid[1] < line_y and ys[-1] >= line_y:
                        totalUp += 1
                        trackable_object.counted = True
                        stream_client.leave()

                    # 検知ラインを下向きにクロスした場合、入場としてカウント
                    elif direction > 0 and centroid[1] > line_y and ys[-1] <= line_y:
                        totalDown += 1
                        trackable_object.counted = True
                        stream_client.enter()
                trackableObjects[objectID] = trackable_object

                # objectIDの描画
                text = f"ID {objectID}"
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
            # 各種情報を描画
            info = [
                ("InArea", nowEnter),
                ("Enter", totalEnter),
                ("Leave", totalLeave),
                ("Status", status)
            ]

            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if gui_show:
                if line_y:
                    cv2.line(frame, (0, line_y), (W, line_y), (0, 255, 255), 2)
                # 検知エリアの描画
                if area:
                    cv2.rectangle(frame, area[:2], area[2:], (0, 0, 255), 2)
                # show the output frame
                cv2.imshow("Frame", frame)
                key = cv2.waitKey(1) & 0xFF

                # if the `q` key was pressed, break from the loop
                if key == ord("q"):
                    print("Exiting...")
                    break
            totalFrames += 1
            fps.update()
        # FPS情報の表示
        fps.stop()
        print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

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
    argparser.add_argument("-l", "--line", type=int, help="y coordinates for line based detection")
    argparser.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
    argparser.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
    argparser.add_argument("-c", "--confidence", type=float, default=0.4, help="minimum probability to filter weak detections")
    argparser.add_argument("--max-disappeared", type=int, default=30, help="maximum consecutive frames a given object is allowed to be marked as 'disappeared'")
    argparser.add_argument("--max-distance", type=int, default=50, help="maximum distance between centroids to associate an object")
    args = vars(argparser.parse_args())
    # AreaとLine両方を指定した時例外を投げる
    if args.get("area") and args.get("line") is not None:
        raise ValueError("Cannot specify both area and line parameters at the same time.")
    main(args)