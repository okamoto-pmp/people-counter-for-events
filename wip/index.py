import cv2
import numpy as np
import argparse
import os
import sys

def main(args):
    # 動画の読み込み
    cap = cv2.VideoCapture(args["input"])
    
    if not cap.isOpened():
        print(f"Error: Cannot open video file {args['input']}")
        return
    
    # 動画の情報を取得
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video info: {width}x{height} @ {fps} FPS")
    
    # 動画保存の設定
    if args["output"]:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        video_writer = cv2.VideoWriter(args["output"], fourcc, fps, (width, height), True)
        print(f"Output video will be saved to: {args['output']}")
    else:
        video_writer = None
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("End of video or failed to read frame")
            break
        
        frame_count += 1
        
        # フレームに情報を描画
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 動画の出力
        if video_writer is not None:
            video_writer.write(frame)
        
        # GUIで表示する場合
        if args.get("gui", False):
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord("q"):
                break
    
    # リソースの解放
    cap.release()
    if video_writer is not None:
        video_writer.release()
        print(f"Video saved successfully: {args['output']}")
    
    if args.get("gui", False):
        cv2.destroyAllWindows()
    
    print(f"Processed {frame_count} frames")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-i", "--input", required=True, type=str,
                          help="path to input video file")
    argparser.add_argument("-o", "--output", type=str,
                          help="path to output video file")
    argparser.add_argument("-g", "--gui", action="store_true",
                          help="whether showing gui")
    
    args = vars(argparser.parse_args())
    main(args)