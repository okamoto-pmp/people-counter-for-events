import cv2
import argparse
import sys
import os
import json
from datetime import datetime

from person_detector import PersonDetector
from person_identifier import PersonIdentifier
from visualizer import Visualizer
from logger import PersonCounterLogger

class PeopleCounterApp:
    """人物検出・識別・描画を統合したアプリケーション"""
    
    def __init__(self, model_path: str, prototxt_path: str, confidence_threshold: float = 0.4, 
                 similarity_threshold: float = 0.7, enable_logging: bool = True):
        """
        Args:
            model_path: 検出モデルのパス
            prototxt_path: prototxtファイルのパス
            confidence_threshold: 検出信頼度の閾値
            similarity_threshold: 識別類似度の閾値
            enable_logging: ログ機能を有効にするかどうか
        """
        self.detector = PersonDetector(model_path, prototxt_path, confidence_threshold)
        self.identifier = PersonIdentifier(similarity_threshold)
        self.visualizer = Visualizer()
        
        # ログ機能の初期化
        if enable_logging:
            self.logger = PersonCounterLogger()
            self.identifier.set_logger(self.logger)
        else:
            self.logger = None
        
    def process_frame(self, frame, frame_num, area=None, line_y=None):
        """
        フレームを処理（検出→識別→描画）
        
        Args:
            frame: 入力フレーム
            frame_num: フレーム番号
            area: 検出エリア (x1, y1, x2, y2) or None
            line_y: 検出ライン or None
            
        Returns:
            処理後のフレーム
        """
        # 1. 人物検出
        person_boxes = self.detector.detect_persons(frame)
        
        # 検出ログ
        if self.logger:
            self.logger.log_person_detection(frame_num, person_boxes)
        
        # 2. 人物識別
        person_ids, similarities = self.identifier.identify_persons(frame, person_boxes, frame_num)
        
        # 3. 描画
        output_frame = frame.copy()
        
        # エリアまたはラインを描画
        if area:
            output_frame = self.visualizer.draw_detection_area(output_frame, area)
        if line_y:
            output_frame = self.visualizer.draw_detection_line(output_frame, line_y)
        
        # バウンディングボックスとIDを描画
        output_frame = self.visualizer.draw_bounding_boxes(output_frame, person_boxes, person_ids)
        
        # 統計情報を描画
        stats = self.identifier.get_statistics()
        info = {
            'Detected': len(person_boxes),
            'Unique': stats['unique_persons'],
            'Total Features': stats['total_features'],
            'Frame': frame_num
        }
        output_frame = self.visualizer.draw_info(output_frame, info)
        
        # フレーム処理ログ
        if self.logger:
            self.logger.log_frame_process(frame_num, len(person_boxes), stats['unique_persons'])
        
        return output_frame, person_boxes, person_ids, similarities

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='People Counter with ReID and Logging')
    parser.add_argument('-m', '--model', required=True, help='Path to detection model')
    parser.add_argument('-p', '--prototxt', required=True, help='Path to prototxt file')
    parser.add_argument('-i', '--input', help='Input video file (default: webcam)')
    parser.add_argument('-c', '--confidence', type=float, default=0.4, help='Detection confidence threshold')
    parser.add_argument('-s', '--similarity', type=float, default=0.7, help='Person identification similarity threshold')
    parser.add_argument('-a', '--area', help='Detection area as x1,y1,x2,y2')
    parser.add_argument('-l', '--line', type=int, help='Detection line Y coordinate')
    parser.add_argument('-o', '--output', help='Output video file path')
    parser.add_argument('--json-output', help='Output JSON results file path')
    parser.add_argument('--no-gui', action='store_true', help='Disable GUI display')
    parser.add_argument('--no-log', action='store_true', help='Disable logging')
    
    args = parser.parse_args()
    
    # Dockerコンテナ内のパスに変換
    model_path = os.path.join('/app', args.model)
    prototxt_path = os.path.join('/app', args.prototxt)
    
    # ファイルパスの存在確認
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return
    
    if not os.path.exists(prototxt_path):
        print(f"Error: Prototxt file not found: {prototxt_path}")
        return
    
    # エリアのパース
    area = None
    if args.area:
        area_coords = [int(x) for x in args.area.split(',')]
        if len(area_coords) == 4:
            area = tuple(area_coords)
        else:
            print("Error: Area must be specified as 'x1,y1,x2,y2'")
            return
    
    # アプリケーション初期化
    app = PeopleCounterApp(
        model_path=model_path,
        prototxt_path=prototxt_path,
        confidence_threshold=args.confidence,
        similarity_threshold=args.similarity,
        enable_logging=not args.no_log
    )
    
    # ビデオキャプチャの初期化
    if args.input:
        input_path = os.path.join('/app', args.input)
        if not os.path.exists(input_path):
            print(f"Error: Input video file not found: {input_path}")
            return
        cap = cv2.VideoCapture(input_path)
    else:
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot open video source")
        return
    
    # ビデオライターの初期化
    # video_writer = None
    # if args.output:
    #     fps = int(cap.get(cv2.CAP_PROP_FPS))
    #     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
    #     output_path = os.path.join('/app', args.output)
    #     fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    #     video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), True)
        
    #     if not video_writer.isOpened():
    #         print(f"Error: Cannot create output video file: {output_path}")
    #         return
        
    #     print(f"Output video will be saved to: {output_path}")
    
    print("Press 'q' to quit")
    print(f"Detection confidence threshold: {args.confidence}")
    print(f"Similarity threshold: {args.similarity}")
    print(f"Logging enabled: {not args.no_log}")
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video or cannot read frame")
                break
            
            frame_count += 1
            
            # フレーム処理
            output_frame, person_boxes, person_ids, similarities = app.process_frame(
                frame, frame_count, area, args.line)
            
            # 動画の出力
            if args["output"] is not None and video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                video_writer = cv2.VideoWriter(args["output"], fourcc, 30,
                                            (W, H), True)
            

            # 結果の表示
            if not args.no_gui:
                cv2.imshow('People Counter with ReID', output_frame)
                
                # キー入力処理
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            
            # 統計情報の定期表示
            if frame_count % 30 == 0:  # 30フレームごと
                stats = app.identifier.get_statistics()
                print(f"Frame {frame_count}: Detected={len(person_boxes)}, Unique={stats['unique_persons']}")
                
                # 類似度の詳細表示
                if len(similarities) > 0:
                    avg_similarity = sum(s for s in similarities if s > 0) / max(1, len([s for s in similarities if s > 0]))
                    print(f"  Average similarity: {avg_similarity:.3f}")
                    for i, (pid, sim) in enumerate(zip(person_ids, similarities)):
                        if sim > 0:
                            print(f"    Person {i}: ID={pid}, similarity={sim:.3f}")
                        else:
                            print(f"    Person {i}: ID={pid} (new)")
            if video_writer is not None:
                video_writer.write(frame)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # クリーンアップ
        cap.release()
        if video_writer is not None:
            video_writer.release()
        if not args.no_gui:
            cv2.destroyAllWindows()
        
        # 最終統計とログ保存
        final_stats = app.identifier.get_statistics()
        print(f"\n=== Final Statistics ===")
        print(f"Total frames processed: {frame_count}")
        print(f"Unique persons detected: {final_stats['unique_persons']}")
        print(f"Total features stored: {final_stats['total_features']}")
        print(f"Similarity threshold: {final_stats['similarity_threshold']}")
        
        # JSON結果の保存
        if args.json_output:
            json_output_path = os.path.join('/app', args.json_output)
            results = {
                "timestamp": datetime.now().isoformat(),
                "total_frames": frame_count,
                "unique_persons": final_stats['unique_persons'],
                "total_features": final_stats['total_features'],
                "similarity_threshold": final_stats['similarity_threshold'],
                "person_database": app.identifier.get_person_database()
            }
            
            with open(json_output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"JSON results saved to: {json_output_path}")
        
        # ログ機能の終了処理
        if app.logger:
            app.logger.print_unique_persons_summary(app.identifier.get_person_database())
            app.logger.close()

if __name__ == "__main__":
    main()