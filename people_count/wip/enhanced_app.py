import cv2
import argparse
import sys
import os
import json
import time
from datetime import datetime
from typing import Optional, Tuple, List, Dict

from person_detector import PersonDetector
from person_identifier import PersonIdentifier
from unified_tracker import UnifiedTracker
from visualizer import Visualizer
from logger import PersonCounterLogger
# np
import numpy as np

class EnhancedPeopleCounterApp:
    """ReIDと入退場管理を統合した高精度人物カウンターアプリケーション"""
    
    def __init__(self, model_path: str, prototxt_path: str, 
                 confidence_threshold: float = 0.4, 
                 similarity_threshold: float = 0.6,
                 max_disappeared: int = 80,
                 max_distance: int = 100,
                 enable_logging: bool = True):
        """
        Args:
            model_path: 検出モデルのパス
            prototxt_path: prototxtファイルのパス
            confidence_threshold: 検出信頼度の閾値
            similarity_threshold: 識別類似度の閾値
            max_disappeared: トラッキングの最大消失フレーム数
            max_distance: トラッキングの最大距離
            enable_logging: ログ機能を有効にするかどうか
        """
        self.detector = PersonDetector(model_path, prototxt_path, confidence_threshold)
        self.identifier = PersonIdentifier(similarity_threshold)
        self.tracker = UnifiedTracker(max_disappeared, max_distance)
        self.visualizer = Visualizer()
        
        # ログ機能の初期化
        if enable_logging:
            self.logger = PersonCounterLogger()
            self.identifier.set_logger(self.logger)
        else:
            self.logger = None
        
        self.confidence_threshold = confidence_threshold
        self.similarity_threshold = similarity_threshold
        
        # 統計情報
        self.total_frames = 0
        self.last_detection_time = time.time()
        self.detection_interval = 0.1  # 100ms間隔で検出
        
    def process_frame(self, frame, frame_num: int, 
                     area: Optional[Tuple[int, int, int, int]] = None,
                     line_y: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        フレームを処理（検出→識別→トラッキング→描画）
        
        Args:
            frame: 入力フレーム
            frame_num: フレーム番号
            area: 検出エリア (x1, y1, x2, y2) or None
            line_y: 検出ライン or None
            
        Returns:
            (処理後のフレーム, 統計情報)
        """
        self.total_frames += 1
        current_time = time.time()
        
        # 検出間隔の制御
        should_detect = (current_time - self.last_detection_time) > self.detection_interval
        
        if should_detect:
            # 1. 人物検出
            person_boxes = self.detector.detect_persons(frame)
            self.last_detection_time = current_time
            
            # 検出ログ
            if self.logger:
                self.logger.log_person_detection(frame_num, person_boxes)
        else:
            person_boxes = []
        
        # 2. 人物識別（検出された場合のみ）
        if person_boxes:
            person_ids, similarities = self.identifier.identify_persons(frame, person_boxes, frame_num)
        else:
            person_ids = []
            similarities = []
        
        # 3. トラッキング更新
        tracked_objects, deregistered_objects = self.tracker.update(person_boxes, area, line_y)
        
        # 4. 描画
        output_frame = self._draw_frame(frame, person_boxes, person_ids, similarities, 
                                      tracked_objects, area, line_y, frame_num)
        
        # 5. 統計情報の取得
        stats = self._get_comprehensive_stats(person_boxes, person_ids, similarities)
        
        # フレーム処理ログ
        if self.logger:
            self.logger.log_frame_process(frame_num, len(person_boxes), stats['unique_persons'])
        
        return output_frame, stats
    
    def _draw_frame(self, frame, person_boxes: List[Tuple[int, int, int, int]], 
                   person_ids: List[int], similarities: List[float],
                   tracked_objects: Dict[int, Tuple[int, int]],
                   area: Optional[Tuple[int, int, int, int]],
                   line_y: Optional[int],
                   frame_num: int) -> np.ndarray:
        """フレームに各種情報を描画"""
        output_frame = frame.copy()
        
        # エリアまたはラインを描画
        if area:
            output_frame = self.visualizer.draw_detection_area(output_frame, area)
        if line_y:
            output_frame = self.visualizer.draw_detection_line(output_frame, line_y)
        
        # バウンディングボックスとIDを描画
        output_frame = self.visualizer.draw_bounding_boxes(output_frame, person_boxes, person_ids)
        
        # トラッキング情報を描画
        for object_id, (cx, cy) in tracked_objects.items():
            cv2.circle(output_frame, (cx, cy), 6, (255, 0, 0), -1)
            cv2.putText(output_frame, f"T{object_id}", (cx + 10, cy - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # 統計情報を描画
        stats = self._get_comprehensive_stats(person_boxes, person_ids, similarities)
        info = {
            'Frame': frame_num,
            'Detected': len(person_boxes),
            'Unique': stats['unique_persons'],
            'Tracked': len(tracked_objects),
            'Entered': stats['entry_count'],
            'Exited': stats['exit_count'],
            'Current': stats['current_count']
        }
        
        output_frame = self.visualizer.draw_info(output_frame, info)
        
        return output_frame
    
    def _get_comprehensive_stats(self, person_boxes: List[Tuple[int, int, int, int]], 
                                person_ids: List[int], similarities: List[float]) -> Dict:
        """包括的な統計情報を取得"""
        reid_stats = self.identifier.get_statistics()
        tracker_stats = self.tracker.get_statistics()
        
        return {
            'detected_persons': len(person_boxes),
            'unique_persons': reid_stats['unique_persons'],
            'total_features': reid_stats['total_features'],
            'tracked_objects': tracker_stats['total_objects'],
            'entry_count': tracker_stats['entry_count'],
            'exit_count': tracker_stats['exit_count'],
            'current_count': tracker_stats['current_count'],
            'avg_similarity': sum(s for s in similarities if s > 0) / max(1, len([s for s in similarities if s > 0])) if similarities else 0.0,
            'similarity_threshold': self.similarity_threshold,
            'confidence_threshold': self.confidence_threshold,
            'total_frames': self.total_frames
        }
    

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='Enhanced People Counter with ReID and Area/Line Tracking')
    parser.add_argument('-m', '--model', required=True, help='Path to detection model')
    parser.add_argument('-p', '--prototxt', required=True, help='Path to prototxt file')
    parser.add_argument('-i', '--input', help='Input video file (default: webcam)')
    parser.add_argument('-c', '--confidence', type=float, default=0.4, help='Detection confidence threshold')
    parser.add_argument('-s', '--similarity', type=float, default=0.6, help='Person identification similarity threshold')
    parser.add_argument('-a', '--area', help='Detection area as x1,y1,x2,y2')
    parser.add_argument('-l', '--line', type=int, help='Detection line Y coordinate')
    parser.add_argument('--max-disappeared', type=int, default=80, help='Maximum disappeared frames for tracking')
    parser.add_argument('--max-distance', type=int, default=100, help='Maximum distance for tracking')
    parser.add_argument('--no-gui', action='store_true', help='Disable GUI display')
    parser.add_argument('--no-log', action='store_true', help='Disable logging')
    parser.add_argument('-o', '--output', help='Output video file path')
    
    args = parser.parse_args()
    
    # エリアとラインの同時指定をチェック
    if args.area and args.line:
        print("Error: Cannot specify both area and line parameters at the same time.")
        return
    
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
    app = EnhancedPeopleCounterApp(
        model_path=model_path,
        prototxt_path=prototxt_path,
        confidence_threshold=args.confidence,
        similarity_threshold=args.similarity,
        max_disappeared=args.max_disappeared,
        max_distance=args.max_distance,
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
    video_writer = None
    if args.output:
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        output_path = os.path.join('/app', args.output)
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), True)
        
        if not video_writer.isOpened():
            print(f"Error: Cannot create output video file: {output_path}")
            return
        
        print(f"Output video will be saved to: {output_path}")
    
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
            output_frame, stats = app.process_frame(frame, frame_count, area, args.line)
            
            # 動画の出力
            if video_writer is not None:
                video_writer.write(output_frame)
            
            # 結果の表示
            if not args.no_gui:
                cv2.imshow('Enhanced People Counter', output_frame)
                
                # キー入力処理
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            
            # 統計情報の定期表示
            if frame_count % 30 == 0:  # 30フレームごと
                print(f"Frame {frame_count}: Detected={stats['detected_persons']}, "
                      f"Unique={stats['unique_persons']}, Tracked={stats['tracked_objects']}, "
                      f"Enter={stats['entry_count']}, Exit={stats['exit_count']}, "
                      f"Current={stats['current_count']}")
                
                if stats['avg_similarity'] > 0:
                    print(f"  Average similarity: {stats['avg_similarity']:.3f}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # クリーンアップ
        cap.release()
        if video_writer is not None:
            video_writer.release()
            print(f"Video saved successfully: {args.output}")
        if not args.no_gui:
            cv2.destroyAllWindows()
        
        # 最終統計の表示
        final_stats = app._get_comprehensive_stats([], [], [])
        print(f"\n=== Final Statistics ===")
        print(f"Total frames processed: {final_stats['total_frames']}")
        print(f"Unique persons detected: {final_stats['unique_persons']}")
        print(f"Total features stored: {final_stats['total_features']}")
        print(f"Entry events: {final_stats['entry_count']}")
        print(f"Exit events: {final_stats['exit_count']}")
        print(f"Current count: {final_stats['current_count']}")
        print(f"Similarity threshold: {final_stats['similarity_threshold']}")
        
        # ログ機能の終了処理
        if app.logger:
            app.logger.print_unique_persons_summary(app.identifier.get_person_database())
            app.logger.close()

if __name__ == "__main__":
    main()