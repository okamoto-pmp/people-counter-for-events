import cv2
import argparse
import sys
import os

from person_detector import PersonDetector
from person_identifier import PersonIdentifier
from visualizer import Visualizer

class PeopleCounterApp:
    """人物検出・識別・描画を統合したアプリケーション"""
    
    def __init__(self, model_path: str, prototxt_path: str, confidence_threshold: float = 0.4, 
                 similarity_threshold: float = 0.7):
        """
        Args:
            model_path: 検出モデルのパス
            prototxt_path: prototxtファイルのパス
            confidence_threshold: 検出信頼度の閾値
            similarity_threshold: 識別類似度の閾値
        """
        self.detector = PersonDetector(model_path, prototxt_path, confidence_threshold)
        self.identifier = PersonIdentifier(similarity_threshold)
        self.visualizer = Visualizer()
        
    def process_frame(self, frame, area=None, line_y=None):
        """
        フレームを処理（検出→識別→描画）
        
        Args:
            frame: 入力フレーム
            area: 検出エリア (x1, y1, x2, y2) or None
            line_y: 検出ライン or None
            
        Returns:
            処理後のフレーム
        """
        # 1. 人物検出
        person_boxes = self.detector.detect_persons(frame)
        
        # 2. 人物識別
        person_ids = self.identifier.identify_persons(frame, person_boxes)
        
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
            'Total Features': stats['total_features']
        }
        output_frame = self.visualizer.draw_info(output_frame, info)
        
        return output_frame, person_boxes, person_ids

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='People Counter with ReID')
    parser.add_argument('-m', '--model', required=True, help='Path to detection model')
    parser.add_argument('-p', '--prototxt', required=True, help='Path to prototxt file')
    parser.add_argument('-i', '--input', help='Input video file (default: webcam)')
    parser.add_argument('-c', '--confidence', type=float, default=0.4, help='Detection confidence threshold')
    parser.add_argument('-s', '--similarity', type=float, default=0.7, help='Person identification similarity threshold')
    parser.add_argument('-a', '--area', help='Detection area as x1,y1,x2,y2')
    parser.add_argument('-l', '--line', type=int, help='Detection line Y coordinate')
    parser.add_argument('--no-gui', action='store_true', help='Disable GUI display')
    
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
        similarity_threshold=args.similarity
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
    
    print("Press 'q' to quit")
    print(f"Detection confidence threshold: {args.confidence}")
    print(f"Similarity threshold: {args.similarity}")
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video or cannot read frame")
                break
            
            frame_count += 1
            
            # フレーム処理
            output_frame, person_boxes, person_ids = app.process_frame(frame, area, args.line)
            
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
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # クリーンアップ
        cap.release()
        if not args.no_gui:
            cv2.destroyAllWindows()
        
        # 最終統計
        final_stats = app.identifier.get_statistics()
        print(f"\n=== Final Statistics ===")
        print(f"Total frames processed: {frame_count}")
        print(f"Unique persons detected: {final_stats['unique_persons']}")
        print(f"Total features stored: {final_stats['total_features']}")
        print(f"Similarity threshold: {final_stats['similarity_threshold']}")

if __name__ == "__main__":
    main()