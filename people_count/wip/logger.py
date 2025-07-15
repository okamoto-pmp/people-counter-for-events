import logging
import os
from datetime import datetime
from typing import Dict, List, Tuple
import json

class PersonCounterLogger:
    """人物カウンターのログ機能"""
    
    def __init__(self, log_dir: str = "/app/logs"):
        """
        Args:
            log_dir: ログディレクトリのパス
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # ログファイル名（タイムスタンプ付き）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 詳細ログ用のファイルハンドラー
        self.detail_log_file = os.path.join(log_dir, f"detail_{timestamp}.log")
        self.debug_log_file = os.path.join(log_dir, f"debug_{timestamp}.log")
        self.person_log_file = os.path.join(log_dir, f"persons_{timestamp}.json")
        
        # ロガーの設定
        self.logger = logging.getLogger('PersonCounter')
        self.logger.setLevel(logging.DEBUG)
        
        # ファイルハンドラーの設定
        file_handler = logging.FileHandler(self.detail_log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # フォーマッターの設定
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        
        # 人物データの保存用リスト
        self.person_data = []
        
        print(f"ログファイル: {self.detail_log_file}")
        print(f"デバッグログ: {self.debug_log_file}")
        print(f"人物データ: {self.person_log_file}")
    
    def log_frame_process(self, frame_num: int, detected_count: int, unique_count: int):
        """フレーム処理のログ"""
        self.logger.info(f"Frame {frame_num}: Detected={detected_count}, Unique={unique_count}")
    
    def log_person_detection(self, frame_num: int, person_boxes: List[Tuple[int, int, int, int]]):
        """人物検出のログ"""
        self.logger.debug(f"Frame {frame_num}: Detected {len(person_boxes)} persons")
        for i, (x1, y1, x2, y2) in enumerate(person_boxes):
            size = (x2 - x1) * (y2 - y1)
            self.logger.debug(f"  Person {i}: bbox=({x1},{y1},{x2},{y2}), size={size}")
    
    def log_person_identification(self, frame_num: int, person_ids: List[int], 
                                 similarities: List[float] = None):
        """人物識別のログ"""
        self.logger.debug(f"Frame {frame_num}: Identified persons: {person_ids}")
        if similarities:
            for i, (person_id, similarity) in enumerate(zip(person_ids, similarities)):
                self.logger.debug(f"  Person {i}: ID={person_id}, similarity={similarity:.3f}")
    
    def log_new_person(self, frame_num: int, person_id: int, bbox: Tuple[int, int, int, int]):
        """新しい人物の登録ログ"""
        self.logger.info(f"Frame {frame_num}: New person registered - ID={person_id}, bbox={bbox}")
        
        # 人物データに追加
        person_info = {
            'frame': frame_num,
            'person_id': person_id,
            'bbox': bbox,
            'timestamp': datetime.now().isoformat(),
            'action': 'new_person'
        }
        self.person_data.append(person_info)
    
    def log_person_match(self, frame_num: int, person_id: int, similarity: float, 
                        bbox: Tuple[int, int, int, int]):
        """既存人物とのマッチングログ"""
        self.logger.debug(f"Frame {frame_num}: Person matched - ID={person_id}, similarity={similarity:.3f}, bbox={bbox}")
        
        # 人物データに追加
        person_info = {
            'frame': frame_num,
            'person_id': person_id,
            'similarity': similarity,
            'bbox': bbox,
            'timestamp': datetime.now().isoformat(),
            'action': 'person_match'
        }
        self.person_data.append(person_info)
    
    def log_bbox_size_change(self, frame_num: int, person_id: int, 
                           old_size: int, new_size: int, change_ratio: float):
        """バウンディングボックスサイズ変化のログ"""
        if change_ratio > 2.0 or change_ratio < 0.5:  # 2倍以上または半分以下の変化
            self.logger.warning(f"Frame {frame_num}: Large bbox size change - "
                              f"ID={person_id}, old={old_size}, new={new_size}, "
                              f"ratio={change_ratio:.2f}")
    
    def log_database_status(self, frame_num: int, database: Dict):
        """人物データベースの状態ログ"""
        self.logger.debug(f"Frame {frame_num}: Database status - {len(database)} persons")
        for person_id, features in database.items():
            self.logger.debug(f"  Person {person_id}: {len(features)} features stored")
    
    def save_person_data(self):
        """人物データをJSONファイルに保存"""
        try:
            with open(self.person_log_file, 'w', encoding='utf-8') as f:
                json.dump(self.person_data, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Person data saved to {self.person_log_file}")
        except Exception as e:
            self.logger.error(f"Failed to save person data: {e}")
    
    def print_unique_persons_summary(self, database: Dict):
        """ユニークな人物の詳細サマリーを表示・保存"""
        summary = {
            'total_unique_persons': len(database),
            'persons': {}
        }
        
        print("\n=== ユニークな人物の詳細 ===")
        for person_id, features in database.items():
            # この人物が登場したフレーム数を計算
            person_frames = [data for data in self.person_data if data['person_id'] == person_id]
            first_frame = min(data['frame'] for data in person_frames) if person_frames else 0
            last_frame = max(data['frame'] for data in person_frames) if person_frames else 0
            
            person_summary = {
                'person_id': person_id,
                'feature_count': len(features),
                'first_appearance': first_frame,
                'last_appearance': last_frame,
                'total_appearances': len(person_frames)
            }
            
            summary['persons'][person_id] = person_summary
            
            print(f"Person ID {person_id}:")
            print(f"  特徴量数: {len(features)}")
            print(f"  初回登場: Frame {first_frame}")
            print(f"  最終登場: Frame {last_frame}")
            print(f"  総登場回数: {len(person_frames)}")
        
        # サマリーをファイルに保存
        summary_file = os.path.join(self.log_dir, f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"\nサマリーファイル: {summary_file}")
        except Exception as e:
            print(f"サマリー保存エラー: {e}")
    
    def close(self):
        """ログ機能を終了"""
        self.save_person_data()
        
        # ハンドラーを閉じる
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)