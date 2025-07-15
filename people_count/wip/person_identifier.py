import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from demographic_analyzer import DemographicAnalyzer

class PersonIdentifier:
    """人物識別クラス（シンプルなReID実装）"""
    
    def __init__(self, similarity_threshold: float = 0.7, enable_demographics: bool = True):
        """
        Args:
            similarity_threshold: 同一人物とみなす類似度の閾値
            enable_demographics: 人口統計分析を有効にするかどうか
        """
        self.similarity_threshold = similarity_threshold
        self.person_database = {}  # {person_id: [feature_vectors]}
        self.person_bbox_history = {}  # {person_id: [bbox_history]}
        self.person_demographics = {}  # {person_id: demographic_info}
        self.next_person_id = 1
        self.logger = None  # ログ機能（後で設定）
        
        # 人口統計分析器の初期化
        if enable_demographics:
            self.demographic_analyzer = DemographicAnalyzer()
        else:
            self.demographic_analyzer = None
        
    def set_logger(self, logger):
        """ログ機能を設定"""
        self.logger = logger
        
    def _filter_bbox_size(self, bbox: Tuple[int, int, int, int], 
                         person_id: int = None) -> Tuple[int, int, int, int]:
        """
        バウンディングボックスのサイズをフィルタリングして安定化
        
        Args:
            bbox: バウンディングボックス (x1, y1, x2, y2)
            person_id: 既存の人物ID（新規の場合はNone）
            
        Returns:
            フィルタリング後のバウンディングボックス
        """
        x1, y1, x2, y2 = bbox
        current_size = (x2 - x1) * (y2 - y1)
        
        # 既存の人物の場合、履歴を使ってサイズを安定化
        if person_id is not None and person_id in self.person_bbox_history:
            history = self.person_bbox_history[person_id]
            if len(history) > 0:
                # 過去のサイズの平均を計算
                past_sizes = [(h[2] - h[0]) * (h[3] - h[1]) for h in history]
                avg_size = np.mean(past_sizes)
                
                # 現在のサイズが過去の平均から大きく外れる場合は調整
                size_ratio = current_size / avg_size if avg_size > 0 else 1.0
                
                if self.logger:
                    self.logger.log_bbox_size_change(0, person_id, int(avg_size), current_size, size_ratio)
                
                # 2倍以上または半分以下の変化の場合は調整
                if size_ratio > 2.0 or size_ratio < 0.5:
                    # 過去の平均サイズに近づける
                    target_size = int(avg_size * 0.8 + current_size * 0.2)  # 重み付き平均
                    scale_factor = (target_size / current_size) ** 0.5
                    
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    width = int((x2 - x1) * scale_factor)
                    height = int((y2 - y1) * scale_factor)
                    
                    x1 = center_x - width // 2
                    y1 = center_y - height // 2
                    x2 = center_x + width // 2
                    y2 = center_y + height // 2
        
        return (x1, y1, x2, y2)
    
    def _update_bbox_history(self, person_id: int, bbox: Tuple[int, int, int, int]):
        """バウンディングボックスの履歴を更新"""
        if person_id not in self.person_bbox_history:
            self.person_bbox_history[person_id] = []
        
        self.person_bbox_history[person_id].append(bbox)
        
        # 履歴は最大10個まで保持
        if len(self.person_bbox_history[person_id]) > 10:
            self.person_bbox_history[person_id].pop(0)
        
    def extract_features(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        人物の特徴量を抽出
        
        Args:
            frame: 入力フレーム
            bbox: バウンディングボックス (x1, y1, x2, y2)
            
        Returns:
            特徴量ベクトル
        """
        x1, y1, x2, y2 = bbox
        
        # 人物領域を切り出し
        person_crop = frame[y1:y2, x1:x2]
        
        if person_crop.size == 0:
            return np.zeros(128)  # ダミーの特徴量
        
        # サイズを正規化
        person_crop = cv2.resize(person_crop, (64, 128))
        
        # HSVカラーヒストグラムを特徴量として使用
        hsv = cv2.cvtColor(person_crop, cv2.COLOR_BGR2HSV)
        
        # 上半身と下半身に分けてヒストグラムを計算
        height = hsv.shape[0]
        upper_half = hsv[:height//2, :]
        lower_half = hsv[height//2:, :]
        
        # HSVチャンネルそれぞれのヒストグラムを計算
        hist_upper = []
        hist_lower = []
        
        for i in range(3):  # H, S, V
            hist_u = cv2.calcHist([upper_half], [i], None, [16], [0, 256])
            hist_l = cv2.calcHist([lower_half], [i], None, [16], [0, 256])
            hist_upper.extend(hist_u.flatten())
            hist_lower.extend(hist_l.flatten())
        
        # 特徴量を結合
        features = np.array(hist_upper + hist_lower)
        
        # L2正規化
        features = features / (np.linalg.norm(features) + 1e-6)
        
        return features
    
    def compute_similarity(self, feature1: np.ndarray, feature2: np.ndarray) -> float:
        """
        2つの特徴量の類似度を計算
        
        Args:
            feature1: 特徴量1
            feature2: 特徴量2
            
        Returns:
            類似度スコア (0.0 ~ 1.0)
        """
        # コサイン類似度を計算
        dot_product = np.dot(feature1, feature2)
        norm_product = np.linalg.norm(feature1) * np.linalg.norm(feature2)
        
        if norm_product == 0:
            return 0.0
        
        similarity = dot_product / norm_product
        return max(0.0, similarity)
    
    def identify_person(self, features: np.ndarray, bbox: Tuple[int, int, int, int], 
                       frame_num: int = 0, frame: Optional[np.ndarray] = None) -> Tuple[int, float]:
        """
        特徴量から人物を識別
        
        Args:
            features: 人物の特徴量
            bbox: バウンディングボックス
            frame_num: フレーム番号
            frame: フレーム画像（人口統計分析用）
            
        Returns:
            (人物ID, 類似度)
        """
        best_match_id = None
        best_similarity = 0.0
        
        # 既存の人物データベースと比較
        for person_id, stored_features in self.person_database.items():
            for stored_feature in stored_features:
                similarity = self.compute_similarity(features, stored_feature)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = person_id
        
        # 類似度が閾値を超えた場合は既存の人物として識別
        if best_similarity > self.similarity_threshold:
            # バウンディングボックスをフィルタリング
            filtered_bbox = self._filter_bbox_size(bbox, best_match_id)
            
            # 新しい特徴量を追加（最大5個まで保持）
            if len(self.person_database[best_match_id]) < 5:
                self.person_database[best_match_id].append(features)
            
            # バウンディングボックス履歴を更新
            self._update_bbox_history(best_match_id, filtered_bbox)
            
            # 人口統計情報を更新（フレームが提供されている場合）
            if frame is not None and self.demographic_analyzer:
                self._update_demographics(best_match_id, frame, filtered_bbox)
            
            # ログ出力
            if self.logger:
                self.logger.log_person_match(frame_num, best_match_id, best_similarity, filtered_bbox)
            
            return best_match_id, best_similarity
        
        # 新しい人物として登録
        new_person_id = self.next_person_id
        self.next_person_id += 1
        self.person_database[new_person_id] = [features]
        
        # バウンディングボックス履歴を初期化
        self._update_bbox_history(new_person_id, bbox)
        
        # 人口統計情報を初期化（フレームが提供されている場合）
        if frame is not None and self.demographic_analyzer:
            self._update_demographics(new_person_id, frame, bbox)
        
        # ログ出力
        if self.logger:
            self.logger.log_new_person(frame_num, new_person_id, bbox)
        
        return new_person_id, 0.0
    
    def identify_persons(self, frame: np.ndarray, bboxes: List[Tuple[int, int, int, int]], 
                        frame_num: int = 0) -> Tuple[List[int], List[float]]:
        """
        複数の人物を識別
        
        Args:
            frame: 入力フレーム
            bboxes: バウンディングボックスのリスト
            frame_num: フレーム番号
            
        Returns:
            (各人物のIDリスト, 類似度リスト)
        """
        person_ids = []
        similarities = []
        
        for bbox in bboxes:
            features = self.extract_features(frame, bbox)
            person_id, similarity = self.identify_person(features, bbox, frame_num, frame)
            person_ids.append(person_id)
            similarities.append(similarity)
        
        # ログ出力
        if self.logger:
            self.logger.log_person_identification(frame_num, person_ids, similarities)
            self.logger.log_database_status(frame_num, self.person_database)
        
        return person_ids, similarities
    
    def get_unique_count(self) -> int:
        """
        現在までに識別されたユニークな人物数を取得
        
        Returns:
            ユニークな人物数
        """
        return len(self.person_database)
    
    def get_statistics(self) -> Dict:
        """
        統計情報を取得
        
        Returns:
            統計情報の辞書
        """
        return {
            'unique_persons': len(self.person_database),
            'total_features': sum(len(features) for features in self.person_database.values()),
            'similarity_threshold': self.similarity_threshold
        }
    
    def get_person_database(self) -> Dict:
        """人物データベースを取得（デバッグ用）"""
        return self.person_database.copy()
    
    def _update_demographics(self, person_id: int, frame: np.ndarray, bbox: Tuple[int, int, int, int]):
        """
        人物の人口統計情報を更新
        
        Args:
            person_id: 人物ID
            frame: フレーム画像
            bbox: バウンディングボックス
        """
        if not self.demographic_analyzer:
            return
        
        # 人口統計分析を実行
        demographics = self.demographic_analyzer.analyze_demographics(frame, bbox)
        
        # 既存の情報と統合（信頼度の高い情報を優先）
        if person_id in self.person_demographics:
            existing = self.person_demographics[person_id]
            
            # 性別情報の更新
            if demographics['gender_confidence'] > existing.get('gender_confidence', 0):
                existing['gender'] = demographics['gender']
                existing['gender_confidence'] = demographics['gender_confidence']
            
            # 年齢情報の更新
            if demographics['age_confidence'] > existing.get('age_confidence', 0):
                existing['age_group'] = demographics['age_group']
                existing['age_confidence'] = demographics['age_confidence']
            
            # 顔検出情報の更新
            if demographics['face_detected']:
                existing['face_detected'] = True
        else:
            # 新規登録
            self.person_demographics[person_id] = demographics
    
    def get_person_demographics(self, person_id: int) -> Optional[Dict]:
        """
        指定された人物の人口統計情報を取得
        
        Args:
            person_id: 人物ID
            
        Returns:
            人口統計情報の辞書、または None
        """
        return self.person_demographics.get(person_id)
    
    def get_all_demographics(self) -> Dict:
        """
        全ての人物の人口統計情報を取得
        
        Returns:
            全人物の人口統計情報
        """
        return self.person_demographics.copy()
    
    def get_demographic_summary(self) -> Dict:
        """
        人口統計情報の要約を取得
        
        Returns:
            統計要約
        """
        if not self.demographic_analyzer:
            return {}
        
        summary = {
            'total_persons': len(self.person_demographics),
            'gender_distribution': {'male': 0, 'female': 0, 'unknown': 0},
            'age_distribution': {
                'child': 0, 'teen': 0, 'young_adult': 0, 
                'middle_aged': 0, 'senior': 0, 'unknown': 0
            },
            'face_detection_rate': 0.0
        }
        
        if len(self.person_demographics) == 0:
            return summary
        
        faces_detected = 0
        
        for person_id, demographics in self.person_demographics.items():
            # 性別分布
            gender = demographics.get('gender')
            if gender:
                summary['gender_distribution'][gender.value] += 1
            
            # 年齢分布
            age_group = demographics.get('age_group')
            if age_group:
                summary['age_distribution'][age_group.value] += 1
            
            # 顔検出率
            if demographics.get('face_detected', False):
                faces_detected += 1
        
        summary['face_detection_rate'] = faces_detected / len(self.person_demographics)
        
        return summary