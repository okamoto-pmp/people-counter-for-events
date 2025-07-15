import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict

class PersonIdentifier:
    """人物識別クラス（シンプルなReID実装）"""
    
    def __init__(self, similarity_threshold: float = 0.7):
        """
        Args:
            similarity_threshold: 同一人物とみなす類似度の閾値
        """
        self.similarity_threshold = similarity_threshold
        self.person_database = {}  # {person_id: [feature_vectors]}
        self.next_person_id = 1
        
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
    
    def identify_person(self, features: np.ndarray) -> int:
        """
        特徴量から人物を識別
        
        Args:
            features: 人物の特徴量
            
        Returns:
            人物ID（新しい人物の場合は新しいIDを発行）
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
            # 新しい特徴量を追加（最大5個まで保持）
            if len(self.person_database[best_match_id]) < 5:
                self.person_database[best_match_id].append(features)
            return best_match_id
        
        # 新しい人物として登録
        new_person_id = self.next_person_id
        self.next_person_id += 1
        self.person_database[new_person_id] = [features]
        
        return new_person_id
    
    def identify_persons(self, frame: np.ndarray, bboxes: List[Tuple[int, int, int, int]]) -> List[int]:
        """
        複数の人物を識別
        
        Args:
            frame: 入力フレーム
            bboxes: バウンディングボックスのリスト
            
        Returns:
            各人物のIDリスト
        """
        person_ids = []
        
        for bbox in bboxes:
            features = self.extract_features(frame, bbox)
            person_id = self.identify_person(features)
            person_ids.append(person_id)
        
        return person_ids
    
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