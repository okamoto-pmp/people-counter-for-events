# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
import json
from datetime import datetime

from .trackableObject import TrackableObject
from .reid_extractor import compute_similarity

class UniquePersonCounter:
    """
    Person Re-identificationを使用したユニークな人物カウンター
    """
    
    def __init__(self, similarity_threshold: float = 0.7):
        """
        Args:
            similarity_threshold: 同一人物とみなす類似度の閾値
        """
        self.similarity_threshold = similarity_threshold
        self.unique_persons = []  # ユニークな人物のリスト
        self.person_groups = {}   # trackableObjectsのグループ分け
        self.next_unique_id = 1
        
    def assign_unique_ids(self, trackable_objects: List[TrackableObject]) -> Dict[int, List[TrackableObject]]:
        """
        trackableObjectsにユニークIDを付与し、グループ分けを行う
        
        Args:
            trackable_objects: 処理対象のtrackableObjectsリスト
            
        Returns:
            ユニークIDごとにグループ分けされたtrackableObjectsの辞書
        """
        # 特徴量を持つオブジェクトのみをフィルタリング
        valid_objects = [obj for obj in trackable_objects if obj.feature_vectors]
        
        if not valid_objects:
            return {}
        
        # 類似度行列を計算
        similarity_matrix = self._compute_similarity_matrix(valid_objects)
        
        # クラスタリング（単純なグリーディアルゴリズム）
        groups = self._cluster_objects(valid_objects, similarity_matrix)
        
        # 各グループにユニークIDを付与
        result = {}
        for group in groups:
            unique_id = self.next_unique_id
            self.next_unique_id += 1
            
            # グループ内の各オブジェクトにユニークIDを設定
            for obj in group:
                obj.unique_person_id = unique_id
            
            result[unique_id] = group
            
        return result
    
    def _compute_similarity_matrix(self, objects: List[TrackableObject]) -> np.ndarray:
        """
        オブジェクト間の類似度行列を計算
        """
        n = len(objects)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                # 各オブジェクトの最も代表的な特徴量を使用
                feature_i = self._get_representative_feature(objects[i])
                feature_j = self._get_representative_feature(objects[j])
                
                if feature_i is not None and feature_j is not None:
                    similarity = compute_similarity(feature_i, feature_j)
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity
        
        return similarity_matrix
    
    def _get_representative_feature(self, obj: TrackableObject) -> np.ndarray:
        """
        オブジェクトの代表的な特徴量を取得
        複数の特徴量がある場合は平均を取る
        """
        if not obj.feature_vectors:
            return None
        
        if len(obj.feature_vectors) == 1:
            return obj.feature_vectors[0]
        
        # 複数の特徴量の平均を計算
        stacked_features = np.stack(obj.feature_vectors)
        mean_feature = np.mean(stacked_features, axis=0)
        
        # 正規化
        return mean_feature / (np.linalg.norm(mean_feature) + 1e-6)
    
    def _cluster_objects(self, objects: List[TrackableObject], similarity_matrix: np.ndarray) -> List[List[TrackableObject]]:
        """
        類似度行列を基にオブジェクトをクラスタリング
        """
        n = len(objects)
        visited = [False] * n
        groups = []
        
        for i in range(n):
            if visited[i]:
                continue
                
            # 新しいグループを開始
            group = [objects[i]]
            visited[i] = True
            
            # 類似度が閾値を超える他のオブジェクトを同じグループに追加
            for j in range(i + 1, n):
                if not visited[j] and similarity_matrix[i, j] > self.similarity_threshold:
                    group.append(objects[j])
                    visited[j] = True
            
            groups.append(group)
        
        return groups
    
    def get_unique_count_statistics(self, person_groups: Dict[int, List[TrackableObject]]) -> Dict:
        """
        ユニークカウントの統計情報を取得
        """
        stats = {
            'total_unique_persons': len(person_groups),
            'total_tracking_objects': sum(len(group) for group in person_groups.values()),
            'person_details': {}
        }
        
        for unique_id, group in person_groups.items():
            # 各ユニークな人物の詳細情報
            enter_count = sum(1 for obj in group if obj.enter_counted)
            leave_count = sum(1 for obj in group if obj.leave_counted)
            
            stats['person_details'][unique_id] = {
                'tracking_objects_count': len(group),
                'enter_events': enter_count,
                'leave_events': leave_count,
                'object_ids': [obj.objectID for obj in group]
            }
        
        return stats
    
    def save_results(self, person_groups: Dict[int, List[TrackableObject]], filepath: str):
        """
        結果をJSONファイルに保存
        """
        stats = self.get_unique_count_statistics(person_groups)
        
        # 保存用のデータ構造を準備
        save_data = {
            'timestamp': datetime.now().isoformat(),
            'similarity_threshold': self.similarity_threshold,
            'statistics': stats
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        print(f"ユニークカウント結果を保存しました: {filepath}")
        
    def print_summary(self, person_groups: Dict[int, List[TrackableObject]]):
        """
        結果のサマリーを表示
        """
        stats = self.get_unique_count_statistics(person_groups)
        
        print("\n=== ユニークカウント結果 ===")
        print(f"ユニークな人物数: {stats['total_unique_persons']}")
        print(f"総追跡オブジェクト数: {stats['total_tracking_objects']}")
        print(f"類似度閾値: {self.similarity_threshold}")
        
        print("\n=== 詳細 ===")
        for unique_id, details in stats['person_details'].items():
            print(f"人物ID {unique_id}:")
            print(f"  追跡オブジェクト数: {details['tracking_objects_count']}")
            print(f"  入場イベント数: {details['enter_events']}")
            print(f"  退場イベント数: {details['leave_events']}")
            print(f"  オブジェクトID: {details['object_ids']}")