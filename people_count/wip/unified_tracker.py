import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import time

class UnifiedTracker:
    """ReIDと入退場管理を統合したトラッカー"""
    
    def __init__(self, max_disappeared: int = 80, max_distance: int = 100):
        """
        Args:
            max_disappeared: オブジェクトが消失してから削除されるまでの最大フレーム数
            max_distance: 同一オブジェクトとみなす最大距離
        """
        self.next_object_id = 1
        self.objects = {}  # {object_id: (centroid_x, centroid_y)}
        self.disappeared = defaultdict(int)  # {object_id: disappeared_frames}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
        # 入退場管理のためのデータ
        self.object_trajectories = defaultdict(list)  # {object_id: [centroid_history]}
        self.object_states = {}  # {object_id: {'entered': bool, 'in_area': bool}}
        self.enter_events = []  # [(object_id, timestamp, centroid)]
        self.exit_events = []   # [(object_id, timestamp, centroid)]
    
    def register(self, centroid: Tuple[int, int]) -> int:
        """新しいオブジェクトを登録"""
        object_id = self.next_object_id
        self.objects[object_id] = centroid
        self.disappeared[object_id] = 0
        self.object_trajectories[object_id] = [centroid]
        self.object_states[object_id] = {'entered': False, 'in_area': False}
        self.next_object_id += 1
        return object_id
    
    def deregister(self, object_id: int):
        """オブジェクトを削除"""
        if object_id in self.objects:
            del self.objects[object_id]
            del self.disappeared[object_id]
            del self.object_trajectories[object_id]
            del self.object_states[object_id]
    
    def update(self, rects: List[Tuple[int, int, int, int]], 
               area: Optional[Tuple[int, int, int, int]] = None, 
               line_y: Optional[int] = None) -> Tuple[Dict[int, Tuple[int, int]], Dict[int, Tuple[int, int]]]:
        """
        トラッキングを更新
        
        Args:
            rects: 検出されたバウンディングボックス
            area: 検出エリア (x1, y1, x2, y2)
            line_y: 検出ライン Y座標
            
        Returns:
            (現在のオブジェクト, 消失したオブジェクト)
        """
        # 検出されたオブジェクトがない場合
        if len(rects) == 0:
            # 消失カウントを増やす
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                # 最大消失フレームを超えたら削除
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects.copy(), {}
        
        # 検出されたオブジェクトの重心を計算
        input_centroids = []
        for (x1, y1, x2, y2) in rects:
            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)
            input_centroids.append((cx, cy))
        
        # 既存のオブジェクトがない場合、全て新規登録
        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)
        else:
            # 既存オブジェクトと新しい重心の距離を計算
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            
            # 距離行列を計算
            distances = np.linalg.norm(
                np.array(object_centroids)[:, np.newaxis] - 
                np.array(input_centroids), axis=2)
            
            # 最小距離でマッチング
            rows = distances.min(axis=1).argsort()
            cols = distances.argmin(axis=1)[rows]
            
            used_row_indices = set()
            used_col_indices = set()
            
            # 既存オブジェクトを更新
            for (row, col) in zip(rows, cols):
                if row in used_row_indices or col in used_col_indices:
                    continue
                
                if distances[row, col] <= self.max_distance:
                    object_id = object_ids[row]
                    self.objects[object_id] = input_centroids[col]
                    self.disappeared[object_id] = 0
                    
                    # 軌跡を更新
                    self.object_trajectories[object_id].append(input_centroids[col])
                    if len(self.object_trajectories[object_id]) > 10:
                        self.object_trajectories[object_id].pop(0)
                    
                    # 入退場管理の更新
                    self._update_entry_exit_tracking(object_id, input_centroids[col], area, line_y)
                    
                    used_row_indices.add(row)
                    used_col_indices.add(col)
            
            # 未使用の行（既存オブジェクト）の消失カウントを増やす
            unused_row_indices = set(range(0, len(object_centroids))) - used_row_indices
            for row in unused_row_indices:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                
                # 最大消失フレームを超えたら削除
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            # 未使用の列（新しい重心）を新規登録
            unused_col_indices = set(range(0, len(input_centroids))) - used_col_indices
            for col in unused_col_indices:
                self.register(input_centroids[col])
        
        return self.objects.copy(), {}
    
    def _update_entry_exit_tracking(self, object_id: int, centroid: Tuple[int, int], 
                                   area: Optional[Tuple[int, int, int, int]], 
                                   line_y: Optional[int]):
        """入退場管理の更新"""
        if area:
            self._update_area_tracking(object_id, centroid, area)
        elif line_y:
            self._update_line_tracking(object_id, centroid, line_y)
    
    def _update_area_tracking(self, object_id: int, centroid: Tuple[int, int], 
                             area: Tuple[int, int, int, int]):
        """エリアベースの入退場管理"""
        x1, y1, x2, y2 = area
        cx, cy = centroid
        
        # 現在エリア内にいるかチェック
        in_area = x1 <= cx <= x2 and y1 <= cy <= y2
        
        # 前回の状態を取得
        prev_state = self.object_states[object_id]
        
        # エリア内に入った場合
        if in_area and not prev_state['in_area'] and not prev_state['entered']:
            self.enter_events.append((object_id, time.time(), centroid))
            self.object_states[object_id]['entered'] = True
            self.object_states[object_id]['in_area'] = True
        
        # エリアから出た場合
        elif not in_area and prev_state['in_area'] and prev_state['entered']:
            self.exit_events.append((object_id, time.time(), centroid))
            self.object_states[object_id]['in_area'] = False
        
        # 状態を更新
        self.object_states[object_id]['in_area'] = in_area
    
    def _update_line_tracking(self, object_id: int, centroid: Tuple[int, int], line_y: int):
        """ラインベースの入退場管理"""
        cx, cy = centroid
        trajectory = self.object_trajectories[object_id]
        
        if len(trajectory) < 2:
            return
        
        prev_centroid = trajectory[-2]
        prev_y = prev_centroid[1]
        
        # ライン交差の判定
        if prev_y <= line_y < cy:  # 上から下へ（入場）
            self.enter_events.append((object_id, time.time(), centroid))
            self.object_states[object_id]['entered'] = True
        elif prev_y >= line_y > cy:  # 下から上へ（退場）
            self.exit_events.append((object_id, time.time(), centroid))
    
    def get_entry_count(self) -> int:
        """入場者数を取得"""
        return len(self.enter_events)
    
    def get_exit_count(self) -> int:
        """退場者数を取得"""
        return len(self.exit_events)
    
    def get_current_count(self) -> int:
        """現在の在室者数を取得"""
        return self.get_entry_count() - self.get_exit_count()
    
    def get_statistics(self) -> Dict:
        """統計情報を取得"""
        return {
            'total_objects': len(self.objects),
            'entry_count': self.get_entry_count(),
            'exit_count': self.get_exit_count(),
            'current_count': self.get_current_count(),
            'disappeared_objects': len(self.disappeared)
        }
    
    def reset_events(self):
        """イベントをリセット"""
        self.enter_events = []
        self.exit_events = []
        for object_id in self.object_states:
            self.object_states[object_id]['entered'] = False