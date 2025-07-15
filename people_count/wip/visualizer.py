import cv2
import numpy as np
from typing import List, Tuple, Optional

class Visualizer:
    """バウンディングボックス描画クラス"""
    
    def __init__(self):
        self.colors = {
            'person': (0, 255, 0),    # 緑
            'unique': (255, 0, 0),    # 青
            'text': (255, 255, 255),  # 白
            'line': (0, 255, 255),    # 黄
            'area': (0, 0, 255)       # 赤
        }
        
    def draw_bounding_boxes(self, frame: np.ndarray, boxes: List[Tuple[int, int, int, int]], 
                           unique_ids: Optional[List[int]] = None) -> np.ndarray:
        """
        バウンディングボックスを描画
        
        Args:
            frame: 描画対象のフレーム
            boxes: バウンディングボックスのリスト [(x1, y1, x2, y2), ...]
            unique_ids: 各ボックスに対応するユニークID
            
        Returns:
            描画後のフレーム
        """
        output_frame = frame.copy()
        
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            # ボックスの描画
            color = self.colors['person']
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
            
            # ユニークIDがある場合はテキストを描画
            if unique_ids and i < len(unique_ids):
                unique_id = unique_ids[i]
                label = f"ID: {unique_id}"
                
                # テキストの背景を描画
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(output_frame, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), color, -1)
                
                # テキストを描画
                cv2.putText(output_frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 2)
            
            # 中心点を描画
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(output_frame, (center_x, center_y), 4, color, -1)
        
        return output_frame
    
    def draw_detection_area(self, frame: np.ndarray, area: Tuple[int, int, int, int]) -> np.ndarray:
        """
        検出エリアを描画
        
        Args:
            frame: 描画対象のフレーム
            area: エリアの座標 (x1, y1, x2, y2)
            
        Returns:
            描画後のフレーム
        """
        output_frame = frame.copy()
        cv2.rectangle(output_frame, area[:2], area[2:], self.colors['area'], 2)
        return output_frame
    
    def draw_detection_line(self, frame: np.ndarray, line_y: int) -> np.ndarray:
        """
        検出ラインを描画
        
        Args:
            frame: 描画対象のフレーム
            line_y: ラインのY座標
            
        Returns:
            描画後のフレーム
        """
        output_frame = frame.copy()
        height, width = frame.shape[:2]
        cv2.line(output_frame, (0, line_y), (width, line_y), self.colors['line'], 2)
        return output_frame
    
    def draw_info(self, frame: np.ndarray, info: dict) -> np.ndarray:
        """
        情報テキストを描画
        
        Args:
            frame: 描画対象のフレーム
            info: 表示する情報の辞書
            
        Returns:
            描画後のフレーム
        """
        output_frame = frame.copy()
        height, width = frame.shape[:2]
        
        y_offset = 30
        for key, value in info.items():
            text = f"{key}: {value}"
            cv2.putText(output_frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
            y_offset += 25
        
        return output_frame