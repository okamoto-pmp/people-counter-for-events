import cv2
import numpy as np
from typing import List, Tuple, Optional

class PersonDetector:
    """人物検出クラス"""
    
    def __init__(self, model_path: str, prototxt_path: str, confidence_threshold: float = 0.4):
        """
        Args:
            model_path: 機械学習モデルのパス
            prototxt_path: prototxtファイルのパス
            confidence_threshold: 検出の信頼度閾値
        """
        self.confidence_threshold = confidence_threshold
        self.net = self._load_model(model_path, prototxt_path)
        
    def _load_model(self, model_path: str, prototxt_path: str):
        """モデルを読み込む"""
        if model_path.endswith('.caffemodel'):
            net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        else:
            net = cv2.dnn.readNetFromTensorflow(model_path, prototxt_path)
        return net
    
    def detect_persons(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        フレーム内の人物を検出
        
        Args:
            frame: 入力フレーム
            
        Returns:
            List of bounding boxes [(x1, y1, x2, y2), ...]
        """
        height, width = frame.shape[:2]
        
        # DNNの前処理
        blob = cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False)
        self.net.setInput(blob)
        detections = self.net.forward()
        
        person_boxes = []
        
        # 検出結果を処理
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > self.confidence_threshold:
                # バウンディングボックスの座標を取得
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                x1, y1, x2, y2 = box.astype(int)
                
                # 有効な範囲内に制限
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width, x2)
                y2 = min(height, y2)
                
                if x2 > x1 and y2 > y1:  # 有効なボックスのみ
                    person_boxes.append((x1, y1, x2, y2))
        
        return person_boxes