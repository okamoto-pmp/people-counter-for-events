# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import cv2
import numpy as np
from typing import Optional, Tuple

class ReIDFeatureExtractor:
    """
    Person Re-identification用の特徴量抽出器
    OpenCVを使用してCPUで動作する軽量な実装
    """
    
    def __init__(self):
        # Color histogram用のパラメータ
        self.hist_bins = 16
        self.hist_ranges = [0, 256]
        
        # HOG特徴量用のパラメータ
        self.hog = cv2.HOGDescriptor()
        
    def extract_features(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        画像から人物の特徴量を抽出する
        
        Args:
            image: 入力画像
            bbox: 人物のバウンディングボックス (x1, y1, x2, y2)
            
        Returns:
            特徴量ベクトル（正規化済み）
        """
        try:
            # 人物領域を切り抜き
            x1, y1, x2, y2 = bbox
            person_crop = image[y1:y2, x1:x2]
            
            if person_crop.size == 0:
                return None
                
            # サイズを正規化 (128x256が一般的)
            person_crop = cv2.resize(person_crop, (64, 128))
            
            # 複数の特徴量を組み合わせる
            color_features = self._extract_color_features(person_crop)
            texture_features = self._extract_texture_features(person_crop)
            
            # 特徴量を結合
            features = np.concatenate([color_features, texture_features])
            
            # L2正規化
            features = features / (np.linalg.norm(features) + 1e-6)
            
            return features
            
        except Exception as e:
            print(f"特徴量抽出エラー: {e}")
            return None
    
    def _extract_color_features(self, image: np.ndarray) -> np.ndarray:
        """
        カラーヒストグラム特徴量を抽出
        """
        # HSV色空間に変換
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 上半身と下半身に分割してヒストグラムを計算
        h, w = image.shape[:2]
        upper_half = hsv[:h//2, :]
        lower_half = hsv[h//2:, :]
        
        # HSVヒストグラムを計算
        hist_upper = []
        hist_lower = []
        
        for i in range(3):  # H, S, V
            hist_u = cv2.calcHist([upper_half], [i], None, [self.hist_bins], self.hist_ranges)
            hist_l = cv2.calcHist([lower_half], [i], None, [self.hist_bins], self.hist_ranges)
            hist_upper.extend(hist_u.flatten())
            hist_lower.extend(hist_l.flatten())
        
        color_features = np.array(hist_upper + hist_lower)
        return color_features
    
    def _extract_texture_features(self, image: np.ndarray) -> np.ndarray:
        """
        テクスチャ特徴量を抽出（簡易LBP）
        """
        # グレースケール変換
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 簡易LBP（Local Binary Pattern）
        lbp = self._calculate_lbp(gray)
        
        # LBPヒストグラム
        hist = cv2.calcHist([lbp], [0], None, [256], [0, 256])
        
        return hist.flatten()
    
    def _calculate_lbp(self, image: np.ndarray) -> np.ndarray:
        """
        簡易Local Binary Pattern計算
        """
        h, w = image.shape
        lbp = np.zeros((h-2, w-2), dtype=np.uint8)
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                center = image[i, j]
                code = 0
                
                # 8近傍の値を比較
                neighbors = [
                    image[i-1, j-1], image[i-1, j], image[i-1, j+1],
                    image[i, j+1], image[i+1, j+1], image[i+1, j],
                    image[i+1, j-1], image[i, j-1]
                ]
                
                for k, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        code += 2**k
                
                lbp[i-1, j-1] = code
        
        return lbp

def compute_similarity(feature1: np.ndarray, feature2: np.ndarray) -> float:
    """
    2つの特徴量間の類似度を計算（コサイン類似度）
    
    Args:
        feature1: 特徴量1
        feature2: 特徴量2
        
    Returns:
        類似度スコア (0.0 ~ 1.0)
    """
    if feature1 is None or feature2 is None:
        return 0.0
    
    # コサイン類似度計算
    dot_product = np.dot(feature1, feature2)
    norm_product = np.linalg.norm(feature1) * np.linalg.norm(feature2)
    
    if norm_product == 0:
        return 0.0
    
    similarity = dot_product / norm_product
    return max(0.0, similarity)  # 負の値を0にクリップ