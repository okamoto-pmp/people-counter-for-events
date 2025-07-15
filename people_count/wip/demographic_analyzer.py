import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List
from enum import Enum

class Gender(Enum):
    MALE = "male"
    FEMALE = "female"
    UNKNOWN = "unknown"

class AgeGroup(Enum):
    CHILD = "child"      # 0-12
    TEEN = "teen"        # 13-19
    YOUNG_ADULT = "young_adult"  # 20-35
    MIDDLE_AGED = "middle_aged"  # 36-55
    SENIOR = "senior"    # 56+
    UNKNOWN = "unknown"

class DemographicAnalyzer:
    """人物の性別・年齢推定を行うクラス"""
    
    def __init__(self):
        """
        デモグラフィック分析器の初期化
        """
        self.gender_net = None
        self.age_net = None
        self.face_net = None
        self._load_models()
    
    def _load_models(self):
        """
        事前学習済みモデルの読み込み
        OpenCVのDNNモジュールを使用
        """
        try:
            # 性別推定モデル
            self.gender_net = cv2.dnn.readNetFromCaffe(
                '/app/models/gender_net.prototxt',
                '/app/models/gender_net.caffemodel'
            )
            
            # 年齢推定モデル
            self.age_net = cv2.dnn.readNetFromCaffe(
                '/app/models/age_net.prototxt', 
                '/app/models/age_net.caffemodel'
            )
            
            # 顔検出モデル
            self.face_net = cv2.dnn.readNetFromTensorflow(
                '/app/models/opencv_face_detector_uint8.pb',
                '/app/models/opencv_face_detector.pbtxt'
            )
            
        except Exception as e:
            print(f"Warning: Could not load demographic models: {e}")
            print("Using fallback heuristic-based analysis")
    
    def _detect_faces(self, frame: np.ndarray, person_bbox: Tuple[int, int, int, int]) -> List[Tuple[int, int, int, int]]:
        """
        人物のバウンディングボックス内で顔を検出
        
        Args:
            frame: 入力フレーム
            person_bbox: 人物のバウンディングボックス (x1, y1, x2, y2)
            
        Returns:
            検出された顔のバウンディングボックスのリスト
        """
        if self.face_net is None:
            return []
        
        x1, y1, x2, y2 = person_bbox
        person_roi = frame[y1:y2, x1:x2]
        
        if person_roi.size == 0:
            return []
        
        # 顔検出用のblob作成
        blob = cv2.dnn.blobFromImage(person_roi, 1.0, (300, 300), [104, 117, 123])
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        faces = []
        h, w = person_roi.shape[:2]
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # 信頼度閾値
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                face_x1, face_y1, face_x2, face_y2 = box.astype(int)
                
                # 元の座標系に変換
                face_x1 += x1
                face_y1 += y1
                face_x2 += x1
                face_y2 += y1
                
                faces.append((face_x1, face_y1, face_x2, face_y2))
        
        return faces
    
    def _predict_gender_from_face(self, face_roi: np.ndarray) -> Tuple[Gender, float]:
        """
        顔画像から性別を推定
        
        Args:
            face_roi: 顔の領域
            
        Returns:
            (性別, 信頼度)
        """
        if self.gender_net is None:
            return self._heuristic_gender_analysis(face_roi)
        
        try:
            # 顔画像の前処理
            blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), 
                                        (78.4263377603, 87.7689143744, 114.895847746), 
                                        swapRB=False)
            self.gender_net.setInput(blob)
            gender_preds = self.gender_net.forward()
            
            # 性別判定
            gender_confidence = gender_preds[0].max()
            gender_idx = gender_preds[0].argmax()
            
            gender = Gender.MALE if gender_idx == 0 else Gender.FEMALE
            
            return gender, gender_confidence
            
        except Exception as e:
            print(f"Gender prediction failed: {e}")
            return self._heuristic_gender_analysis(face_roi)
    
    def _predict_age_from_face(self, face_roi: np.ndarray) -> Tuple[AgeGroup, float]:
        """
        顔画像から年齢を推定
        
        Args:
            face_roi: 顔の領域
            
        Returns:
            (年齢グループ, 信頼度)
        """
        if self.age_net is None:
            return self._heuristic_age_analysis(face_roi)
        
        try:
            # 顔画像の前処理
            blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), 
                                        (78.4263377603, 87.7689143744, 114.895847746), 
                                        swapRB=False)
            self.age_net.setInput(blob)
            age_preds = self.age_net.forward()
            
            # 年齢判定
            age_confidence = age_preds[0].max()
            age_idx = age_preds[0].argmax()
            
            # 年齢グループのマッピング
            age_groups = [
                AgeGroup.CHILD,      # 0-2
                AgeGroup.CHILD,      # 4-6
                AgeGroup.CHILD,      # 8-12
                AgeGroup.TEEN,       # 15-20
                AgeGroup.YOUNG_ADULT,# 25-32
                AgeGroup.MIDDLE_AGED,# 38-43
                AgeGroup.MIDDLE_AGED,# 48-53
                AgeGroup.SENIOR      # 60-100
            ]
            
            if age_idx < len(age_groups):
                age_group = age_groups[age_idx]
            else:
                age_group = AgeGroup.UNKNOWN
            
            return age_group, age_confidence
            
        except Exception as e:
            print(f"Age prediction failed: {e}")
            return self._heuristic_age_analysis(face_roi)
    
    def _heuristic_gender_analysis(self, face_roi: np.ndarray) -> Tuple[Gender, float]:
        """
        ヒューリスティックな性別分析（モデルが利用できない場合）
        髪の長さ、肌の色調、輪郭の特徴を使用
        """
        if face_roi.size == 0:
            return Gender.UNKNOWN, 0.0
        
        # HSV変換
        hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        
        # 髪の検出（上部領域）
        h, w = face_roi.shape[:2]
        hair_region = hsv[:h//3, :]
        
        # 髪の暗さ（V値の平均）
        hair_darkness = np.mean(hair_region[:, :, 2])
        
        # 肌の色調分析（中央領域）
        skin_region = hsv[h//3:2*h//3, w//4:3*w//4]
        skin_saturation = np.mean(skin_region[:, :, 1])
        
        # 簡単なヒューリスティック
        gender_score = 0.0
        
        # 髪の長さ推定（エッジ検出）
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        long_hair_indicator = np.sum(edges[:h//2, :]) / (h * w)
        
        if long_hair_indicator > 0.1:
            gender_score += 0.3  # 女性寄り
        
        if skin_saturation > 50:
            gender_score += 0.2  # 女性寄り
        
        # 簡単な判定
        if gender_score > 0.3:
            return Gender.FEMALE, min(gender_score, 0.8)
        elif gender_score < 0.1:
            return Gender.MALE, min(1.0 - gender_score, 0.8)
        else:
            return Gender.UNKNOWN, 0.5
    
    def _heuristic_age_analysis(self, face_roi: np.ndarray) -> Tuple[AgeGroup, float]:
        """
        ヒューリスティックな年齢分析（モデルが利用できない場合）
        顔のサイズ、シワ、肌の質感を使用
        """
        if face_roi.size == 0:
            return AgeGroup.UNKNOWN, 0.0
        
        h, w = face_roi.shape[:2]
        
        # 顔のサイズ（子供は相対的に小さい）
        face_size = h * w
        
        # グレースケール変換
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # シワ検出（高周波成分）
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        wrinkle_score = np.var(laplacian)
        
        # 肌の質感分析
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        texture_score = np.var(gray - blur)
        
        # 年齢推定ヒューリスティック
        age_score = 0.0
        
        # 顔のサイズ
        if face_size < 5000:
            age_score += 0.4  # 子供寄り
        elif face_size > 15000:
            age_score -= 0.2  # 大人寄り
        
        # シワスコア
        if wrinkle_score > 100:
            age_score -= 0.3  # 高齢寄り
        
        # 質感スコア
        if texture_score < 50:
            age_score += 0.2  # 若い寄り
        
        # 年齢グループ判定
        if age_score > 0.3:
            return AgeGroup.CHILD, min(age_score, 0.8)
        elif age_score > 0.0:
            return AgeGroup.TEEN, 0.6
        elif age_score > -0.2:
            return AgeGroup.YOUNG_ADULT, 0.7
        elif age_score > -0.4:
            return AgeGroup.MIDDLE_AGED, 0.6
        else:
            return AgeGroup.SENIOR, min(abs(age_score), 0.8)
    
    def analyze_demographics(self, frame: np.ndarray, person_bbox: Tuple[int, int, int, int]) -> Dict:
        """
        人物の性別・年齢を分析
        
        Args:
            frame: 入力フレーム
            person_bbox: 人物のバウンディングボックス (x1, y1, x2, y2)
            
        Returns:
            人口統計情報の辞書
        """
        result = {
            'gender': Gender.UNKNOWN,
            'gender_confidence': 0.0,
            'age_group': AgeGroup.UNKNOWN,
            'age_confidence': 0.0,
            'face_detected': False
        }
        
        # 顔検出
        faces = self._detect_faces(frame, person_bbox)
        
        if faces:
            result['face_detected'] = True
            
            # 最も大きな顔を使用
            largest_face = max(faces, key=lambda f: (f[2]-f[0]) * (f[3]-f[1]))
            fx1, fy1, fx2, fy2 = largest_face
            
            # 顔の領域を切り出し
            face_roi = frame[fy1:fy2, fx1:fx2]
            
            if face_roi.size > 0:
                # 性別推定
                gender, gender_conf = self._predict_gender_from_face(face_roi)
                result['gender'] = gender
                result['gender_confidence'] = gender_conf
                
                # 年齢推定
                age_group, age_conf = self._predict_age_from_face(face_roi)
                result['age_group'] = age_group
                result['age_confidence'] = age_conf
        else:
            # 顔が検出できない場合は全身から推定
            x1, y1, x2, y2 = person_bbox
            person_roi = frame[y1:y2, x1:x2]
            
            if person_roi.size > 0:
                # 身体全体からの簡単な推定
                gender, gender_conf = self._analyze_body_shape(person_roi)
                result['gender'] = gender
                result['gender_confidence'] = gender_conf
                
                # 身長からの年齢推定
                age_group, age_conf = self._analyze_body_size(person_roi, person_bbox)
                result['age_group'] = age_group
                result['age_confidence'] = age_conf
        
        return result
    
    def _analyze_body_shape(self, person_roi: np.ndarray) -> Tuple[Gender, float]:
        """
        身体の形状から性別を推定
        """
        if person_roi.size == 0:
            return Gender.UNKNOWN, 0.0
        
        h, w = person_roi.shape[:2]
        
        # 上半身と下半身の幅比率
        upper_body = person_roi[:h//2, :]
        lower_body = person_roi[h//2:, :]
        
        # エッジ検出
        gray = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # 肩幅の推定
        shoulder_region = edges[:h//3, :]
        shoulder_width = np.sum(shoulder_region > 0, axis=1).max()
        
        # 腰幅の推定
        hip_region = edges[h//2:2*h//3, :]
        hip_width = np.sum(hip_region > 0, axis=1).max()
        
        # 簡単な判定
        if shoulder_width > 0 and hip_width > 0:
            ratio = shoulder_width / hip_width
            if ratio > 1.2:
                return Gender.MALE, 0.6
            elif ratio < 0.9:
                return Gender.FEMALE, 0.6
        
        return Gender.UNKNOWN, 0.3
    
    def _analyze_body_size(self, person_roi: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[AgeGroup, float]:
        """
        身体のサイズから年齢グループを推定
        """
        x1, y1, x2, y2 = bbox
        height = y2 - y1
        width = x2 - x1
        
        # 身長の推定（簡単な判定）
        if height < 150:
            return AgeGroup.CHILD, 0.7
        elif height < 200:
            return AgeGroup.TEEN, 0.5
        elif height < 300:
            return AgeGroup.YOUNG_ADULT, 0.6
        else:
            return AgeGroup.MIDDLE_AGED, 0.5
    
    def format_demographics(self, demographics: Dict) -> str:
        """
        人口統計情報を文字列形式でフォーマット
        """
        gender_str = demographics['gender'].value if demographics['gender'] != Gender.UNKNOWN else "?"
        age_str = demographics['age_group'].value if demographics['age_group'] != AgeGroup.UNKNOWN else "?"
        
        confidence_str = ""
        if demographics['gender_confidence'] > 0.5:
            confidence_str += f"G:{demographics['gender_confidence']:.2f}"
        if demographics['age_confidence'] > 0.5:
            if confidence_str:
                confidence_str += " "
            confidence_str += f"A:{demographics['age_confidence']:.2f}"
        
        result = f"{gender_str}/{age_str}"
        if confidence_str:
            result += f" ({confidence_str})"
        
        return result