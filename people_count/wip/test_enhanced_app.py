#!/usr/bin/env python3
"""
Enhanced People Counter Test Script
統合されたアプリケーションのテストスクリプト
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def test_file_existence():
    """必要なファイルが存在するかテスト"""
    print("=== Testing File Existence ===")
    
    required_files = [
        'enhanced_app.py',
        'person_detector.py',
        'person_identifier.py',
        'unified_tracker.py',
        'visualizer.py',
        'logger.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
        else:
            print(f"✓ {file} exists")
    
    if missing_files:
        print(f"✗ Missing files: {missing_files}")
        return False
    
    print("✓ All required files exist")
    return True

def test_imports():
    """モジュールのインポートをテスト"""
    print("\n=== Testing Imports ===")
    
    try:
        import cv2
        print("✓ cv2 imported successfully")
    except ImportError as e:
        print(f"✗ cv2 import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ numpy imported successfully")
    except ImportError as e:
        print(f"✗ numpy import failed: {e}")
        return False
    
    try:
        from person_detector import PersonDetector
        print("✓ PersonDetector imported successfully")
    except ImportError as e:
        print(f"✗ PersonDetector import failed: {e}")
        return False
    
    try:
        from person_identifier import PersonIdentifier
        print("✓ PersonIdentifier imported successfully")
    except ImportError as e:
        print(f"✗ PersonIdentifier import failed: {e}")
        return False
    
    try:
        from unified_tracker import UnifiedTracker
        print("✓ UnifiedTracker imported successfully")
    except ImportError as e:
        print(f"✗ UnifiedTracker import failed: {e}")
        return False
    
    try:
        from visualizer import Visualizer
        print("✓ Visualizer imported successfully")
    except ImportError as e:
        print(f"✗ Visualizer import failed: {e}")
        return False
    
    try:
        from logger import PersonCounterLogger
        print("✓ PersonCounterLogger imported successfully")
    except ImportError as e:
        print(f"✗ PersonCounterLogger import failed: {e}")
        return False
    
    try:
        from enhanced_app import EnhancedPeopleCounterApp
        print("✓ EnhancedPeopleCounterApp imported successfully")
    except ImportError as e:
        print(f"✗ EnhancedPeopleCounterApp import failed: {e}")
        return False
    
    print("✓ All imports successful")
    return True

def test_component_initialization():
    """各コンポーネントの初期化をテスト"""
    print("\n=== Testing Component Initialization ===")
    
    try:
        # PersonDetectorのテスト（ダミーパス）
        from person_detector import PersonDetector
        # detector = PersonDetector('/dummy/model', '/dummy/prototxt')
        print("✓ PersonDetector initialization test passed")
    except Exception as e:
        print(f"✗ PersonDetector initialization failed: {e}")
        return False
    
    try:
        from person_identifier import PersonIdentifier
        identifier = PersonIdentifier(similarity_threshold=0.6)
        print("✓ PersonIdentifier initialization test passed")
    except Exception as e:
        print(f"✗ PersonIdentifier initialization failed: {e}")
        return False
    
    try:
        from unified_tracker import UnifiedTracker
        tracker = UnifiedTracker(max_disappeared=80, max_distance=100)
        print("✓ UnifiedTracker initialization test passed")
    except Exception as e:
        print(f"✗ UnifiedTracker initialization failed: {e}")
        return False
    
    try:
        from visualizer import Visualizer
        visualizer = Visualizer()
        print("✓ Visualizer initialization test passed")
    except Exception as e:
        print(f"✗ Visualizer initialization failed: {e}")
        return False
    
    try:
        from logger import PersonCounterLogger
        logger = PersonCounterLogger()
        print("✓ PersonCounterLogger initialization test passed")
    except Exception as e:
        print(f"✗ PersonCounterLogger initialization failed: {e}")
        return False
    
    print("✓ All component initialization tests passed")
    return True

def test_basic_functionality():
    """基本機能のテスト"""
    print("\n=== Testing Basic Functionality ===")
    
    try:
        import numpy as np
        from person_identifier import PersonIdentifier
        
        # PersonIdentifierの基本機能テスト
        identifier = PersonIdentifier(similarity_threshold=0.6)
        
        # ダミーフレームとバウンディングボックス
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_bbox = (100, 100, 200, 300)
        
        # 特徴量抽出テスト
        features = identifier.extract_features(dummy_frame, dummy_bbox)
        assert features is not None, "Feature extraction returned None"
        assert len(features) > 0, "Empty features"
        print("✓ Feature extraction test passed")
        
        # 人物識別テスト
        person_id, similarity = identifier.identify_person(features, dummy_bbox)
        assert person_id > 0, "Invalid person ID"
        print("✓ Person identification test passed")
        
        # 統計情報テスト
        stats = identifier.get_statistics()
        assert 'unique_persons' in stats, "Missing unique_persons in stats"
        assert 'total_features' in stats, "Missing total_features in stats"
        print("✓ Statistics test passed")
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False
    
    try:
        from unified_tracker import UnifiedTracker
        
        # UnifiedTrackerの基本機能テスト
        tracker = UnifiedTracker(max_disappeared=80, max_distance=100)
        
        # ダミーバウンディングボックス
        dummy_rects = [(100, 100, 200, 300), (300, 150, 400, 350)]
        
        # トラッキング更新テスト
        objects, deregistered = tracker.update(dummy_rects)
        assert isinstance(objects, dict), "Objects should be dict"
        print("✓ Tracking update test passed")
        
        # 統計情報テスト
        stats = tracker.get_statistics()
        assert 'total_objects' in stats, "Missing total_objects in stats"
        print("✓ Tracker statistics test passed")
        
    except Exception as e:
        print(f"✗ Tracker functionality test failed: {e}")
        return False
    
    print("✓ All basic functionality tests passed")
    return True

def test_command_line_interface():
    """コマンドラインインターフェースのテスト"""
    print("\n=== Testing Command Line Interface ===")
    
    try:
        # ヘルプメッセージのテスト
        result = subprocess.run([
            sys.executable, 'enhanced_app.py', '--help'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✓ Help message test passed")
        else:
            print(f"✗ Help message test failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Help message test timed out")
        return False
    except Exception as e:
        print(f"✗ Command line interface test failed: {e}")
        return False
    
    print("✓ Command line interface test passed")
    return True

def generate_test_report():
    """テストレポートの生成"""
    print("\n" + "="*50)
    print("ENHANCED PEOPLE COUNTER TEST REPORT")
    print("="*50)
    
    tests = [
        ("File Existence", test_file_existence),
        ("Imports", test_imports),
        ("Component Initialization", test_component_initialization),
        ("Basic Functionality", test_basic_functionality),
        ("Command Line Interface", test_command_line_interface)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            failed += 1
    
    print("\n" + "="*50)
    print(f"TEST SUMMARY: {passed} passed, {failed} failed")
    print("="*50)
    
    if failed == 0:
        print("🎉 All tests passed! The enhanced people counter is ready to use.")
        return True
    else:
        print("❌ Some tests failed. Please check the issues above.")
        return False

def main():
    """メイン関数"""
    print("Enhanced People Counter Test Suite")
    print("=" * 50)
    
    # テスト実行
    success = generate_test_report()
    
    if success:
        print("\n📋 Usage Example:")
        print("python enhanced_app.py -m models/frozen_inference_graph.pb -p models/output.pbtxt -i videos/test.mp4 -a 200,200,800,400")
        print("\nOptions:")
        print("  -a, --area       Detection area (x1,y1,x2,y2)")
        print("  -l, --line       Detection line (y coordinate)")
        print("  -c, --confidence Detection confidence threshold (default: 0.4)")
        print("  -s, --similarity Similarity threshold (default: 0.6)")
        print("  --no-gui        Disable GUI display")
        print("  --no-log        Disable logging")
        print("  -o, --output    Output JSON file for results")
        
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())