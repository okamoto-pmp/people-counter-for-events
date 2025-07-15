# Enhanced People Counter

ReIDと入退場管理を統合した高精度人物カウンターシステム

## 概要

このシステムは、既存の`people_count/index.py`のReID機能と`people_count/people_counter.py`のエリア・ライン検出機能を統合し、より高い精度で人物の検出・識別・追跡を行います。

## 主な機能

### 1. 高精度人物検出 (`person_detector.py`)
- バウンディングボックスのサイズフィルタリング
- アスペクト比による人物らしさの判定
- ノイズ除去機能

### 2. ReID による人物識別 (`person_identifier.py`)
- HSVカラーヒストグラムによる特徴量抽出
- コサイン類似度による人物マッチング
- バウンディングボックスの安定化機能

### 3. 統合トラッキング (`unified_tracker.py`)
- エリアベースとラインベースの入退場管理
- オブジェクト追跡機能
- 入退場イベントの記録

### 4. 可視化機能 (`visualizer.py`)
- バウンディングボックス描画
- 統計情報表示
- エリア・ライン表示

### 5. ログ機能 (`logger.py`)
- 詳細なログ記録
- 人物データベースの管理
- 統計情報の出力

## 使用方法

### 基本的な使用例

```bash
# エリアベースの検出
python enhanced_app.py -m models/frozen_inference_graph.pb -p models/output.pbtxt -i videos/test.mp4 -a 200,200,800,400

# ラインベースの検出
python enhanced_app.py -m models/frozen_inference_graph.pb -p models/output.pbtxt -i videos/test.mp4 -l 300

# Webカメラからの入力
python enhanced_app.py -m models/frozen_inference_graph.pb -p models/output.pbtxt -a 200,200,800,400
```

### オプション

- `-m, --model`: 検出モデルファイルのパス（必須）
- `-p, --prototxt`: prototxtファイルのパス（必須）
- `-i, --input`: 入力ビデオファイル（省略時はWebカメラ）
- `-a, --area`: 検出エリア（x1,y1,x2,y2形式）
- `-l, --line`: 検出ライン（Y座標）
- `-c, --confidence`: 検出信頼度閾値（デフォルト: 0.4）
- `-s, --similarity`: 類似度閾値（デフォルト: 0.6）
- `--max-disappeared`: 追跡の最大消失フレーム数（デフォルト: 80）
- `--max-distance`: 追跡の最大距離（デフォルト: 100）
- `--no-gui`: GUI表示を無効化
- `--no-log`: ログ機能を無効化
- `-o, --output`: 結果出力JSONファイルのパス

### Docker環境での使用

```bash
# エリアベースの検出
docker compose run people_count python wip/enhanced_app.py -m models/ssd_mobilenet_v2_coco/frozen_inference_graph.pb -p models/ssd_mobilenet_v2_coco/output.pbtxt -i videos/test.mp4 -a 200,200,800,400

# ラインベースの検出
docker compose run people_count python wip/enhanced_app.py -m models/ssd_mobilenet_v2_coco/frozen_inference_graph.pb -p models/ssd_mobilenet_v2_coco/output.pbtxt -i videos/test.mp4 -l 300
```

## 改善点

### 検出精度の向上
1. **サイズフィルタリング**: 最小サイズ（30x50）未満のバウンディングボックスを除外
2. **アスペクト比フィルタリング**: 人物らしくない形状（比率1.2～5.0以外）を除外
3. **バウンディングボックス安定化**: 過去の履歴を使用してサイズ変動を抑制

### 識別精度の向上
1. **より詳細な特徴量**: 上半身・下半身別のHSVヒストグラム
2. **適応的閾値**: 類似度閾値の調整が可能
3. **特徴量データベース**: 人物ごとに最大5つの特徴量を保存

### 追跡精度の向上
1. **統合トラッキング**: ReIDとオブジェクト追跡の組み合わせ
2. **エリア・ライン対応**: 両方の検出方式に対応
3. **入退場管理**: 正確な入退場イベントの記録

## 出力形式

### コンソール出力
```
Frame 30: Detected=1, Unique=1, Tracked=1, Enter=1, Exit=0, Current=1
  Average similarity: 0.808
```

### JSONファイル出力
```json
{
  "timestamp": "2025-01-15T10:40:03.123456",
  "total_frames": 1034,
  "unique_persons": 3,
  "entry_events": 3,
  "exit_events": 1,
  "current_count": 2,
  "person_database": {
    "1": {
      "feature_count": 5,
      "total_appearances": 277
    }
  }
}
```

## ログファイル

実行時に以下のログファイルが生成されます：
- `logs/detail_YYYYMMDD_HHMMSS.log`: 詳細ログ
- `logs/debug_YYYYMMDD_HHMMSS.log`: デバッグログ
- `logs/persons_YYYYMMDD_HHMMSS.json`: 人物データベース

## テスト

システムの動作確認には以下のテストスクリプトを使用します：

```bash
python test_enhanced_app.py
```

## 既存システムとの比較

| 機能 | 従来のindex.py | 従来のpeople_counter.py | Enhanced App |
|------|----------------|-------------------------|--------------|
| ReID | ✓ | - | ✓ |
| エリア検出 | - | ✓ | ✓ |
| ライン検出 | - | ✓ | ✓ |
| 統合トラッキング | - | - | ✓ |
| 詳細ログ | - | - | ✓ |
| 高精度フィルタリング | - | - | ✓ |
| JSON出力 | - | - | ✓ |

## 注意点

1. **エリアとラインの同時指定**: 両方を同時に指定することはできません
2. **モデルファイル**: 事前にモデルファイル（`.pb`）とprototxtファイル（`.pbtxt`）を用意する必要があります
3. **パフォーマンス**: ReIDと追跡の両方を行うため、処理負荷は高くなります

## トラブルシューティング

### よくある問題

1. **"Model file not found"エラー**: モデルファイルのパスを確認してください
2. **"Cannot open video source"エラー**: ビデオファイルの存在またはカメラの接続を確認してください
3. **低い検出精度**: 信頼度閾値（`-c`）を調整してください
4. **過剰な識別**: 類似度閾値（`-s`）を調整してください

### パフォーマンス調整

- 検出間隔を調整: `enhanced_app.py`内の`detection_interval`を変更
- 追跡パラメータの調整: `--max-disappeared`と`--max-distance`を調整
- ログ機能の無効化: `--no-log`オプションを使用