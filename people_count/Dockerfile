FROM python:3.10.10

RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    gfortran \
    libatlas-base-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 必要なファイルをコピー
COPY requirements_dev.txt .
COPY people_counter.py .
COPY people_count_stream_manager.py .
COPY video_manager.py .
COPY ../index.py .
COPY lib/ ./lib/
COPY models/ ./models/
COPY videos/ ./videos/

# Pythonの依存関係をインストール
RUN pip install --no-cache-dir -r requirements_dev.txt

# デフォルトの実行コマンド
# 動画テスト用
# CMD ["python", "-c", "import cv2; import time; cap = cv2.VideoCapture('/app/videos/test.mp4'); print('Video loaded successfully' if cap.isOpened() else 'Failed to load video'); frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); print(f'Video has {frame_count} frames'); cap.release(); time.sleep(10)"]

# index.py実行用
CMD ["python", "index.py"]