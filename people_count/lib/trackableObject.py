# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

class TrackableObject:
	def __init__(self, objectID, centroid):
		# オブジェクトIDと中心座標の履歴を定義
		self.objectID = objectID
		self.centroids = [centroid]
		# すでにカウントされたかどうか保存
		self.counted = False
		self.enter_counted = False
		self.leave_counted = False
		# ReID用の特徴量を保存
		self.feature_vectors = []
		self.unique_person_id = None  # 後処理で付与されるユニークID
		self.crop_images = []  # 人物の切り抜き画像を保存（デバッグ用）