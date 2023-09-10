import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from PIL import Image

# データをロード
digits = load_digits()

# 画像データを数字ごとに分類
sorted_data = [digits.images[digits.target == i] for i in range(10)]

# 各数字のサンプルをトレーニング用と評価用に分割
training_data = [data[:150] for data in sorted_data]
test_data = [data[150:174] for data in sorted_data]

# トレーニング用のサンプルを連結して1つの画像を作成
concat_training_images = [np.concatenate(data, axis=0) for data in training_data]
final_training_image = np.concatenate(concat_training_images, axis=1)

# 評価用のサンプルを連結して1つの画像を作成
concat_test_images = [np.concatenate(data, axis=0) for data in test_data]
final_test_image = np.concatenate(concat_test_images, axis=1)

# トレーニング用の画像を表示・保存
plt.imshow(final_training_image, cmap="gray")
plt.axis("off")
plt.savefig("digits_training.png", bbox_inches="tight", pad_inches=0, dpi=300)
plt.show()

# 評価用の画像を表示・保存
plt.imshow(final_test_image, cmap="gray")
plt.axis("off")
plt.savefig("digits_test.png", bbox_inches="tight", pad_inches=0, dpi=300)
plt.show()
