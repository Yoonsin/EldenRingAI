import tensorflow as tf
import numpy as np
import os

# 예제 데이터 (실제 오디오 FFT 데이터를 여기에 대체)
X = np.random.rand(500, 512).astype(np.float32)  # 512차원 FFT 벡터
y = np.random.randint(0, 2, size=(500, 1)).astype(np.float32)  # 0 또는 1 (성공/실패)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(512,)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # binary output
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

# 모델 저장
model.export("parry_detector") 
print("[✔] parry_detector 모델 저장 완료")

