import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers.legacy import SGD
import numpy as np

# XOR 예제
x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]])

# 신경망 모델 구성
model = Sequential([
    Dense(2, input_dim=2, activation='sigmoid'),  # 입력층 + 은닉층
    Dense(1, activation='sigmoid')  # 출력층
])

# 모델 컴파일
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.07), loss='binary_crossentropy',
              metrics=['accuracy'])

# 1000에폭 7번 학습으로 총 7000에폭 학습
for step in range(7):
    # 모델 학습
    model.fit(x_train, y_train, epochs=1000, verbose=0)

    # 모델 평가
    loss, accuracy = model.evaluate(x_train, y_train, verbose=0)
    print(f'단계: {step + 1}, 에폭 수: {(step + 1) * 1000}, 손실 함수 값: {loss:.4f}, 정확도: {accuracy:.2f}')

# 예측
predictions = model.predict(x_train)
print(predictions)

