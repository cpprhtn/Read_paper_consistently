Automatic Modulation Classification: A Deep Learning Enabled Approach

초록 : 자동 변조 분류는 ML과 기능 기반 AMC로 분류할 수 있다. 하지만 ML(Maximum Likehood)기반 AMC는 높은 계산 복잡성으로 인해 배포가 어려우며 수동으로 추출한 기능에는 전문적인 기술 지식이 필요하다.

따라서 SNR과 함께 긴 신호 속도 관찰 시퀀스에서 특징을 자동으로 추출하는 CNN기반 AMC가 제안되었다.

CNN-AMC는 반송파 위상 오프셋 및 SNR에 대한 추정 오류에 대한 특정 견고성을 가지고 있다.

또한 최적의 ML-AMC에 더 가까운 근사치를 얻을 수 있다.

CNN model

(CNN*5 -> dense)+dense -> concat -> dense -> output

시뮬레이션 코드에 대한 주소
https://github.com/mengxiaomao/CNN_AMC

전이학습도 사용함. 2가지 방법에 따라 진행됨.

1. 다양한 변조 방식

2. 다른 환경




### 코드 리뷰
python 3.5
tensorflow version 1.15
using keras
using matlab

Conv 1D, average-pooling, Dense

```py
y = Input(shape=(signal_size,2), dtype='float32', name='Input')

x = Conv1D(12, 3, padding='same', activation='relu')(y)
x = Conv1D(12, 3, padding='same', activation='relu')(x)
x = AveragePooling1D(pool_size=2)(x)
x = Conv1D(24, 3, padding='same', activation='relu')(x)
x = AveragePooling1D(pool_size=2)(x)
x = Conv1D(24, 3, padding='same', activation='relu')(x)
x = AveragePooling1D(pool_size=2)(x)
x = Conv1D(32, 3, padding='same', activation='relu')(x)
x = AveragePooling1D(pool_size=2)(x)
x = Flatten()(x)
x = Dense(256, activation='relu')(x)

snr = Input(shape=(1,), dtype='float32', name='snr')
n = Dense(10, activation='relu')(snr)
m = keras.layers.concatenate([x, n], axis=-1)
f = Dense(L_1, activation='relu')(m)
p = Dense(class_num, activation='softmax', name='p')(f)
```

check-point callback function 사용

전이 학습을 위해 fit를 두번 사용.

데이터에 따라 파일을 분류해둠. but 모델은 모두 동일.