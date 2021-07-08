# CalibNet_with_Pytorch
CalibNet Pytorch Version 구현해보기

2019년 논문인 CalibNet(https://arxiv.org/abs/1803.08181) 은 Tensorflow로 구현이 되어있는데, Python 2.7을 사용하고 Tensorflow 1.3 버전, CUDA 8.0, cuDNN 6.0 버전을 사용하고 있다.

이 모델을 Pytorch 버전으로 바꿔서 훈련을 진행해보고, 더 나아가 모델을 개량하여 더 좋은 성능을 낼 수 있는가에 대해 테스트 해본다.

모델을 훈련하기 위해 Geforece RTX 3090 을 사용하여 진행할 예정이며, Pytorch 1.8.0 버전에서 진행한다.
