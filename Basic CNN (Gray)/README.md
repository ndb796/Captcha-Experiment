## CNN Captcha Solver (Gray)

### Experiment 1 (Basic Attack)

* 숫자 이미지(0 ~ 9)를 이용하여 4자리 Captcha를 순열마다 3개씩 생성 (15,120 개 / Test: 5040 개)
* Output 크기가 40인 CNN 을 이용하여 학습 진행 (Batch Size: 64, Epoch: 30)
* 학습 결과, Captcha Attack Success Rate: 70%

### Experiment 2 (Basic Attack: Benchmark)

* 숫자 이미지(0 ~ 9)를 이용하여 4자리 Captcha를 순열마다 6개씩 생성 (30,240 개 / Test: 5040 개)
* Output 크기가 40인 CNN 을 이용하여 학습 진행 (Batch Size: 64, Epoch: 30)
* 학습 결과, Captcha Attack Success Rate: 91%

### Experiment 3 (FGSM Defense)

* 각 Test 이미지에 대하여 FGSM으로 Perturbed Image 생성
* Epsilon = 0.05
* Captcha Attack Success Rate = 32.5%
* 다만 Gray 전처리 된 Image에 대하여 FGSM을 수행