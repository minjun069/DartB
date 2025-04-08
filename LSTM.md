LSTM Long Short-Term Memory  
RNN의 한 종류. 순서가 중요한 모든 데이터를 처리할 수 있는 모델 (문자열, 음성, 금융, 센서, 비디오 등)  
오래된 정보도 잊지 않고 기억하려고 만든 네트워크  
보통의 RNN은 기울기 소실 문제로 오래된 정보를 잘 기억하지 못하는데, LSTM은 해결

✅데이터 종류 별 LSTM 사용 목적  
(텍스트) 문장의 감정 분석, 다음 단어 예측, 문장 단위 요약   
(금융) 주가예측_회귀, 주가상승하락여부예측_분류, 변동성 예측 


✅ LSTM 사용 시 조절 대상  
1. 입력 피처 - 종가, 고가, 저가, 거래량, 기술적지표 (MA, RSI, MACD 등), 외부 데이터 (뉴스지수, 환율, 금리 등)  
2. 타겟 - 종가, 주가상승하락 여부, 일주일 뒤 종가 평균, 최대 낙폭 등  
3. 입력 시퀀스 길이 (30일치, 90일치 등)  
4. 모델 구조와 패러미터: LSTM 레이어 개수, 은닉차원, 드랍아웃 등  
5. 전처리 방식 ex: 가격 정규화, 로그변환, 이상치 제거 등

[LSTM 구조.pdf](https://github.com/user-attachments/files/19646126/LSTM.pdf)

✅ LSTM 구조  
LSTM Layer 수: 층이 여러 개 일수록 복잡한 패턴 학습 가능 하지만 과적합 가능성  
Hidden Size: LSTM 각 층의 뉴런 수, 출력차원. 내부적으로 얼마나 많은 정보를 기억할지 결정. 크면 복잡한 패턴학습 느려지고 과적합  
Dropout: 학습 중 일부 뉴런을 랜덤으로 꺼버려 과적합하지 않게. 보통 0.2~0.5  
-노이즈가 많고 데이터 양이 적당히 있는 경우는 dropout을 통해 특정 패턴을 외우지 않도록 함.  
BatchSize: 한 번에 몇 개 데이터를 묶어서 학습할지. 크면 빠르고, 작으면 느리고 불안정  
Optimizer: 모델의 패러미터 조정 방법 결정


Batchsize  
모델은 한 번 학습할 때 하나씩 넣으면 느릴 뿐 아니라 한 데이터만 학습하고 가중치 수정 시 극단적인 값으로 인해 불안정할 수 있다.  
따라서 여러 데이터를 평균내서 업데이터하면 안정적인 학습이 가능하고 한번에 여러 데이터의 패턴을 보면서 학습하면 덜 과적합 될 수 있다.  
+  
batchsize와 learning rate를 같은 방향으로 움직이는 것이 좋다  
-batchsize가 작으면 가중치 방향이 부정확할 수 있어 learning rate가 클 경우 최적점을 지나칠 수 있다  
-batchsize가 크면 방향이 매우 정확해지기 때문에 learning rate를 크게 줘도 안정적으로 갈 수 있다.

*Linear Scaling Rule: Batch Size를 k배 늘렸으면, Learning Rate도 k배 늘려라.  
Learning Rate Warm-up: 초반에는 learning rate를 작게 시작해서, 학습이 안정되면 점점 크게 키우는 방법.  
EX: 0.0001 → 0.001 → 0.01

✅ 계산 순서  
1. 입력  
2. LSTM 1층에서 gate 계산  
3. hidden output h_1 생성  
4. Dropout 적용  
5. LSTM 2층...  
...  
6. 마지막 output을 Dense 층에 넣어 최종 예측

헷갈린 내용 정리  
-첫 레이어에서 시퀀스 내 n개의 데이터를 각각 계산하여 각각 h1, ... hn 를 생성  
-다음 레이어로 갈 때 계산된 모든 h1...hn을 넘김  
-드랍아웃에서 말하는 노드는 각 h내부의 벡터 하나하나를 말하는 것.  
-첫 레이어에서는 x_t, h_t-1을 쓰지만 다음 레이어에서는 h_t(이전 레이어), h_t-1(해당 레이어) 를 씀.

출력 직전 결정해야할 것  
-마지막 hidden state만 사용할 것인가, 전체 hidden state의 평균을 사용할 것인가.


✅LSTM 금융데이터 적용 한계점  
-금융데이터는 노이즈가 많다. (변동이 매우 랜덤하고, 뉴스 등 외부변수가 너무 크다)  
-LSTM은 긴 시퀀스를 기억하는데 한계가 있다. (길어질 수록 셀 상태 왜곡 가능)  
-과적합 위험  
=> 따라서 다른 기법과 같이 쓰는 편 (외부데이터 추가, 앙상블 모델)  
요즘은 Transformer 계열이 더 좋은 성과를 내는 경우가 많다.

✅ LSTM 주가 예측 시 주류 세팅  
-입력: 시가, 고가, 저가, 종가, 거래량 + MA5, MA10, MA20 / RSI / MACD / 볼린저 밴드 / 거시경제지표(금리, 환율, VIX, 금 등) / 뉴스 스코어  
-시퀀스 길이: 60 ~ 120일  
-모델 구조 : 2층, hidden = 128~256, droupout 0.2 ~ 0.3, batchnorm 또는 layernorm 추가, bidirectional LSTM 사용하기도 함  
-타겟: 1일 뒤 뿐 아니라 5일 뒤 10일 뒤 등 다중 스텝 예측  
-학습 설정: AdamW, ReduceLROnPlateau, 회귀 시 MSE, 분류 시 CrossEntropy, EarlyStopping: Validation loss 개선 안 될 경우

✅ 피처 중요도 파악 방법  
-피처 하나씩 제거 (Ablation Stay)  
-피처 중요도 평가 모델 이용 (LightGBM 등 )  
-SHAP 값 해석 (요즘 가장 많이 씀) (SHapley Additive exPlanations)


캐글_5d 만  
https://colab.research.google.com/drive/1iXNSGB3jdghKbYh2RDJwvedIzaQ7gEke

첫 시도  
https://colab.research.google.com/drive/1O1mgjZm0Cxp7YlPtL-dn7f_XZ9CnYJOK

전반적인 차이 정리  

|구분	|캐글	|내 코드|
|---|---|---|
|데이터 피처|	'Open','High','Low','Close', '5d_sma' |	기본 피처(Open/High/Low/Close/Volume)|
|정규화	|Low 가격 기준 MinMaxScaler (0~2 범위)|	정규화 안함|
|학습 구조	|DataLoader 사용, 배치 학습, Validation Loss 기준 모델 저장	|배치 학습 없음(또는 수동), 모델 저장 안함|
|num_layers|	1층(설정 없음) |	2층 |
|Dropout	|없음|0.2|
|Hidden Latyer Size	| 64	| 128 |
|모델 아키텍처	|LSTM hidden state → Linear → 출력	|LSTM 마지막 output → Linear → 출력|
|출력 대상	|모든 피처(Open, High, Low, Close 등) 동시 예측	|종가(Close) 하나만 예측|
|검증	|매 epoch 마다 valid loss 계산 후 모델 저장 | 검증 없음|
|loss 계산	| batch마다 쌓아서 전체 loss 계산	| 전체 X_train 한꺼번에 |
|overfitting 방지	| validation loss로 early stopping 준비 가능	| 확인 못 함|

수정 후  
https://colab.research.google.com/drive/1rSVKu_p_55b6Bq7f1rBsy-kp6TyNGcJT

수정 전  
![image](https://github.com/user-attachments/assets/09532d34-157f-4e84-ae2c-ad3ee8f27bc0)

![image](https://github.com/user-attachments/assets/b053edd4-33e6-453e-9dad-57877f7cf1d3)

수정 후  
![image](https://github.com/user-attachments/assets/9ab8ddef-b996-4a63-a5e0-1b1d721e261e)


