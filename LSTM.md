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
-시퀀스 길이: 60~120일  
-모델 구조 : 2층, hidden = 128~256, droupout 0.2~0.3, batchnorm 또는 layernorm 추가, bidirectional LSTM 사용하기도 함  
-타겟: 1일 뒤 뿐 아니라 5일 뒤 10일 뒤 등 다중 스텝 예측  
-학습 설정: AdamW, ReduceLROnPlateau, 회귀 시 MSE, 분류 시 CrossEntropy, EarlyStopping: Validation loss 개선 안 될 경우

✅ 피처 중요도 파악 방법  
-피처 하나씩 제거 (Ablation Stay)  
-피처 중요도 평가 모델 이용 (LightGBM 등 )  
-SHAP 값 해석 (요즘 가장 많이 씀) (SHapley Additive exPlanations)

```python
import pandas as pd
import numpy as np

price = pd.read_csv("C:\\Users\\minju\\OneDrive\\바탕 화면\\다트비\\캐금스\\prices.csv")
price_wltw = price[price['symbol']=='WLTW']
price_wltw = price_wltw[['open','close','low','high','volume']]
features = ['open', 'close', 'low', 'high', 'volume']
price_wltw = price_wltw[features].values

X = []
y = []

seq_length = 30

for i in range(len(price_wltw) - seq_length):
    X.append(price_wltw[i:i+seq_length])
    y.append(price_wltw[i+seq_length][1])

X = np.array(X)
y = np.array(y)

X_tensor = torch.tensor(X, dtype=torch.float32)  # 입력
y_tensor = torch.tensor(y, dtype=torch.float32)  # 정답
y_tensor = y_tensor.unsqueeze(1)

# 가장 간단한 LSTM 모델
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layer):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layer, batch_first=True)  # batch_first=True로 입력을 (batch, seq_len, input_size)로
        self.fc = nn.Linear(hidden_size, output_size)  # LSTM의 출력 h_t를 받아서 최종 결과 출력
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)  # x: (batch_size, seq_len, input_size)
        # out: (batch_size, seq_len, hidden_size)
        # 보통 마지막 timestep의 출력을 사용
        out = out[:, -1, :]  # 마지막 시점(timestep)의 hidden state만 가져오기
        out = self.dropout(out)  # ⬅️ Dropout 적용 (hidden feature에)
        out = self.fc(out)   # 최종 출력
        return out

# 하이퍼파라미터 설정
input_size = 5     # 하루 입력 데이터 피처 수 (예: 시가, 고가, 저가, 종가, 거래량)
hidden_size = 128   # LSTM hidden size
output_size = 1    # 예측할 결과 크기 (예: 내일 종가)
num_layers = 2
# 모델 생성
model = SimpleLSTM(input_size, hidden_size, output_size, num_layers)

# 전체 데이터 크기
n_samples = X.shape[0]

# 80%를 train, 20%를 test로
train_size = int(n_samples * 0.8)

X_train = X_tensor[:train_size]
y_train = y_tensor[:train_size]
X_test = X_tensor[train_size:]
y_test = y_tensor[train_size:]

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

num_epochs = 500

for epoch in range(num_epochs):
    model.train()
    # 1. 모델에 입력
    output = model(X_train)
    # 2. loss 계산
    loss = criterion(output, y_train)
    # 3. optimizer 초기화 (gradient를 0으로 초기화)
    optimizer.zero_grad()
    # 4. 역전파
    loss.backward()
    # 5. 가중치 업데이트
    optimizer.step()
    # 6. 출력
    if (epoch + 1) % 25 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 테스트 모드로 전환
model.eval()

# 테스트 데이터로 예측
with torch.no_grad():  # 테스트할 때는 gradient 계산 X
    test_output = model(X_test)
    test_loss = criterion(test_output, y_test)

print(f"Test Loss: {test_loss.item():.4f}")

import matplotlib.pyplot as plt

# 1. 모델 예측
model.eval()
with torch.no_grad():
    y_pred = model(X_test).squeeze().cpu().numpy()  # (batch_size, 1) → (batch_size,)

# 2. 실제 정답
y_true = y_test.squeeze().cpu().numpy()

# 3. 그리기
plt.figure(figsize=(12,6))
plt.plot(y_true, label='True Price')
plt.plot(y_pred, label='Predicted Price')
plt.title('Predicted vs True Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
```
![image](https://github.com/user-attachments/assets/23aae696-c4f5-48e1-bf8a-f2fe662a484c)

![image](https://github.com/user-attachments/assets/8be14458-15a0-4d84-8a78-0eca40136395)

![image](https://github.com/user-attachments/assets/95a77428-7c04-42d5-808f-9b557a50eb4a)

![image](https://github.com/user-attachments/assets/736d7260-946b-4ece-88e5-97c6a0a7c1e2)

![image](https://github.com/user-attachments/assets/09532d34-157f-4e84-ae2c-ad3ee8f27bc0)

![image](https://github.com/user-attachments/assets/b053edd4-33e6-453e-9dad-57877f7cf1d3)

![image](https://github.com/user-attachments/assets/9ab8ddef-b996-4a63-a5e0-1b1d721e261e)


