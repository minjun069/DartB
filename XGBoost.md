로지스틱 회귀와 XGBoost 비교

![image](https://github.com/user-attachments/assets/5b24fed2-b9b3-486d-b229-e94a99ef8cf4)
![image](https://github.com/user-attachments/assets/f6785790-df1e-48a3-99df-b8d6ba0e18ed)
![image](https://github.com/user-attachments/assets/79d37f0c-e4b1-4878-9775-a6b4dbc4c8b3)
![image](https://github.com/user-attachments/assets/d96d1ddc-5e3c-4804-a582-7cff2fa4b012)
![image](https://github.com/user-attachments/assets/c5ae906d-be37-436e-b31f-5b899d513bf4)
![image](https://github.com/user-attachments/assets/4d4bcb83-16e1-40e1-a27f-022858746e59)


결정트리 분류 과정 예시
![image](https://github.com/user-attachments/assets/37d4a718-fa5b-414c-a995-44857c96e9f8)
![image](https://github.com/user-attachments/assets/d5af6add-12f1-4100-baef-b04882ffaceb)

![image](https://github.com/user-attachments/assets/145d915a-263f-4105-a42e-f92b40bd653d)


XGBoost Regressor와 XGBoost Classifier의 차이
![image](https://github.com/user-attachments/assets/0d13f383-e3be-4cbc-b20b-eecc6064598e)
![image](https://github.com/user-attachments/assets/f27782d1-d3fe-402d-b0a7-f089b033e60e)

![image](https://github.com/user-attachments/assets/21573e4b-d45c-436f-a6d6-20a11cdabca1)

XGBoost의 feature importance 의미
![image](https://github.com/user-attachments/assets/7a2891ad-ac58-400d-b132-62ef79c6ff37)
![image](https://github.com/user-attachments/assets/12c5ea0c-e7f4-4c9b-bcfc-3e8d7af70bd4)
![image](https://github.com/user-attachments/assets/d2e47827-977d-4d2c-b51c-edd0d808feff)
![image](https://github.com/user-attachments/assets/762b4cc1-c6e9-464a-a288-0661d84cae68)


결과물  
1) XGBoost Regressor
features = ['EMA_9','SMA_5','SMA_10','SMA_15','SMA_30','RSI','MACD','MACD_signal']
mean_squared_error = 10699.185664349674
![image](https://github.com/user-attachments/assets/bc203c71-af6c-4eb8-8adb-15c1fea480da)

features = 101 alphas 중 80개
mean_squared_error = 31377.58524459858
![image](https://github.com/user-attachments/assets/bd34c1dd-5dab-421b-8267-6ce83200b577)

2) XGBoost Classifier (101 alphas + RSI, MACD, SMA 등)
![image](https://github.com/user-attachments/assets/cde8fab3-73ea-4054-b905-6cf1bcc0c9b8)
