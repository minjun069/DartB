Logistic Regression의 개념  
: feature 공간 상에 초평면을 그어서 데이터를 분리하고,  
그 초평면으로부터의 거리(선형 조합 값을 sigmoid 변환한 값)를 확률로 변환하는 모델


LogisticRegression 모델의 패러미터

```python
model2 = LogisticRegression(
    penalty='l1',
    C=0.1,  # 처음은 1.0로 시작, 튜닝할 때 0.1이나 10도 테스트
    solver='liblinear',
    class_weight=None,
    max_iter=1000,
    random_state=42
)
```

1) penalty : 'l1' or 'l2'  
![image](https://github.com/user-attachments/assets/9f409f49-66b7-4ccf-86c3-d746c130f7d8)

-l1 이 feature weight를 0으로 만드는 과정  
![image](https://github.com/user-attachments/assets/c9b99253-234d-40d9-a69e-5d80e7d54012)  
![image](https://github.com/user-attachments/assets/f3fc0bb1-675d-4792-94bf-2aea7751e2f9)

-l1에 의해 0이 된 feature의 의미는?  
![image](https://github.com/user-attachments/assets/4b6de151-d8ae-4f34-bb0b-2bd537200365)  
![image](https://github.com/user-attachments/assets/542ba0b4-f5f4-4383-8d11-effc55419955)


2) class_weight = 'balanced' or None  
![image](https://github.com/user-attachments/assets/13ee374a-bb94-40b5-8990-84ecffbabfb0)

3) random state의 역할 (solver의 무작위성 결정을 위한 것)  
![image](https://github.com/user-attachments/assets/3e3de7c2-2be5-4ce2-a572-c72bfa786bd1)

4) solver 종류 별 특징  
![image](https://github.com/user-attachments/assets/a623cb21-9614-4732-b581-224cb5673800)  
![image](https://github.com/user-attachments/assets/b43f5e4b-eb15-456b-b44b-bb244662af11)  
![image](https://github.com/user-attachments/assets/9f130d41-55b6-4328-8635-ce7a417f901b)  
![image](https://github.com/user-attachments/assets/d62b621b-430d-40ee-abdf-c00d64deeb92)  
![image](https://github.com/user-attachments/assets/9b28fed8-7412-47f3-8ea2-069424b53d5a)



Feature Selection  

-RFE (Recursive Feature Elimination)
: 모델을 반복적으로 학습시키고, 중요도가 가장 낮은 feature를 하나씩 제거하는 방식  
![image](https://github.com/user-attachments/assets/8a708e1d-996c-4e22-9273-7b624da66e58)

=> 각 feature가 단독으로 target에 기여하는 중요도만 보고 제거를 결정하기 때문에,
feature 간 복합적인 상호작용에 의해 생기는 성능 향상 가능성은 고려하지 못함.

RFE vs L1   
![image](https://github.com/user-attachments/assets/11805d87-2dab-4de2-be17-c94841129fcc)  
![image](https://github.com/user-attachments/assets/fe956f55-ccf6-4335-9921-6c5dc1d85393)


parameter, feature selection 외에 추가로 고려할 수 있는 부분
![image](https://github.com/user-attachments/assets/86c80b62-357f-4596-af8b-cb31fb7fb395)  

