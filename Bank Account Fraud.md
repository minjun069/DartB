🎯 타겟 변수  
fraud_bool: 이 거래가 사기인지 여부

💰 재무 및 신용 관련  
income: 고객의 소득 수준 (nunique=9 0.1~0.9)
credit_risk_score: 신용 위험 점수 (높을수록 위험??)
bank_months_count: 해당 은행과 거래한 개월 수 (-1은??)  
has_other_cards: 다른 신용카드를 보유하고 있는지 여부 (0, 1 인지 2 3 4 도 되는지)
proposed_credit_limit: 제안된 신용한도 (12가지 - 
intended_balcon_amount: 의도한 대출 혹은 크레딧 신청 금액 (단위??)  
payment_type: 결제 유형 (5가지- AA~AE)  

🧍 개인 정보  
customer_age: 고객 나이 (1~9) 
employment_status: 고용 상태 코드 (7가지 - CA: , CB: , CC: , CD: , CE: , CF: , CG: )  
housing_status: 주거 형태 (7가지 - BA: , BB: , BC: , BD: , BE: , BF: , BG: )
foreign_request: 외국인 여부??

📧 이메일 관련  
name_email_similarity: 이름과 이메일 주소의 유사도 (이름 기반 이메일인지 확인 가능)  
distinct_emails_4w: 4주간 사용된 고유 이메일 수 (다양한 이메일로 시도한 흔적)  
email_is_free: 이메일이 무료 서비스 제공자인지 여부
date_of_birth_distinct_emails_4w: 같은 생년월일을 가진 사람들 중 최근 4주간 사용된 고유 이메일 수 (다중 계정 시도 가능성 탐지)

🕰️ 주소 
prev_address_months_count: 이전 주소에 머문 기간 (개월 수) (-1일 경우 해당 없음?)  
current_address_months_count: 현재 주소에 머문 기간 (개월 수)  
zip_count_4w: 4주 동안 사용된 우편번호 개수 (거래 지역 다양성) 맞나??

⏱️ 시간 기반 특성
days_since_request: 계좌 신청 또는 요청 후 경과일 수 (float 형식인데??)
month: 요청 발생 월 (nunique=8)

📞 연락처
phone_home_valid: 집 전화번호가 유효한지 여부  
phone_mobile_valid: 모바일 번호가 유효한지 여부

🔢 거래
source: 계좌 요청 수 (INTERNET, TELEAPP)
velocity_6h / velocity_24h / velocity_4w: 해당 기간 동안 발생한 거래횟수 - 근데 어떻게 소수점??  
bank_branch_count_8w: 8주간 사용한 은행 지점 수 ??

💻 디바이스 및 세션 정보
session_length_in_minutes: 세션 지속 시간
device_os: 사용된 운영체제 (windows, macintosh, x11, linux, other)
keep_alive_session: 세션 유지 여부 - 정확히 무슨 말??
device_distinct_emails_8w: 동일 디바이스에서 8주간 사용된 이메일 수 맞나?? (1, 2, 0, -1 각각 무슨 의미인지 헷갈림)
device_fraud_count: 동일 디바이스에서 발생한 사기 건수 (nunique=1)


fraud vs not fraud 에서 확실히 차이나는 feature (distribution 시각화 상)
: name_email_similarity, prev_address_months_count, current_address_months_count,
케이스가 더 적은 특성 상 fraud의 분포가 더 넓은 것(분산이 큰 것)이 당연할 것이라는 생각은 했는데,
분산의 차이도 유의미한 차이로 고려해야할지? 그렇다면 기준은?


카테고리컬 데이터와 수치형 데이터 간 상관관계는 안 봐도 되나??

카이제곱 검정을 통해 카테고리 변수를 추리는 과정이 있었는데,
카테고리 변수의 클래스 불균형으로 인해 의미에 왜곡이 들어가지는 않는가?
 
extra trees classifier for feature selection
그냥 실제 모델링 과정 중에 해도 되는 작업이 아닌가
굳이 전처리 과정으로 할 이유가 있는가? 성능이 더 좋은 모델인가?
모델 성능보다 해석력, 효율성, 과적합 방지를 위한 목적이 더 크다. 고 함


SMOTE는 cross-validation 전에 적용하지 않아야 함?
validation set에 영향을 미쳐서 과적합 가능성을 높이기 때문
그래서 train data에만 추가하고, validation data는 건드리지 않는다
그럼 validation은 불균형 상태로 ?
그럼 성능 평가는 어떻게?	F1, ROC-AUC, PR Curve 등 불균형에 강한 지표 사용
+
SMOTENC: 범주형 변수를 인식하는 SMOTE (즉, one-hot encoding 없이도 사용 가능)
categorical_features는 숫자 인덱스로 주어져야 함 (컬럼 이름 아님)

