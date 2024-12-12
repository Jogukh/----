# %% Import Modules
import re
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from konlpy.tag import Okt
from xgboost import DMatrix, cv, train
import numpy as np

# %% Load Data
# CSV 파일 인코딩 문제 해결
try:
    df = pd.read_csv('국토안전관리원_건설안전사고사례_20240630.csv', encoding='utf-8')
except UnicodeDecodeError:
    try:
        df = pd.read_csv('국토안전관리원_건설안전사고사례_20240630.csv', encoding='cp949')
    except UnicodeDecodeError:
        raise ValueError("파일 인코딩이 지원되지 않습니다. utf-8 또는 cp949로 저장된 파일인지 확인하세요.")

df.dropna(inplace=True)

# 사고 유형 단순화
df['인적사고'] = df['인적사고종류'].replace({
    '넘어짐(기타)': '넘어짐',
    '넘어짐(미끄러짐)': '넘어짐',
    '넘어짐(물체에 걸림)': '넘어짐',
    '떨어짐(분류불능)': '떨어짐',
    '떨어짐(2미터 미만)': '떨어짐',
    '떨어짐(2미터 이상 ~ 3미터 미만)': '떨어짐',
    '떨어짐(3미터 이상 ~ 5미터 미만)': '떨어짐',
    '떨어짐(5미터 이상 ~ 10미터 미만)': '떨어짐',
    '떨어짐(10미터 이상)': '떨어짐',
    '분류불능': '기타'
})

# 불필요한 사고 유형 제외
excluded_types = ['깔림', '없음', '질병', '찔림', '화상', '교통사고', '감전', '질식', '기타', '미입력']
df = df[~df['인적사고종류'].isin(excluded_types)]

# 특수문자 제거
X = df['사고경위'].apply(lambda x: re.sub(r'[-=+,#/\?:^$.@*"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', ' ', x))
y = df['인적사고']

# %% Data Splitting
# 데이터 분리 (훈련, 테스트, 검증 데이터)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print(f"훈련 데이터 크기: {len(X_train)}, 테스트 데이터 크기: {len(X_test)}, 검증 데이터 크기: {len(X_val)}")

# %% TF-IDF 적용
def custom_tokenizer(text):
    okt = Okt()
    tokens = okt.pos(text, stem=True)
    return [token[0] for token in tokens if token[1] in ("Noun", "Verb")]

tfidf_vect = TfidfVectorizer(tokenizer=custom_tokenizer, min_df=2, max_df=2000)
X_train_tf = tfidf_vect.fit_transform(X_train)
X_test_tf = tfidf_vect.transform(X_test)
X_val_tf = tfidf_vect.transform(X_val)

# Label Encoding
le = LabelEncoder()
y_train_le = le.fit_transform(y_train)
y_test_le = le.transform(y_test)
y_val_le = le.transform(y_val)

# %% XGBoost Training
params = {
    "objective": "multi:softprob",
    "num_class": len(le.classes_),
    "random_state": 42,
    "tree_method": "hist",
    "learning_rate": 0.1,  # 학습률 초기값
    "max_depth": 6         # 초기 최대 깊이
}

dtrain = DMatrix(X_train_tf, label=y_train_le)
dtest = DMatrix(X_test_tf, label=y_test_le)
dval = DMatrix(X_val_tf, label=y_val_le)

cv_results = cv(
    params=params,
    dtrain=dtrain,
    num_boost_round=500,
    nfold=5,
    metrics="mlogloss",
    early_stopping_rounds=10,
    verbose_eval=True
)

best_model = train(params, dtrain, num_boost_round=cv_results.shape[0])

# %% First Prediction and Parameter Adjustment
y_test_pred = best_model.predict(dtest).argmax(axis=1)
y_test_pred_top2 = np.argsort(-best_model.predict(dtest), axis=1)[:, :2]

second_correct = 0
adjustment_needed = False

for i in range(len(y_test_le)):
    if y_test_pred[i] != y_test_le[i] and y_test_pred_top2[i, 1] == y_test_le[i]:
        second_correct += 1
        adjustment_needed = True

# 재학습 조건 확인
if adjustment_needed:
    second_correct_rate = second_correct / len(y_test_le)
    print(f"두 번째 예측이 실제와 일치한 비율: {second_correct_rate:.3f}")

    # 파라미터 조정
    params["learning_rate"] = max(0.01, params["learning_rate"] * 0.8)  # 학습률 감소
    params["max_depth"] = min(10, params["max_depth"] + 1)  # 최대 깊이 증가
    f_accuracy = accuracy_score(y_test_le, y_test_pred)
    print(f'1차 정확도: {f_accuracy}')
    print(f"조정된 파라미터: {params}")

    # 재학습
    best_model_2nd = train(params, dtrain, num_boost_round=cv_results.shape[0])

#%%

print(f'1차 정확도: {f_accuracy}')

# %% Final Evaluation
y_val_pred = best_model_2nd.predict(dval).argmax(axis=1)
val_accuracy = accuracy_score(y_val_le, y_val_pred)
print(f"검증 데이터 최종 정확도: {val_accuracy:.3f}")
# %%
