import re
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from konlpy.tag import Okt
from xgboost import DMatrix, cv

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

# 넘어짐, 떨어짐, 물체에 맞음, 끼임, 기타, 절단, 베임, 부딪힘만 고려
cond = df['인적사고종류'].isin(['깔림', '없음', '질병', '찔림', '화상', '교통사고', '감전', '질식'])
df = df[~cond]

# 특수문자 공백으로 변환
X = df['사고경위'].apply(lambda x: re.sub(r'[-=+,#/\?:^$.@*"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', ' ', x))
y = df['인적사고']

# 훈련, 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

# 한국어 형태소 분석기 정의
def tokenizer_kr(text):
    okt = Okt()
    tokens_tmp = okt.pos(text, stem=True)
    tokens_kr = [token[0] for token in tokens_tmp if token[1] in ("Noun", "Verb")]
    return tokens_kr

# TfidfVectorizer 적용
tfidf_vect = TfidfVectorizer(tokenizer=tokenizer_kr, min_df=2, max_df=2000)
X_train_tf = tfidf_vect.fit_transform(X_train)
X_test_tf = tfidf_vect.transform(X_test)

# Label-encoding y
le = LabelEncoder()
y_train_le = le.fit_transform(y_train)
y_test_le = le.transform(y_test)

# XGBoost 설정 및 Cross Validation
params = {
    "objective": "multi:softprob",
    "num_class": len(le.classes_),
    "random_state": 42,
    "tree_method": "hist"
}

dtrain = DMatrix(X_train_tf, label=y_train_le)
dtest = DMatrix(X_test_tf, label=y_test_le)

cv_results = cv(
    params=params,
    dtrain=dtrain,
    num_boost_round=500,
    nfold=5,
    metrics="mlogloss",
    early_stopping_rounds=10,
    verbose_eval=True
)

print("CV 결과:", cv_results)

# 최적 모델로 예측
from xgboost import train
best_model = train(params, dtrain, num_boost_round=cv_results.shape[0])

y_pred = best_model.predict(dtest)
y_pred_labels = y_pred.argmax(axis=1)

print('정확도: {0:0.3f}'.format(accuracy_score(y_test_le, y_pred_labels)))

# 테스트 결과 확인
pred_chr = le.inverse_transform(y_pred_labels)
pred_series = pd.Series(pred_chr, index=X_test.index, name='예측')
res = pd.concat([X_test, y_test, pred_series], axis=1)

for i in range(5):
    print('사고경위 :', res.iloc[i, 0])
    print('보고 : ', res.iloc[i, 1], ', 예측 : ', res.iloc[i, 2])
    print()
