import time
import pandas as pd
import requests
from lxml import html
import chardet  # 인코딩 감지를 위한 라이브러리

# User-Agent 설정
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.5481.177 Safari/537.36"
}

# 시작 번호와 크롤링 개수 설정
start_no = 32380  # 시작 case_no 번호
n = 5             # 크롤링할 데이터 개수

# 데이터를 저장할 리스트
data_list = []

# 크롤링 시작
for i in range(start_no, start_no - n, -1):
    try:
        url = f'https://www.csi.go.kr/acd/acdCaseView.do?case_no={i}'
        print(f"Processing: {url}")
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        # 인코딩 자동 감지
        detected_encoding = chardet.detect(response.content)['encoding']
        response.encoding = detected_encoding

        # HTML 파싱
        tree = html.fromstring(response.text)

        # XPath를 사용하여 데이터 추출
        incident = tree.xpath('//*[@id="main"]/div[1]/table/tbody/tr[5]/td[3]/text()')
        description = tree.xpath('//*[@id="main"]/div[1]/table/tbody/tr[12]/td[2]/text()')

        # 결과 처리
        incident_text = incident[0].strip() if incident else None
        description_text = description[0].strip() if description else None

        # 디버깅 로그 추가
        print(f"[DEBUG] Incident: {incident_text}, Description: {description_text}")

        if incident_text or description_text:
            data_list.append({'인적사고': incident_text, '사고경위': description_text})
        else:
            print(f"No data found for case_no={i}")
    except Exception as e:
        print(f"Error occurred for case_no={i}: {e}")
    time.sleep(1)  # 요청 간격 조정

# DataFrame으로 저장 및 CSV 파일로 출력
df = pd.DataFrame(data_list)
df.to_csv('건설사고수집.csv', index=False, encoding='utf-8-sig')