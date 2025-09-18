# Food Classification & Recipe Recommendation

**컴퓨터공학 소프트웨어공학 프로젝트**  
이미지 분류와 레시피 검색을 결합한 음식 인식 및 레시피 제공 시스템입니다.  
사용자가 음식 사진을 업로드하거나 재료를 입력하면, 분류 모델과 공공 Recipe API를 활용하여 적합한 레시피를 제공합니다.

---

## 🎯 프로젝트 목표

1. 사용자가 촬영한 음식 이미지를 업로드하면 이를 자동으로 판별
2. 인식된 음식 이름과 태그 기반으로 레시피를 검색 및 제공
3. 결과를 텍스트 또는 PDF 형식으로 출력
4. 직관적 UI/UX를 통해 검색 경험 개선

---

## ✨ 주요 기능

- **이미지 분류**  
  - EfficientNetB4 기반 모델로 음식 이미지 분류
  - 식품안전나라 데이터셋 기반 학습

- **레시피 검색**  
  - 음식명 또는 재료명을 Recipe API와 연동
  - 조리 방법, 재료 리스트, 영양 정보 제공

- **복합 검색 지원**  
  - 이미지 + 텍스트 입력을 동시에 활용 가능

- **결과 제공**  
  - 클라이언트 선택에 따라 텍스트 / PDF 출력
  - 잘못된 이미지 형식·용량 초과 시 오류 메시지 표시

---

## 🏗️ 시스템 아키텍처

```
Client (웹 브라우저)
   │
   ▼
Nginx (리버스 프록시)
   │
   ▼
WSGI (Flask 인터페이스)
   │
   ▼
Flask Application ── EfficientNetB4 (이미지 분류)
        │
        └── Recipe API (레시피 검색)
   │
   ▼
결과 반환 (JSON / Text / PDF)
```

- **Client**: 이미지 업로드 & 재료 입력 인터페이스
- **Nginx**: 보안 및 요청 라우팅
- **Flask**: 핵심 로직 (이미지 분석, 레시피 API 호출, 결과 가공)
- **EfficientNetB4**: 음식 이미지 분류 모델
- **Recipe API**: 식품안전나라 DB(Open API) 사용

---

## ⚙️ 개발 환경

- **OS**: Windows 11  
- **CPU**: AMD Ryzen 5 5600  
- **GPU**: NVIDIA GeForce RTX 4060  
- **RAM**: 16GB  

- **Python**: 3.9.19  
- **CUDA**: 12.6  
- **cuDNN**: 9.5.1  
- **Nginx**: 1.26.2  

- **Dependencies**:  
  ```
  torch, torchvision, Pillow, efficientnet-pytorch,
  requests, flask, flask-cors, waitress,
  reportlab, pandas, scikit-learn
  ```

---

## 🚀 실행 방법

### 1. 서버 구동
1. `main.py`의 경로 및 하이퍼 파라미터 설정
2. 포트 포워딩 및 `nginx.conf` 수정
3. `server.py` 실행

### 2. 모델 학습
1. `dataset/` 폴더에 클래스별 이미지 데이터 준비  
   ```
   dataset/
   ├── Class1/
   │    └── img1.jpg
   └── Class2/
        └── img2.jpg
   ```
2. 클래스명 매핑용 Excel 파일 준비
3. `main.py` 실행 → 학습 완료 후 가중치 `model_save_path`에 저장

---

## 🧪 테스트

- **기능 테스트**: 이미지 업로드 → 분류 결과 출력 → API 검색 → 결과 확인
- **비기능 테스트**: 응답 속도(10초 이내), 동시 사용자 200명 이상 처리, 오류 메시지 처리

---

## 📚 배운 점

- 머신러닝 모델(EfficientNet)과 실서비스(Flask+Nginx) 연동 경험
- 이미지 기반 분류 + 텍스트 기반 검색을 결합한 복합 서비스 설계
- API 활용을 통한 외부 데이터 통합
- 성능 최적화(응답 시간 단축, 동시성 고려) 경험

---

## 📄 라이선스

학부 소프트웨어공학 과제 목적으로 작성된 프로젝트입니다.  
개인 학습 및 참고용으로 활용 가능합니다.
