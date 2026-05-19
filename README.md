# CAVE

CAVE(Credibility Audit for AI-Generated Evidence)는 AI 생성물/딥페이크 증거를 여러 레이어로 분석하고, 유포 정황을 함께 반영해 피해 확산 위험을 정량화하는 로컬 데모 프로젝트입니다.

## 주요 기능

- C2PA/메타데이터 기반 출처 인증 검사
- AI 생성 이미지/영상 탐지
- 영상 rPPG 기반 생체 신호 보조 분석
- 생성 모델 fingerprint attribution
- Cross-layer Audit 기반 증거 일관성 판정
- GNN 기반 확산 위험도 및 learned 재유포 위험도 산정
- Streamlit 기반 로컬 웹 데모

## 실행

```bash
pip install -r requirements.txt
streamlit run app.py
```

브라우저에서 `http://localhost:8501`을 열고 이미지 또는 영상을 업로드하면 됩니다.

## 데이터

원본 데이터셋과 데모 파일은 용량과 라이선스 문제 때문에 GitHub에 포함하지 않습니다.

제외되는 주요 경로:

- `FaceForensics++_C23/`
- `RedFace/`
- `data/`
- `test_data/`
- `output/`

모델 체크포인트와 calibration 파일은 데모 재현성을 위해 `models/`에 포함합니다.

## 현재 상태

웹 데모는 레이어 1~7 분석 결과를 화면에 표시합니다. PDF 보고서 생성 기능은 한글 폰트 깨짐 문제 때문에 현재 웹 데모에서 비활성화되어 있으며, 관련 코드는 추후 재활성화를 위해 `layers/report_generator.py`에 보존되어 있습니다.

자세한 구현 상태와 평가 결과는 [CONTEXT.md](CONTEXT.md)를 참고하세요.
