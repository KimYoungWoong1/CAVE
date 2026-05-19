# CAVE 프로젝트 컨텍스트
## AI 생성물 증거 검증 및 피해 정량화 파이프라인

---

## 프로젝트 개요

**프로젝트명**: CAVE (Credibility Audit for AI-Generated Evidence)
**목적**: 딥페이크 성범죄를 포함한 AI 생성물 관련 범죄에서 제출된 디지털 파일의 AI 생성·조작 여부를 다층적으로 검증하고 피해 규모를 수치화하는 파이프라인 구현
**배경**: 성균관대학교 I2S Lab, Applied Artificial Intelligence Department
**맥락**: 글로벌리더학부 × 글로벌융합학부 융합 학술제 — AI 규제와 혁신 정책 아이디어 대회

---

## 핵심 문제의식

재판에서 두 가지 상황이 발생하고 있음:
1. 가해자가 AI로 만든 가짜 증거를 제출하는 경우 (예: AI 합성 알리바이 사진)
2. 진짜 증거인데 변호인이 "딥페이크 아닌가요?"라고 주장해서 무력화하는 경우

현재 한국 형사소송법에는 디지털 증거가 AI로 만들어진 건지 판단하는 공식 기준이 없음.
해외(미국 FRE Rule 901, EU AI Act)에서는 논의 중이나 한국 법제화는 전무.

---

## 전체 파이프라인 구조

```
[디지털 파일 입력]
        ↓
[레이어 1: C2PA 출처 검증] ← 암호화 기반 디지털 서명 표준 (비AI)
        ↓
[레이어 2: 워터마킹 탐지] ← 디지털 신호처리 기반 (비AI)
        ↓
[레이어 3: AI 탐지 모델 분석] ← EfficientNet-B7 + ViT 하이브리드 (AI)
        ↓
[레이어 4: 생체·음성 신호 일치성 검사] ← rPPG 기반 심박 신호 분석 (AI)
        ↓
[레이어 5: 생성 모델 핑거프린트 추정] ← 주파수·노이즈 패턴 분석 (AI)
        ↓
[레이어 6: Cross-layer Audit] ← cosine similarity 기반 충돌 판정 (AI)
        ↓
[레이어 7: 피해 규모 정량화] ← GNN 기반 확산 추적 + 가중 합산 (AI)
        ↓
[레이어 8: 법원 제출용 기술 보고서 생성] ← 현재 웹 데모 비활성화
```

---

## 레이어별 상세 스펙

### 레이어 1 — C2PA 출처 검증
- **기술**: C2PA (Content Provenance and Authenticity) 표준 v2.2
- **라이브러리**: `c2pa-python`
- **입력**: 이미지/영상 파일
- **출력**:
  - manifest 존재 여부 (True/False)
  - 전자서명 유효성 (Valid/Invalid/None)
  - AI 사용 여부 정보 포함 여부
  - 편집 이력 요약
- **핵심 한계**: C2PA는 출처(provenance) 기록 메커니즘일 뿐, 진정성(authenticity) 보증 불가. 메타데이터 삭제·타임스탬프 조작 취약점 존재 (Golaszewski et al., 2026)
- **판정**: manifest 없음 → 다른 레이어 결과와 종합 판단 필요

### 레이어 2 — 워터마킹 탐지
- **기술**: 디지털 신호처리 기반 invisible watermark 탐지
- **라이브러리**: `invisible-watermark` 또는 `imwatermark`
- **참고**: Google SynthID (이미지·텍스트·오디오·영상 지원, 비공개 API) → 오픈소스 대체 모델 사용
- **입력**: 이미지/영상 파일
- **출력**:
  - AI 워터마크 존재 여부 (True/False)
  - 신뢰도 점수 (0~1)
- **핵심 한계**: 압축·크롭·재인코딩·화면녹화 시 워터마크 손상 가능. "robust" 워터마크도 단 1장의 샘플로 제거·위조 가능 (arXiv:2502.06418, 2025)
- **Integrity Clash**: C2PA가 인간 제작 → 워터마킹이 AI 생성 → 충돌 발생 시 레이어 6에서 처리

### 레이어 3 — AI 탐지 모델 분석
- **모델**: EfficientNet-B7 + Vision Transformer 하이브리드
  - EfficientNet-B7: 국소적 시각 특징 (얼굴 경계, 피부 질감, 압축 흔적) 추출
  - ViT: 전역적 특징 (이미지 전체 구조, 조명 일관성, 프레임 간 비일관성) 포착
- **참고 논문**: Coccomini et al., ICIAP 2022 (DFDC AUC 0.951, F1 88.0%)
- **Hugging Face**: pretrained 딥페이크 탐지 모델 활용 가능
- **입력**: 이미지/영상 파일
- **출력**:
  - AI 생성 가능성 점수 (0.0~1.0)
  - 얼굴 합성 가능성 (높음/중간/낮음)
  - 프레임 불일치 여부
  - 조명 불일치 여부
  - 최종 판정 (진본 의심 / 위조 의심 / 불확실)
- **핵심 한계**: cross-dataset 환경에서 AUC 최대 50% 급락 (Deepfake-Eval-2024, Chandra et al., 2025)
- **음성 파일**: 별도 음성 포렌식 분석 (음색·억양·호흡·발화 리듬 분석)

### 레이어 4 — 생체 신호 일치성 검사
- **기술**: rPPG (Remote Photoplethysmography)
  - 얼굴 피부색 미세 변화로 심박 신호 추정
  - 실제 영상 vs 딥페이크 영상의 생리학적 패턴 차이 탐지
- **라이브러리**: `rppg-toolbox` 또는 `pyVHR`
- **참고 논문**:
  - DeepFakesON-Phys (Hernandez-Ortega et al., AAAI Workshop 2021)
  - DeepRhythm (Qi et al., ACM MM 2020)
- **입력**: 얼굴 영상 (최소 10초 이상 권장)
- **출력**:
  - 심박 신호 패턴 자연스러움 점수 (0~1)
  - 생체 신호 일치 여부
- **핵심 한계**: 조명 변화·낮은 해상도에서 신뢰도 저하. 고품질 딥페이크는 일부 생리학적 패턴 모방 가능 → 보조 레이어로 활용
- **음성**: 발화자 음색·억양·호흡·감정 표현 자연스러움 분석

### 레이어 5 — 생성 모델 핑거프린트 추정
- **기술**: 생성형 AI 모델이 이미지/영상 생성 시 남기는 고유 흔적 분석
  - 주파수·노이즈 패턴 분석 (FFT 기반)
  - diffusion reconstruction 반응 분석
  - 색상·텍스처 분포 분석
- **라이브러리**: `numpy`, `scipy` (FFT), `torch` (모델 분석)
- **참고 논문**:
  - Yu et al., ICCV 2021 (Artificial Fingerprinting, GAN attribution)
  - AuthPrint (arXiv:2508.05691, 2025) — 적대적 환경에서도 attribution 가능
  - ACM TOMM 2024 — GAN vs Diffusion 구분 97% 이상 정확도
- **입력**: 이미지/영상 파일
- **출력**:
  - AI 생성 가능성 (높음/중간/낮음)
  - 생성 방식 추정 (face-swap / diffusion / GAN 계열)
  - 모델 계열 추정
  - 신뢰도
- **주의**: 특정 모델명 단정 아닌 확률적·보조적 지표로 제시

### 레이어 6 — Cross-layer Audit
- **핵심 개념**: Integrity Clash (Nemecek et al., arXiv:2603.02378, 2026)
  - C2PA와 워터마킹이 비동기적으로 작동 → 두 검증 결과가 모순될 수 있음
  - metadata washing workflow로 암호학적 침해 없이 인증된 가짜 생성 가능
- **구현**: 각 레이어 출력값의 방향성 불일치를 cosine similarity로 측정
  - 임계값 이하 → Integrity Clash로 분류
- **판정 매트릭스**:

| C2PA 결과 | 워터마킹 결과 | AI 탐지 결과 | 판정 |
|-----------|-------------|------------|------|
| 인간 제작 | 워터마크 없음 | 낮음 | 진본 가능성 높음 |
| AI 생성 | AI 워터마크 있음 | 높음 | AI 생성 가능성 높음 |
| 인간 제작 | AI 워터마크 있음 | 높음 | Integrity Clash → 정밀감정 |
| AI 생성 | 워터마크 없음 | 높음 | 워터마크 손상·제거 의심 |

- **출력**:
  - 최종 판정 (4가지 중 하나)
  - 각 레이어 충돌 상세 내역
  - 정밀감정 필요 여부

### 레이어 7 — 피해 규모 정량화
- **기술**: GNN 기반 확산 추적 + 가중 합산
- **참고 논문**:
  - Bi-GCN (Bian et al., AAAI 2020) — 정보 전파 방향·역방향 구조 학습
  - Song et al., Information Processing & Management 2021 — 시간 변화형 GNN
- **피해 규모 점수 산출식**:
```
피해 규모 점수
= 1.5 × (확산 점수 + 재유포 위험 점수)
+ 1.0 × (식별 위험 점수 + 침해 심각도 점수)
+ 0.5 × (경제적 피해 점수 + 사회적 영향 점수)
```
- **항목별 평가 내용**:
  - 확산 점수: 게시물 수, 플랫폼 수, 공유 수, 조회수, 확산 속도 (GNN 출력값 + heuristic blend)
  - 재유포 위험 점수: 변형본 존재, 폐쇄형 플랫폼 유포, 삭제 후 재등장 여부 (RandomForest learned score + heuristic blend)
  - 식별 위험 점수: 얼굴·음성 일치도, 실명 언급, 소속 정보
  - 침해 심각도 점수: 명예훼손성, 협박성, 성적 조작 여부
  - 경제적 피해 점수: 금전 피해, 기업 평판 손실
  - 사회적 영향 점수: 선거·여론 조작 가능성
- **피해 등급**:
  - 0~7점: 제한적 피해
  - 8~15점: 중간 수준 피해
  - 16~23점: 심각한 피해
  - 24~30점: 광범위·고위험 피해
- **가중치 조정**: 범죄 유형에 따라 조정 가능 (금융사기 → 경제적 피해 가중치 1.5, 선거조작 → 사회적 영향 가중치 1.5)
- **현재 구현**:
  - 확산 위험도는 `DamageInputs` 메타데이터를 approximate propagation graph로 변환한 뒤 `SpreadRiskGCN`으로 예측한다.
  - `DamageInputs`는 유포 플랫폼, 최초 발견 시각, URL/캡처 보전 수, 피해자 특정 가능 여부 같은 유포 정황 메타데이터를 함께 보존한다.
  - approximate graph에는 variant, closed-platform, reappeared-after-deletion motif 노드를 추가해 재유포 구조가 GNN topology에 반영되도록 했다.
  - `DamageResult`에는 `gnn_risk`, `gnn_converted_score`, `heuristic_diffusion`, `blend_ratio`, `gnn_test_acc`, `gnn_fallback_reason`이 구조화되어 저장된다.
  - 재유포 위험도는 `models/redistribution_risk_classifier.joblib`의 RandomForest classifier를 사용하며, learned 환산 점수와 rule heuristic을 `70:30`으로 blend한다.
  - GNN fallback은 `입력 없음`, `모델 없음`, `로딩 실패`로 구분해 웹 근거에 표시한다.

### 레이어 8 — 법원 제출용 기술 보고서 생성
- **현재 상태**: 한글 폰트 깨짐 이슈 때문에 웹 데모에서는 PDF 생성/다운로드를 비활성화했다. `layers/report_generator.py`와 CLI 보고서 생성 경로는 추후 재활성화를 위해 보존한다.
- **라이브러리**: `reportlab` 또는 `fpdf2`
- **출력 항목**:
  - 파일 기본 정보 (파일명, 해시값, 제출 경로)
  - C2PA 검증 결과
  - 워터마킹 결과
  - AI 탐지 결과 (점수 및 주요 근거)
  - 생체·음성 신호 분석
  - 생성 모델 추정
  - 확산 분석
  - 피해 유형 분석
  - 최종 의견 (진본 가능성 / 위조 가능성 / 정밀감정 필요)
  - 피해 규모 점수

---

## 프로젝트 폴더 구조

```
cave_pipeline/
├── CONTEXT.md              # 이 파일
├── main.py                 # 전체 파이프라인 실행
├── layers/
│   ├── c2pa_check.py       # 레이어 1
│   ├── watermark_check.py  # 레이어 2
│   ├── ai_detection.py     # 레이어 3
│   ├── rppg_check.py       # 레이어 4
│   ├── fingerprint.py      # 레이어 5
│   ├── cross_layer_audit.py# 레이어 6
│   ├── damage_score.py     # 레이어 7
│   └── report_generator.py # 레이어 8
├── models/                 # pretrained 모델 가중치
├── test_data/              # 로컬 테스트용 파일 (GitHub 제외)
│   ├── real/               # 진짜 영상
│   └── deepfake/           # 딥페이크 영상
└── output/
    └── report.pdf          # 최종 출력
```

---

## 데이터 준비 계획

1. **진짜 영상**: CC0 라이선스 영상 또는 직접 촬영
2. **딥페이크 영상**: SimSwap 오픈소스 모델로 직접 생성
3. **C2PA 검증 확인**: https://contentcredentials.org/verify

---

## 현재 구현 및 데이터 적용 상태

- **이미지 AI 탐지**: `Organika/sdxl-detector` diffusion detector를 유지하고, GenImage 기반 general AIGC classifier, 얼굴 crop 기반 Xception/EfficientNet/R3D 계열 앙상블, fingerprint 점수를 결합한다.
- **영상 딥페이크 탐지**: MediaPipe 얼굴 crop/align 후 DeepfakeBench-style EfficientNet-B4, Xception, R3D-18, ResNet18, legacy EfficientNet-B0 앙상블을 적용한다.
- **rPPG 생체 신호**: 영상은 CHROM rPPG feature를 추출한 뒤 FFPP C23으로 학습한 RandomForest classifier를 보조 AI 레이어로 사용한다. 모델 파일이 없으면 CHROM heuristic으로 fallback한다.
- **생성 모델 핑거프린트**: 이미지는 RedFace feature로 학습한 RandomForest fingerprint classifier와 GenImage 기반 generator fingerprint classifier를 ensemble한다. 영상은 FFPP C23 얼굴 crop 시계열 feature로 학습한 video fingerprint classifier를 별도로 사용한다. 각 모델 파일이 없으면 기존 heuristic으로 fallback한다.
- **웹 데모 UX**: `app.py`는 최종 판정, 핵심 근거 3줄, 정밀감정 권고 이유, 피해/GNN 근거를 상단에 먼저 보여준다. 사이드바에는 발견 플랫폼, 최초 발견 시각, URL/캡처 보전 수, 확산 규모, 재유포 위험, 피해자 특정성 입력 UI를 제공하며, 세부 결과는 `증거 레이어`와 `피해 산정` 탭으로 분리해 발표용으로 빠르게 읽히도록 정리했다. PDF 보고서 생성은 한글 폰트 문제로 웹 데모에서 비활성화했다.
- **RedFace 적용**: `RedFace/` 원본을 `test_data/redface/{calibration,eval,holdout}/{real,fake}`로 정리했다. fake는 EFS/FAM/FR/FS 방식이 파일명 prefix로 보존된다.
- **RedFace 영상 적용**: `RedFace/FR/videos`는 `test_data/redface_video/{calibration,eval,holdout}/deepfake`로 정리했다. RedFace 내 real video counterpart는 없으므로 영상 calibration에는 보조 데이터로만 사용한다.
- **FaceForensics++ C23 적용**: `FaceForensics++_C23/` 원본을 `test_data/ffpp_c23/{calibration,eval,holdout}/{real,deepfake}`로 정리했다. fake 파일명은 Deepfakes/Face2Face/FaceShifter/FaceSwap/NeuralTextures/DeepFakeDetection prefix를 유지한다.
- **GenImage 적용 경로 추가**: `scripts/prepare_genimage_dataset.py`가 GenImage 원본을 `test_data/genimage/{calibration,eval,holdout}/{real,ai}`로 정리한다. Layer 3 general AIGC classifier와 Layer 5 generator fingerprint classifier의 학습 입력으로 사용한다.
- **Calibration 파일**:
  - 기본 이미지 calibration: `models/image_calibration.json`
  - general AIGC classifier 학습 후 생성 경로: `models/general_aigc_classifier.joblib`
  - general AIGC classifier metadata 학습 후 생성 경로: `models/general_aigc_classifier.meta.json`
  - GenImage fingerprint classifier 학습 후 생성 경로: `models/genimage_fingerprint_classifier.joblib`
  - GenImage fingerprint classifier metadata 학습 후 생성 경로: `models/genimage_fingerprint_classifier.meta.json`
  - 기본 영상 calibration: `models/video_calibration.json`
  - FaceForensics++ 균형 영상 calibration 후보: `models/video_calibration_ffpp_c23_balanced.json`
  - fingerprint classifier: `models/fingerprint_classifier.joblib`
  - fingerprint classifier metadata: `models/fingerprint_classifier.meta.json`
  - rPPG classifier: `models/rppg_classifier.joblib`
  - rPPG classifier metadata: `models/rppg_classifier.meta.json`
  - video fingerprint classifier: `models/video_fingerprint_classifier.joblib`
  - video fingerprint classifier metadata: `models/video_fingerprint_classifier.meta.json`
  - redistribution risk classifier: `models/redistribution_risk_classifier.joblib`
  - redistribution risk classifier metadata: `models/redistribution_risk_classifier.meta.json`
- **피해 확산 GNN**: `DamageInputs` 메타데이터를 합성 전파 그래프로 변환한 뒤 `SpreadRiskGCN`으로 확산 위험도를 추정한다. 기본 체크포인트는 `models/gnn_spread_model.pt`, 학습 메타데이터는 `models/gnn_spread_model_meta.json`에 저장된다. 최종 확산 점수는 GNN 환산 점수와 heuristic 점수를 65:35로 blend한다. 재유포 위험도는 RF learned score와 heuristic을 70:30으로 blend한다.
- **Calibration 교체 실행**:
  - 이미지: `CAVE_IMAGE_CALIBRATION=models/image_calibration.json python ...`
  - 영상: `CAVE_VIDEO_CALIBRATION=models/video_calibration_ffpp_c23_balanced.json python ...`

### 이미지 레이어 3/5 구현 상태

#### 레이어 3 — 이미지 AI 탐지 모델

- **구현 파일**: `layers/ai_detection.py`, `layers/deepfake_image_detectors.py`
- **현재 구성**:
  - `Organika/sdxl-detector`로 SDXL/diffusion 계열 이미지 생성 확률을 먼저 계산한다.
  - `models/general_aigc_classifier.joblib`가 있으면 GenImage 기반 general AIGC 확률을 추가로 계산한다.
  - Xception/EfficientNet/R3D/ResNet 계열 얼굴 조작 detector suite와 레이어 5 fingerprint 점수를 함께 사용한다.
  - RedFace 얼굴 조작 detector와 GenImage 일반 생성 detector를 이미지 ensemble feature로 함께 사용한다.
  - `models/image_calibration.json`의 logistic calibration으로 최종 `ai_probability`를 산출한다.
- **현재 평가 기준**: RedFace eval split에서 `python scripts/compare_image_dataset.py --max-per-label 40 --fake-per-method 10 --seed 42` 실행.
- **평가 결과**:
  - Layer 3 image detector: AUC `0.977`, acc@0.5 `0.938`, best threshold `0.489`, best acc `0.950`
  - 방식별 AUC: EFS `1.000`, FAM `0.927`, FR `1.000`, FS `0.980`
- **해석**: RedFace 계열 얼굴 조작 이미지 데모에는 충분히 쓸 수 있다. 다만 특정 데이터셋과 현재 보정값에 맞춘 결과이므로, 완전한 범용 탐지기나 법적 확정 판정기로 표현하지 않는다.

#### 레이어 5 — 이미지 생성 모델 핑거프린트 attribution

- **구현 파일**: `layers/fingerprint.py`, `scripts/train_fingerprint_classifier.py`
- **현재 구성**:
  - 이미지 입력에서는 `models/fingerprint_classifier.joblib`의 RandomForest classifier를 우선 사용한다.
  - `models/genimage_fingerprint_classifier.joblib`가 있으면 GenImage 기반 generator fingerprint classifier를 함께 사용한다.
  - feature는 FFT 주파수 대역, spectral flatness/centroid/rolloff, 채널 노이즈 상관, 잔차 통계, 엔트로피, gradient, 색상·채도 통계 등 17개다.
  - 출력은 real/fake 확률, RedFace-style 조작 계열 attribution, GenImage generator attribution이다.
  - RedFace 방식군: `entire-face-synthesis`, `face-attribute-manipulation`, `face-reenactment`, `face-swap`.
  - GenImage generator attribution: 원본 generator 폴더명을 보존해 `general-aigc:{generator}` 형태로 표시한다.
  - 모델 파일이 없거나 로딩 실패 시 FFT/노이즈 heuristic fallback으로 동작한다.
- **현재 학습 기준**: RedFace calibration split에서 `python scripts/train_fingerprint_classifier.py --max-per-label 600 --fake-per-method 150 --seed 42` 실행.
- **평가 결과**:
  - Layer 5 fingerprint classifier: AUC `0.943`, acc@0.5 `0.883`, balanced acc@0.5 `0.883`, best threshold `0.462`, best acc `0.889`
  - Method attribution accuracy: `0.920`
  - 방식별 AUC: EFS `1.000`, FAM `0.868`, FR `0.998`, FS `0.906`
- **해석**: 레이어 5는 이제 단순 FFT 점수가 아니라 학습된 이미지 attribution 레이어다. 단, 특정 모델명 단정이 아니라 RedFace 방식군에 대한 확률적 attribution이며, 영상에는 아직 이 image-trained classifier를 적용하지 않는다.

#### 이미지 Cross-layer Audit 기준

- 정지 이미지는 레이어 4 rPPG가 `None`으로 제외된다.
- C2PA 매니페스트 부재와 저신뢰 워터마크 부재는 “인간 제작 증거”로 취급하지 않고 Audit에서 제외한다.
- 이미지에서 레이어 3과 레이어 5가 함께 강한 AI 신호를 내면 `AI 생성 가능성 높음 — 탐지 모델과 핑거프린트 일치` 또는 출처 인증 없는 탐지 양성으로 판정한다.
- C2PA/워터마크가 인간 방향인데 레이어 3/5가 AI 방향이면 Integrity Clash 또는 정밀 감정 필요로 올라간다.

### 영상 레이어 4 구현 상태

- **구현 파일**: `layers/rppg_check.py`, `scripts/train_rppg_classifier.py`
- **현재 구성**:
  - 얼굴 ROI에서 CHROM rPPG 원시 신호를 추출한다.
  - peak ratio, temporal stability, SNR, peak BPM, harmonic ratio, band energy, signal variation, face detection ratio 등 feature를 만든다.
  - `models/rppg_classifier.joblib`의 RandomForest classifier로 deepfake 확률을 추정한다.
  - `models/rppg_classifier.meta.json`의 `best_threshold`를 읽어 rPPG 확률을 보정한다.
  - 모델 파일이 없거나 실패하면 기존 CHROM heuristic 점수로 fallback한다.
- **현재 학습 기준**: FFPP C23 calibration/eval split에서 `python scripts/train_rppg_classifier.py --max-per-label 30 --eval-max-per-label 24 --fake-per-method 6 --eval-fake-per-method 4 --seed 42` 실행.
- **평가 결과**:
  - Layer 4 rPPG RF classifier: AUC `0.569`, acc@0.5 `0.562`, balanced acc@0.5 `0.562`, best threshold `0.382`, best acc `0.604`
  - 방식별 AUC: NeuralTextures `0.823`, Face2Face `0.646`, FaceShifter `0.625`, Deepfakes `0.594`, FaceSwap `0.552`, DeepFakeDetection `0.177`
- **해석**: 레이어 4는 AI classifier로 구현됐지만 성능이 아직 약하다. Cross-layer Audit에서는 단독 확정 근거로 쓰지 않고, detector/fingerprint가 이미 중간 이상일 때만 보조 신호로 사용한다.

### 영상 레이어 5 구현 상태

- **구현 파일**: `layers/fingerprint.py`, `scripts/train_video_fingerprint_classifier.py`
- **현재 구성**:
  - MediaPipe/MTCNN/Haar fallback 얼굴 crop에서 프레임별 FFT·잔차·노이즈 feature를 추출한다.
  - 프레임 feature의 평균, 표준편차, p10, p90과 frame score 통계, temporal delta, face detection ratio를 합쳐 75차원 video-level feature를 만든다.
  - `models/video_fingerprint_classifier.joblib`의 RandomForest classifier로 real/fake 확률과 FFPP 방식군 attribution을 추정한다.
  - `models/video_fingerprint_classifier.meta.json`의 `best_threshold`를 읽어 영상 확률을 보정한다. 현재 `best_threshold=0.560`을 영상 AI support 기준점 `0.600`에 맞춰 재스케일한다.
  - 방식군별 threshold도 metadata에 저장하되, eval AUC가 `0.80` 이상인 방식군만 attribution threshold를 사용하고 약한 방식군은 global threshold로 fallback한다.
  - 영상용 낮음/중간/높음 기준은 이미지와 분리한다: 낮음 `<0.45`, 중간 `0.45~0.60`, 높음 `>=0.60`.
  - 모델 파일이 없으면 영상 압축 보정 heuristic으로 fallback한다.
- **현재 학습 기준**: FFPP C23 calibration/eval split에서 `python scripts/train_video_fingerprint_classifier.py --max-per-label 120 --eval-max-per-label 80 --fake-per-method 20 --eval-fake-per-method 12 --frames 8 --seed 42` 실행.
- **평가 결과**:
  - Video Layer 5 fingerprint classifier: AUC `0.803`, acc@0.5 `0.750`, balanced acc@0.5 `0.749`, best threshold `0.560`, best acc `0.770`
  - 방식별 AUC: DeepFakeDetection `0.916`, Deepfakes `0.885`, FaceSwap `0.870`, NeuralTextures `0.829`, Face2Face `0.666`, FaceShifter `0.650`
- **해석**: 영상 레이어 5도 이제 image-trained classifier가 아니라 별도 학습된 AI 레이어다. 다만 Face2Face/FaceShifter 계열은 아직 약해서, 영상 최종 판정에서는 레이어 3 detector와 rPPG 보조 신호를 함께 해석해야 한다.

#### 영상 Cross-layer Audit 검증

- **검증 스크립트**: `scripts/evaluate_video_audit.py`
- **데모 세트 준비**: `scripts/prepare_video_demo_set.py --overwrite`
- **데모 영상 세트**: `test_data/demo_videos/`
  - `real/`: FFPP real 1개
  - `deepfake/`: DeepFakeDetection, Deepfakes, FaceSwap, NeuralTextures 대표 샘플 4개
- **데모 세트 검증 결과**:
  - 명령: `python scripts/evaluate_video_audit.py --real-dir test_data/demo_videos/real --fake-dir test_data/demo_videos/deepfake --max-per-label 0 --fake-per-method 0`
  - files `5`, real `1`, deepfake `4`
  - false positive(real→deepfake) `0`, false negative(deepfake→real) `0`
  - deepfake review `0`, real review `0`
  - fingerprint AUC `1.000`, acc@0.5 `0.800`; AI detector AUC `0.750`; rPPG AUC `0.000` on demo set
- **FFPP eval 샘플 검증 결과**:
  - 명령: `python scripts/evaluate_video_audit.py --max-per-label 6 --fake-per-method 1 --output output/video_audit_ffpp_eval_sample_after_rules.csv`
  - files `12`, real `6`, deepfake `6`
  - false positive(real→deepfake) `0`, false negative(deepfake→real) `0`
  - real review `3`, deepfake review `3`
  - 확정 판정 대상 exact match `6/6`
  - fingerprint AUC `0.667`, acc@0.5 `0.750`; AI detector AUC `0.722`; rPPG AUC `0.472`
- **해석**: 현재 영상 Audit은 오탐을 피하는 보수적 판정이다. detector와 fingerprint가 함께 강하면 `ai_suspected_unverified`로 올리고, fingerprint 강함 + rPPG 강함만 있는 경우는 대부분 `video_review_recommended`로 둔다. rPPG RF는 약한 보조 신호라 단독 양성으로 판정을 뒤집지 않는다.

### 데모 이미지 세트

- **경로**: `test_data/demo_images/`
- **구성 목적**:
  - `real/`: RedFace Original 기반 진본 얼굴 이미지 샘플
  - `fake/`: RedFace EFS/FAM/FR/FS 조작 방식별 샘플
- **검증 명령**:
  - `python scripts/evaluate_image_audit.py --input-dir test_data/demo_images`
  - `python main.py test_data/demo_images/fake/redface_efs_fake.jpg --demo`
- **현재 검증 결과**: 기본 평가는 RedFace real/fake 데모 샘플 5장만 포함한다. `exact_match=5/5`, review `0`, RedFace fake 4장은 `ai_suspected_unverified`, RedFace real 1장은 `authentic_likely`로 판정된다.

### 전체 레이어 통합 평가

- **통합 평가 스크립트**: `scripts/evaluate_full_pipeline.py`
- **출력 경로**: `output/full_eval/`
  - `full_pipeline_media_rows.csv`: 이미지/영상 파일별 레이어 1~6 점수와 Audit 판정
  - `full_pipeline_damage_rows.csv`: Layer 7 피해/GNN 시나리오별 구조화 결과
  - `full_pipeline_summary.json`: 기계 판독용 요약
  - `full_pipeline_summary.md`: 발표/보고서용 Markdown 표
- **기본 실행 기준**:
  - 명령: `python scripts/evaluate_full_pipeline.py --output-dir output/full_eval`
  - 평가 파일: demo image는 RedFace real/fake 기본 세트 `5`장만 포함한다.
  - demo_images: exact `5/5`, review `0`, FP `0`, FN `0`
  - demo_videos: FFPP 기반 real/deepfake 시연 세트로 평가
  - ffpp_eval_sample: FFPP eval split에서 method별 샘플링 평가
  - accuracy/AUC 요약은 RedFace/FFPP 등 평가 가능한 데이터셋만 comparable row로 집계한다.
  - Layer 7 예시: deepfake sexual demo는 `GNN risk=0.993`, `GNN 환산=4.965`, `heuristic=2.956`, `blend=65:35`, total `21.97`

### 평가 명령

```bash
python scripts/evaluate_full_pipeline.py --output-dir output/full_eval
python scripts/evaluate_full_pipeline.py --skip-ffpp --output-dir output/full_eval_quick
python scripts/compare_image_dataset.py --max-per-label 40 --fake-per-method 10
python scripts/evaluate_image_audit.py --input-dir test_data/demo_images
python scripts/compare_video_pairs.py --real-dir test_data/ffpp_c23/eval/real --fake-dir test_data/ffpp_c23/eval/deepfake --unpaired --max-per-label 18 --fake-per-method 3
python scripts/evaluate_video_audit.py --max-per-label 6 --fake-per-method 1 --output output/video_audit_ffpp_eval_sample_after_rules.csv
python scripts/prepare_video_demo_set.py --overwrite
CAVE_VIDEO_CALIBRATION=models/video_calibration_ffpp_c23_balanced.json python scripts/compare_video_pairs.py --real-dir test_data/ffpp_c23/eval/real --fake-dir test_data/ffpp_c23/eval/deepfake --unpaired --max-per-label 18 --fake-per-method 3
python scripts/train_gnn_spread.py --epochs 80 --samples 1000 --device cpu
python scripts/train_redistribution_risk.py --samples-per-class 1200 --seed 42
python scripts/train_fingerprint_classifier.py --max-per-label 600 --fake-per-method 150
python scripts/prepare_genimage_dataset.py --source GenImage --overwrite
python scripts/train_general_aigc_classifier.py --max-per-label 3000
python scripts/train_genimage_fingerprint_classifier.py --max-per-label 3000
python scripts/train_rppg_classifier.py --max-per-label 30 --eval-max-per-label 24 --fake-per-method 6 --eval-fake-per-method 4
python scripts/train_video_fingerprint_classifier.py --max-per-label 120 --eval-max-per-label 80 --fake-per-method 20 --eval-fake-per-method 12 --frames 8
python scripts/calibrate_image_detector.py --max-per-label 80 --fake-per-method 20
```

---

## 구현 우선순위

**1단계** (먼저): 레이어 1 (C2PA) + 레이어 6 (Cross-layer Audit) + 레이어 7 (피해 점수)

**2단계**: 레이어 3 (AI 탐지) + 레이어 5 (핑거프린트)

**3단계**: 레이어 2 (워터마킹) + 레이어 4 (rPPG)

---

## 법률 파트 연결 (기술 파트와 연동 지점)

- Cross-layer Audit 결과 "위조 가능성 높음" → 형사소송법 개정안 제○조에 따라 AI 생성 증거로 분류
- 피해 규모 점수 → 소송촉진 등에 관한 특례법상 배상명령 신청 시 손해 규모 입증 기술적 근거자료

---

## 주요 참고문헌

1. Nemecek et al. (2026). Authenticated Contradictions from Desynchronized Provenance and Watermarking. arXiv:2603.02378.
2. Golaszewski et al. (2026). Verifying Provenance of Digital Media: Why the C2PA Specifications Fall Short. arXiv:2604.24890.
3. Chandra et al. (2025). Deepfake-Eval-2024. arXiv:2503.02857.
4. Coccomini et al. (2022). Combining EfficientNet and Vision Transformers for Video Deepfake Detection. ICIAP 2022.
5. Bian et al. (2020). Rumor Detection on Social Media with Bi-Directional Graph Convolutional Networks. AAAI 2020.
6. Hernandez-Ortega et al. (2021). DeepFakesON-Phys. AAAI Workshop 2021.
7. Qi et al. (2020). DeepRhythm. ACM MM 2020.
8. Yu et al. (2021). Artificial Fingerprinting for Generative Models. ICCV 2021.
