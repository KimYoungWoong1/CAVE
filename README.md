# CAVE

**CAVE: Credibility Audit for AI-Generated Evidence**

CAVE는 AI 생성 이미지와 딥페이크 영상이 증거·범죄·피해 산정 영역에서 사용되는 상황을 가정하고, 디지털 파일의 생성/조작 가능성과 유포 피해 위험을 다층적으로 분석하는 로컬 웹 데모 프로젝트입니다.

이 프로젝트의 핵심은 단순히 “AI 이미지인가?”를 판별하는 데서 끝나지 않는 것입니다. CAVE는 출처 인증, 워터마크·메타데이터, AI 탐지 모델, rPPG 생체 신호, 생성 모델 fingerprint, Cross-layer Audit, GNN 기반 피해 확산 산정을 하나의 파이프라인으로 연결합니다.

## Motivation

AI 생성 이미지와 딥페이크 영상은 이제 범죄 수단이자 증거 공격 수단이 될 수 있습니다.

예를 들어 다음과 같은 상황을 고려합니다.

- 피해자 얼굴이 합성된 딥페이크 성범죄 이미지가 온라인에 유포됨
- 가해자가 AI 합성물을 실제 자료처럼 제출함
- 반대로 실제 증거에 대해 “딥페이크일 수 있다”는 주장이 제기됨
- 동일하거나 변형된 콘텐츠가 여러 플랫폼에서 재유포됨

CAVE는 이러한 상황에서 디지털 파일 자체의 AI 생성·조작 가능성을 분석하고, 동시에 유포 정황을 그래프 구조로 반영해 피해 확산 위험을 정량화하는 것을 목표로 합니다.

## Core Idea

CAVE는 하나의 모델 판정에 의존하지 않고 여러 레이어의 근거를 교차 검증합니다.

```text
Digital File
  -> Provenance / C2PA Check
  -> Watermark & Metadata Signal
  -> AI Detection Model
  -> rPPG Biometric Signal for Video
  -> Generator Fingerprint Attribution
  -> Cross-layer Audit
  -> GNN-based Harm Propagation Assessment
```

각 레이어는 독립적인 신호를 생성하고, Cross-layer Audit은 출처 기반 신호와 탐지 기반 신호가 서로 일관되는지 확인합니다. 이후 Layer 7은 게시물 수, 플랫폼 수, 공유 수, 조회수, 변형본 존재, 폐쇄형 플랫폼 유포 여부 등을 바탕으로 GNN 기반 피해 확산 위험을 산정합니다.

## Implemented Layers

### Layer 1. Provenance Authentication

- C2PA manifest 존재 여부 확인
- 전자서명 상태 확인
- AI 사용 선언 여부 확인
- 생성 도구 및 편집 이력 요약

### Layer 2. Watermark & Metadata Signal

- 파일 메타데이터 기반 AI 생성 흔적 확인
- 워터마크 또는 AI marker 신호를 보조 근거로 반영
- 출처 레이어와 탐지 레이어 사이의 충돌 여부를 Cross-layer Audit에 전달

### Layer 3. AI Detection Model

- 이미지와 영상을 분리해 AI 생성/딥페이크 가능성을 분석
- 이미지:
  - Hugging Face 기반 diffusion detector
  - GenImage 기반 general AIGC detector
  - 얼굴 crop 기반 이미지 탐지 신호
  - RedFace 얼굴 조작 detector와 GenImage 일반 생성 detector ensemble
  - calibration score 적용
- 영상:
  - MediaPipe 기반 face crop/align
  - EfficientNet/Xception/R3D/ResNet 계열 face deepfake detector ensemble
  - frame-level score를 video-level score로 집계

### Layer 4. rPPG Biometric Signal

- 영상 얼굴 영역에서 CHROM rPPG feature 추출
- FFPP 기반 feature로 학습한 RandomForest classifier 적용
- 심박 신호 자연스러움과 생체 신호 일관성을 보조 AI 레이어로 반영

### Layer 5. Generator Fingerprint Attribution

- 이미지:
  - RedFace 기반 feature로 학습한 RandomForest fingerprint classifier
  - GenImage 기반 generator fingerprint classifier
  - 일반 AI 생성 이미지 attribution과 얼굴 조작 attribution ensemble
  - 생성 방식 및 모델 계열 attribution 표시
- 영상:
  - FaceForensics++ C23 기반 face crop temporal feature 추출
  - video fingerprint classifier로 deepfake/real 신호 분석
  - learned probability, calibrated score, temporal delta 등 근거 표시

### Layer 6. Cross-layer Audit

- C2PA, watermark, AI detection, rPPG, fingerprint 결과를 통합
- 출처 기반 신호와 탐지 기반 신호의 충돌 여부 판단
- 최종 판정, consistency score, expert review 필요 여부 산출
- 영상에서는 detector, rPPG, fingerprint 조합을 함께 반영

### Layer 7. GNN-based Harm Assessment

- 유포 정황 입력을 approximate propagation graph로 변환
- GNN 기반 확산 위험도 산정
- RandomForest 기반 learned redistribution risk 산정
- heuristic score와 learned score를 blend해 최종 피해 규모 점수 산출
- 웹 UI에서 다음 입력값을 반영:
  - 발견 플랫폼
  - 최초 발견 시각
  - URL/캡처 보전 수
  - 동일·유사 게시물 수
  - 플랫폼 수
  - 공유 수
  - 조회수
  - 변형본 존재 여부
  - 폐쇄형 플랫폼 유포 여부
  - 삭제 후 재등장 여부
  - 피해자 특정 가능성
  - 얼굴/음성 동일성 점수
  - 성적 조작, 협박성, 명예훼손성, 평판 손실 여부

## Web Demo

Streamlit 기반 로컬 웹 데모를 제공합니다.

웹 화면에서는 다음 흐름을 확인할 수 있습니다.

1. 이미지 또는 영상 업로드
2. 범죄 유형 선택
3. 유포 정황 입력 또는 데모값 사용
4. 레이어별 분석 실행
5. 종합 판정 확인
6. Cross-layer Audit 근거 확인
7. GNN 피해 확산 위험도 및 재유포 위험도 확인

## Scenario Example

대표 시나리오는 딥페이크 성범죄 의심 이미지/영상 유포 사건입니다.

```text
피해자 A의 얼굴이 합성된 것으로 보이는 이미지가
Telegram, X, 온라인 커뮤니티에서 유포됨.

동일하거나 변형된 이미지가 여러 게시물로 재업로드되고,
피해자의 실명 일부와 소속 정보가 함께 노출됨.

CAVE는 업로드된 파일의 AI 생성·합성 가능성을 분석하고,
유포 정황을 GNN 입력으로 변환해 피해 확산 위험도를 계산함.
```

이 시나리오에서 CAVE는 단순 탐지기가 아니라, AI 생성 증거의 신뢰성 감사와 피해 확산 정량화를 함께 수행하는 파이프라인으로 동작합니다.

## Project Structure

```text
CAVE/
├── app.py                         # Streamlit web demo
├── main.py                        # CLI pipeline entry point
├── layers/
│   ├── c2pa_check.py              # Layer 1
│   ├── watermark_check.py         # Layer 2
│   ├── ai_detection.py            # Layer 3
│   ├── general_aigc_detector.py   # Layer 3 general AI-generated image branch
│   ├── rppg_check.py              # Layer 4
│   ├── fingerprint.py             # Layer 5
│   ├── cross_layer_audit.py       # Layer 6
│   ├── damage_score.py            # Layer 7
│   ├── gnn_spread_model.py        # GNN spread model
│   └── redistribution_risk.py     # Learned redistribution risk
├── scripts/
│   ├── prepare_redface_dataset.py
│   ├── prepare_genimage_dataset.py
│   ├── prepare_tiny_genimage_dataset.py
│   ├── prepare_faceforensics_dataset.py
│   ├── train_fingerprint_classifier.py
│   ├── train_general_aigc_classifier.py
│   ├── train_genimage_fingerprint_classifier.py
│   ├── train_video_fingerprint_classifier.py
│   ├── train_rppg_classifier.py
│   ├── train_gnn_spread.py
│   ├── train_redistribution_risk.py
│   └── evaluate_full_pipeline.py
├── models/                        # Small local checkpoints and calibration files
├── requirements.txt
└── CONTEXT.md
```

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

## CLI Usage

```bash
python main.py path/to/image_or_video --crime-type deepfake_sexual --demo
```

Available crime types:

- `default`
- `deepfake_sexual`
- `financial_fraud`
- `election_manipulation`

## Evaluation Scripts

```bash
python scripts/evaluate_full_pipeline.py --output-dir output/full_eval
python scripts/evaluate_video_audit.py
python scripts/evaluate_image_audit.py
```

The evaluation scripts summarize layer scores, audit decisions, and Layer 7 GNN harm assessment results.

## 활용 데이터

이 프로젝트는 이미지, 영상, 유포 그래프 정황을 분리해 각 레이어에 맞게 활용했습니다.

### RedFace

RedFace는 실제 환경의 얼굴 딥페이크 생성 상황을 반영한 데이터셋으로, 이미지 기반 생성 흔적 분석에 활용했습니다.

- 활용 레이어: Layer 3, Layer 5
- 활용 목적:
  - real/fake 이미지 분리
  - 이미지 fingerprint classifier 학습 및 평가
  - EFS, FAM, FR, FS 방식군을 파일명 prefix로 보존해 생성 방식별 attribution 근거로 활용
- 로컬 정리 경로:
  - `test_data/redface/{calibration,eval,holdout}/{real,fake}`

### GenImage

GenImage는 얼굴 딥페이크 전용이 아니라 일반 AI 생성 이미지 탐지를 위한 real/AI image pair 데이터셋입니다. CAVE에서는 얼굴 합성 데이터만으로는 부족한 “일반 AI 생성 이미지” 신호를 보강하는 용도로 분리해 활용합니다.

- 활용 레이어: Layer 3, Layer 5
- 활용 목적:
  - Layer 3의 `general_aigc_detector` 학습 및 평가
  - Layer 5의 generator fingerprint classifier 학습 및 평가
  - Midjourney, Stable Diffusion, GLIDE 등 generator별 attribution 근거 확보
  - RedFace 기반 얼굴 조작 detector와 image-level ensemble 구성
- 로컬 정리 경로:
  - `test_data/genimage/{calibration,eval,holdout}/{real,ai}`
- 준비 및 학습 명령:

```bash
python scripts/prepare_tiny_genimage_dataset.py --overwrite
python scripts/prepare_genimage_dataset.py --source GenImage --overwrite
python scripts/train_general_aigc_classifier.py --max-per-label 3000
python scripts/train_genimage_fingerprint_classifier.py --max-per-label 3000
python scripts/calibrate_image_detector.py --max-per-label 80 --fake-per-method 20
```

대용량 원본 GenImage 전체가 부담되는 경우에는 `prepare_tiny_genimage_dataset.py`를 우선 사용합니다. 이 스크립트는 Hugging Face Tiny-GenImage를 streaming으로 읽고, 선택한 generator에서 필요한 수량만 JPEG로 저장합니다. 기본 generator는 `Midjourney`, `SD15`, `GLIDE`, `Wukong`, `VQDM`입니다.

### FaceForensics++ C23

FaceForensics++ C23은 real/deepfake 영상 구분과 영상 기반 feature 학습에 활용했습니다.

- 활용 레이어: Layer 3, Layer 4, Layer 5
- 활용 목적:
  - 영상 deepfake detector calibration
  - face crop 기반 video fingerprint classifier 학습
  - rPPG feature classifier 학습
  - real/deepfake 영상 쌍 비교 평가
- 포함 방식군:
  - Deepfakes
  - Face2Face
  - FaceShifter
  - FaceSwap
  - NeuralTextures
  - DeepFakeDetection
- 로컬 정리 경로:
  - `test_data/ffpp_c23/{calibration,eval,holdout}/{real,deepfake}`

### Demo Video/Image Sets

웹 시연을 위해 이미지와 영상 샘플을 별도 demo set으로 구성했습니다.

- 이미지:
  - `test_data/demo_images/real`
  - `test_data/demo_images/fake`
- 영상:
  - `test_data/demo_videos/real`
  - `test_data/demo_videos/deepfake`
- 활용 목적:
  - 웹 UI에서 real/fake 대조군 시연
  - 레이어별 점수와 최종 Audit 판정 비교
  - 딥페이크 성범죄 유포 시나리오 데모 구성

### Synthetic Propagation Graph Data

Layer 7의 GNN 피해 확산 산정과 재유포 위험도 모델에는 유포 정황 기반 합성 그래프 데이터를 활용했습니다.

- 활용 레이어: Layer 7
- 활용 목적:
  - 게시물 수, 플랫폼 수, 공유 수, 조회수, 확산 속도를 graph feature로 변환
  - 변형본, 폐쇄형 플랫폼 유포, 삭제 후 재등장 여부를 graph motif로 반영
  - GNN spread risk model 학습
  - RandomForest redistribution risk classifier 학습
- 관련 스크립트:
  - `scripts/train_gnn_spread.py`
  - `scripts/train_redistribution_risk.py`

## Data Policy

Original datasets and generated outputs are not included in this repository.

The following paths are intentionally excluded:

- `FaceForensics++_C23/`
- `RedFace/`
- `GenImage/`
- `GenImage-Dataset/`
- `data/`
- `test_data/`
- `output/`

This keeps the repository lightweight and avoids redistributing licensed or sensitive media data. Dataset preparation scripts are included so the local experimental setup can be reconstructed when the datasets are available.

## Models

The repository includes small model artifacts and calibration files used by the local demo:

- image fingerprint classifier
- video fingerprint classifier
- rPPG feature classifier
- GNN spread model checkpoint
- redistribution risk classifier
- image/video calibration JSON files

When GenImage is prepared locally, the following additional artifacts are generated by the included training scripts:

- general AIGC image classifier
- GenImage generator fingerprint classifier

## Keywords

`deepfake detection`, `AI-generated media`, `digital evidence`, `C2PA`, `rPPG`, `fingerprint attribution`, `cross-layer audit`, `GNN`, `harm assessment`, `redistribution risk`
