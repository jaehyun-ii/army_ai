# 적대적 공격 실험 보고서 템플릿

이 템플릿은 AI 객체 탐지 모델에 대한 적대적 공격 실험 결과를 체계적으로 문서화하기 위한 HTML 보고서 양식입니다.

## 📁 파일 구성

- `report_template.html` - 보고서 HTML 템플릿
- `style.css` - 보고서 스타일시트
- `example_report.html` - 예시 보고서 (브라우저에서 미리보기)
- `generate_report.py` - PDF 보고서 생성 스크립트 (WeasyPrint)
- `example_data.json` - 예시 데이터
- `example_usage.py` - 사용 예시 코드
- `requirements.txt` - 필요한 Python 라이브러리
- `README.md` - 사용 설명서 (이 파일)

## 📋 보고서 구성

### 1. 표지 페이지
- 보고서 제목
- 실험 기본 정보 (일시, 실험자, 시스템)

### 2. 모델 및 데이터셋 정보
- 대상 모델 상세 정보
- 실험 데이터셋 통계

### 3. 공격 기법 정보
- 사용된 공격 방법
- 공격 파라미터
- 실험 환경

### 4. 시각적 비교 분석
- 원본 vs 공격 이미지 비교 (3개 샘플)
- Detection bbox 및 confidence 표시
- 패치 시각화 (패치 공격인 경우)

### 5. 정량적 결과 분석
- **5.1 전체 성능 비교 테이블**: Clean vs Attacked 수치 비교
  - mAP@0.5, mAP@0.75, mAP@0.5:0.95
  - Precision, Recall, F1 Score
  - 탐지 성공률

- **5.2 공격 효과 지표**: ASR, ΔmAP, FNR 증가, Confidence 감소

- **5.3 성능 비교 그래프**:
  - **5.3.1 Clean vs Attacked 막대 그래프**: 6개 지표 (F1, AP@50, AP@75, AP@[50:95], Precision, Recall) 비교
  - **5.3.2 Precision-Recall Curve**: Clean과 Attacked PR Curve, AUC 비교
  - **5.3.3 클래스별 성능 변화**: 각 클래스별 AP 감소율
  - **5.3.4 Confidence Score 분포**: 히스토그램으로 confidence 변화 시각화

### 6. 결론
- 실험 결과 요약
- 주요 발견사항
- 권장사항

## 🔧 사용 방법

### 방법 1: WeasyPrint를 사용한 자동 PDF 생성 (권장)

#### 1-1. 설치

```bash
# Python 라이브러리 설치
pip install -r requirements.txt

# 한글 폰트 설치 (Linux)
sudo apt-get install fonts-nanum fonts-nanum-coding

# 한글 폰트 설치 (macOS)
brew install font-nanum

# Windows: NanumGothic 폰트를 수동으로 설치
```

#### 1-2. 커맨드라인에서 사용

```bash
# 기본 사용
python generate_report.py --data example_data.json --output my_report.pdf

# 옵션 설명
python generate_report.py \
    --data evaluation_result.json \      # 평가 결과 JSON 파일
    --output adversarial_report.pdf \    # 출력 PDF 경로
    --template-dir ./templates \         # 템플릿 디렉토리 (선택)
    --no-charts \                        # 그래프 자동 생성 비활성화 (선택)
    --keep-temp                          # 임시 파일 유지 (디버깅용)
```

#### 1-3. Python 코드에서 사용

```python
from generate_report import AdversarialReportGenerator
import json

# 1. 보고서 생성기 초기화
generator = AdversarialReportGenerator()

# 2. 데이터 로드
with open('example_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 3. 보고서 생성 (그래프 자동 생성 포함)
output_path = generator.generate_report(
    data=data,
    output_pdf='adversarial_attack_report.pdf',
    generate_charts=True  # 막대 그래프, PR Curve 등 자동 생성
)

# 4. 임시 파일 정리
generator.cleanup()

print(f"보고서 생성 완료: {output_path}")
```

#### 1-4. 평가 서비스와 통합

```python
# backend/app/services/evaluation_service.py 에 추가

async def generate_pdf_report(
    self,
    db: AsyncSession,
    evaluation_id: UUID
) -> str:
    """평가 결과로부터 PDF 보고서 생성"""

    from generate_report import AdversarialReportGenerator

    # 평가 결과 가져오기
    evaluation = await crud.evaluation.get(db, id=evaluation_id)

    # 보고서 데이터 구성
    report_data = {
        'EXPERIMENT_DATE': evaluation.created_at.strftime('%Y-%m-%d'),
        'MODEL_NAME': evaluation.model.name,
        'ATTACK_TYPE': evaluation.attack_method,

        # 성능 지표
        'clean_metrics': {
            'f1': evaluation.clean_f1,
            'ap50': evaluation.clean_map50,
            'ap75': evaluation.clean_map75,
            'ap': evaluation.clean_map,
            'precision': evaluation.clean_precision,
            'recall': evaluation.clean_recall
        },
        'attacked_metrics': {
            'f1': evaluation.attacked_f1,
            'ap50': evaluation.attacked_map50,
            'ap75': evaluation.attacked_map75,
            'ap': evaluation.attacked_map,
            'precision': evaluation.attacked_precision,
            'recall': evaluation.attacked_recall
        },

        # ... 기타 필드
    }

    # PDF 생성
    generator = AdversarialReportGenerator()
    pdf_path = generator.generate_report(
        data=report_data,
        output_pdf=f'reports/evaluation_{evaluation_id}.pdf',
        generate_charts=True
    )
    generator.cleanup()

    return pdf_path
```

### 방법 2: 수동 템플릿 치환

템플릿의 `{{PLACEHOLDER}}` 형식의 텍스트를 실제 데이터로 치환합니다.

```python
# 예시: Python을 사용한 치환
with open('report_template.html', 'r', encoding='utf-8') as f:
    template = f.read()

# 데이터 딕셔너리
data = {
    'EXPERIMENT_DATE': '2025-12-29',
    'EXPERIMENTER_NAME': '홍길동',
    'MODEL_NAME': 'YOLOv8n',
    'ATTACK_TYPE': 'PGD (Projected Gradient Descent)',
    # ... 더 많은 데이터
}

# 치환
for key, value in data.items():
    template = template.replace(f'{{{{{key}}}}}', str(value))

# 저장
with open('report_output.html', 'w', encoding='utf-8') as f:
    f.write(template)
```

### 2. 이미지 파일 준비

다음 이미지들을 준비하고 경로를 템플릿에 입력합니다:

**샘플 이미지 (각 3개씩)**
- `{{SAMPLE1_CLEAN_IMAGE}}` - 샘플 1 원본 이미지
- `{{SAMPLE1_ATTACKED_IMAGE}}` - 샘플 1 공격 이미지
- `{{SAMPLE2_CLEAN_IMAGE}}` - 샘플 2 원본 이미지
- `{{SAMPLE2_ATTACKED_IMAGE}}` - 샘플 2 공격 이미지
- `{{SAMPLE3_CLEAN_IMAGE}}` - 샘플 3 원본 이미지
- `{{SAMPLE3_ATTACKED_IMAGE}}` - 샘플 3 공격 이미지

**패치 이미지 (패치 공격인 경우)**
- `{{PATCH_IMAGE}}` - 생성된 패치 이미지
- `{{PATCH_PLACEMENT_IMAGE}}` - 패치 배치 예시 이미지

**그래프 이미지**
- `{{METRICS_BAR_CHART}}` - Clean vs Attacked 6개 지표 막대 그래프 (F1, AP@50, AP@75, AP@[50:95], Precision, Recall)
- `{{PR_CURVE_CHART}}` - Precision-Recall Curve (Clean과 Attacked 비교)
- `{{PER_CLASS_AP_CHART}}` - 클래스별 Average Precision 비교 그래프
- `{{CONFIDENCE_DISTRIBUTION_CHART}}` - Confidence Score 분포 히스토그램

### 3. 공격 유형별 파라미터 처리

템플릿은 모든 공격 유형의 파라미터를 포함하고 있습니다. 실제 보고서 생성 시 해당하지 않는 행을 제거하거나 숨겨야 합니다.

**노이즈 공격** (FGSM, PGD, Universal Noise, Noise OSFD)
- `.noise-attack-params` 클래스 행 표시
- `.patch-attack-params` 클래스 행 숨김

**패치 공격** (Patch, D-Patch, Robust D-Patch, NAP)
- `.patch-attack-params` 클래스 행 표시
- `.noise-attack-params` 클래스 행 숨김

**세부 파라미터**
- `.pgd-only` - PGD만 해당
- `.adv-patch-only` - Adversarial Patch만 해당
- `.robust-dpatch-only` - Robust D-Patch만 해당
- `.nap-only` - NAP만 해당

### 4. 결과 평가 클래스 설정

결과의 심각도에 따라 CSS 클래스를 설정합니다:

**성능 변화 클래스**
- `negative` - 부정적 변화 (빨간색)
- `positive` - 긍정적 변화 (녹색)

**공격 성공 배지**
- `success` - 공격 성공 (녹색)
- `partial` - 부분 성공 (주황색)
- `failed` - 공격 실패 (빨간색)

**성능 중요도**
- `performance-critical` - 치명적 (빨간 배경)
- `performance-warning` - 경고 (노란 배경)
- `performance-good` - 양호 (녹색 배경)

## 📊 주요 플레이스홀더 목록

### 기본 정보
- `{{EXPERIMENT_DATE}}` - 실험 일시
- `{{EXPERIMENTER_NAME}}` - 실험자 이름
- `{{SYSTEM_NAME}}` - 대상 시스템 이름
- `{{REPORT_VERSION}}` - 보고서 버전

### 모델 정보
- `{{MODEL_NAME}}` - 모델 이름
- `{{MODEL_ARCHITECTURE}}` - 모델 아키텍처
- `{{MODEL_SIZE}}` - 모델 크기
- `{{INPUT_RESOLUTION}}` - 입력 해상도
- `{{NUM_CLASSES}}` - 클래스 수
- `{{CLASS_NAMES}}` - 클래스 목록
- `{{INFERENCE_DEVICE}}` - 추론 장치
- `{{CONFIDENCE_THRESHOLD}}` - Confidence 임계값

### 데이터셋 정보
- `{{DATASET_NAME}}` - 데이터셋 이름
- `{{TOTAL_IMAGES}}` - 총 이미지 수
- `{{TARGET_CLASS}}` - 타겟 클래스
- `{{TARGET_CLASS_IMAGES}}` - 타겟 클래스 이미지 수
- `{{IMAGE_RESOLUTION_RANGE}}` - 이미지 해상도 범위
- `{{ANNOTATION_FORMAT}}` - 어노테이션 형식

### 공격 정보
- `{{ATTACK_TYPE}}` - 공격 유형
- `{{ATTACK_DESCRIPTION}}` - 공격 설명
- `{{ATTACK_INTENSITY}}` - 공격 강도 (weak/medium/strong)

### 노이즈 공격 파라미터
- `{{EPSILON}}` - Epsilon 값
- `{{ITERATIONS}}` - 반복 횟수
- `{{ALPHA}}` - Alpha (PGD 스텝 크기)

### 패치 공격 파라미터
- `{{PATCH_SIZE}}` - 패치 크기
- `{{LEARNING_RATE}}` - 학습률
- `{{ITERATIONS}}` - 반복 횟수
- `{{ROTATION_RANGE}}` - 회전 범위
- `{{SCALE_RANGE}}` - 크기 조정 범위
- `{{EOT_SAMPLES}}` - EoT 샘플 수
- `{{GAN_CLASS_ID}}` - GAN 클래스 ID
- `{{TV_WEIGHT}}` - TV loss 가중치
- `{{PATCH_SCALE}}` - 패치 스케일

### 실험 환경
- `{{GPU_INFO}}` - GPU 정보
- `{{BATCH_SIZE}}` - 배치 크기
- `{{TOTAL_TIME}}` - 총 처리 시간
- `{{AVG_TIME_PER_IMAGE}}` - 이미지당 평균 시간

### 성능 지표 (Clean)
- `{{CLEAN_MAP50}}` - Clean mAP@0.5 (AP@50)
- `{{CLEAN_MAP75}}` - Clean mAP@0.75 (AP@75)
- `{{CLEAN_MAP}}` - Clean mAP@0.5:0.95 (AP@[50:95])
- `{{CLEAN_PRECISION}}` - Clean Precision
- `{{CLEAN_RECALL}}` - Clean Recall
- `{{CLEAN_F1}}` - Clean F1 Score
- `{{CLEAN_DETECTION_RATE}}` - Clean 탐지 성공률

### 성능 지표 (Attacked)
- `{{ATTACKED_MAP50}}` - Attacked mAP@0.5 (AP@50)
- `{{ATTACKED_MAP75}}` - Attacked mAP@0.75 (AP@75)
- `{{ATTACKED_MAP}}` - Attacked mAP@0.5:0.95 (AP@[50:95])
- `{{ATTACKED_PRECISION}}` - Attacked Precision
- `{{ATTACKED_RECALL}}` - Attacked Recall
- `{{ATTACKED_F1}}` - Attacked F1 Score
- `{{ATTACKED_DETECTION_RATE}}` - Attacked 탐지 성공률

### 공격 효과 (Delta/변화량)
- `{{DELTA_MAP50}}` - ΔmAP@0.5
- `{{DELTA_MAP75}}` - ΔmAP@0.75
- `{{DELTA_MAP}}` - ΔmAP@0.5:0.95
- `{{DELTA_PRECISION}}` - ΔPrecision
- `{{DELTA_RECALL}}` - ΔRecall
- `{{DELTA_F1}}` - ΔF1 Score
- `{{DELTA_DETECTION_RATE}}` - Δ탐지율
- `{{DELTA_MAP_PERCENT}}` - mAP 감소율 (%)
- `{{ASR_VALUE}}` - Attack Success Rate (%)
- `{{FNR_INCREASE}}` - False Negative Rate 증가
- `{{AVG_CONF_DECREASE}}` - Average Confidence 감소

### 샘플 이미지 정보 (1-3)
- `{{SAMPLEn_CLEAN_DETECTIONS}}` - Clean 탐지 개수
- `{{SAMPLEn_CLEAN_AVG_CONF}}` - Clean 평균 confidence
- `{{SAMPLEn_CLEAN_MAX_CONF}}` - Clean 최고 confidence
- `{{SAMPLEn_ATTACKED_DETECTIONS}}` - Attacked 탐지 개수
- `{{SAMPLEn_ATTACKED_AVG_CONF}}` - Attacked 평균 confidence
- `{{SAMPLEn_ATTACKED_MAX_CONF}}` - Attacked 최고 confidence
- `{{SAMPLEn_SUCCESS_CLASS}}` - 공격 성공 클래스 (success/partial/failed)
- `{{SAMPLEn_RESULT_TEXT}}` - 결과 텍스트

### 결론
- `{{FINDINGS_TEXT}}` - 주요 발견사항
- `{{RECOMMENDATIONS_TEXT}}` - 권장사항
- `{{REPORT_GENERATION_DATE}}` - 보고서 생성일

## 🎨 스타일 커스터마이징

`style.css` 파일을 수정하여 보고서 디자인을 변경할 수 있습니다:

- 색상 테마: `#1B1760` (진한 파랑), `#4C4CFF` (밝은 파랑)
- 페이지 크기: A4 (210mm × 297mm)
- 폰트: Noto Sans KR, Malgun Gothic

## 📝 자동화 스크립트 예시

보고서 생성을 자동화하는 Python 스크립트 예시입니다:

```python
import json
from pathlib import Path
from datetime import datetime

def generate_report(evaluation_result, output_path):
    """
    평가 결과에서 보고서 생성

    Args:
        evaluation_result: 평가 결과 딕셔너리
        output_path: 출력 HTML 파일 경로
    """
    # 템플릿 로드
    template_path = Path('report_template.html')
    with open(template_path, 'r', encoding='utf-8') as f:
        template = f.read()

    # 데이터 매핑
    data = {
        # 기본 정보
        'EXPERIMENT_DATE': datetime.now().strftime('%Y-%m-%d'),
        'EXPERIMENTER_NAME': evaluation_result.get('experimenter', 'Unknown'),
        'SYSTEM_NAME': 'Army AI Defense System',
        'REPORT_VERSION': '1.0',

        # 모델 정보
        'MODEL_NAME': evaluation_result['model']['name'],
        'MODEL_ARCHITECTURE': evaluation_result['model']['architecture'],
        'INPUT_RESOLUTION': f"{evaluation_result['model']['input_size'][0]}×{evaluation_result['model']['input_size'][1]}",

        # 성능 지표
        'CLEAN_MAP50': f"{evaluation_result['clean_metrics']['map50']:.4f}",
        'ATTACKED_MAP50': f"{evaluation_result['attacked_metrics']['map50']:.4f}",
        'DELTA_MAP50': f"{evaluation_result['delta']['map50']:.4f}",

        # ... 더 많은 매핑
    }

    # 치환
    for key, value in data.items():
        template = template.replace(f'{{{{{key}}}}}', str(value))

    # 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(template)

    print(f"보고서 생성 완료: {output_path}")

# 사용 예시
if __name__ == '__main__':
    # 평가 결과 로드 (예시)
    with open('evaluation_result.json', 'r') as f:
        result = json.load(f)

    # 보고서 생성
    generate_report(result, 'adversarial_attack_report.html')
```

## 🖨️ 인쇄 및 PDF 변환

### 브라우저에서 PDF로 저장
1. HTML 파일을 브라우저에서 열기
2. Ctrl+P (인쇄)
3. "PDF로 저장" 선택
4. 여백: 기본값
5. 배경 그래픽: 포함

### Python으로 PDF 변환

```python
from weasyprint import HTML

HTML('report_output.html').write_pdf('report_output.pdf')
```

## 📌 주의사항

1. **이미지 경로**: 상대 경로 또는 절대 경로 사용 가능
2. **인코딩**: UTF-8 인코딩 필수
3. **브라우저 호환성**: 최신 Chrome, Firefox, Edge에서 테스트됨
4. **이미지 크기**: 큰 이미지는 자동으로 크기 조정됨
5. **페이지 나누기**: 자동으로 페이지 분할됨

## 📞 지원

문제가 발생하거나 개선 사항이 있으면 프로젝트 관리자에게 문의하세요.

---

**Army AI Defense System**
Adversarial Attack Report Template v1.0
Generated: 2025-12-29
