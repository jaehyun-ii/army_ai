# PDF 보고서 생성 기능 설정 가이드

## 📋 개요

적대적 공격 평가 완료 시 자동으로 PDF 보고서가 생성됩니다.
- **자동 생성**: Attack dataset 평가 완료 시 자동으로 PDF 보고서 생성
- **선택적 기능**: WeasyPrint 미설치 시 보고서 생성 건너뜀 (평가는 정상 완료)
- **저장 위치**: `storage/reports/evaluation_{id}_{timestamp}.pdf`

## 🚀 설치 방법

### 1. PDF 생성 라이브러리 설치

```bash
cd /home/jaehyun/army_ai/backend

# WeasyPrint 및 의존성 설치
pip install -r requirements-report.txt
```

### 2. 시스템 패키지 설치 (Linux)

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    libcairo2-dev \
    libpango1.0-dev \
    fonts-nanum \
    fonts-nanum-coding

# 설치 확인
fc-list | grep -i nanum
```

### 3. macOS

```bash
brew install cairo pango gdk-pixbuf libffi
brew install font-nanum
```

### 4. Windows

1. GTK+ Runtime 설치: https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer/releases
2. NanumGothic 폰트 설치: https://hangeul.naver.com/font

## 📁 파일 구조

```
backend/
├── app/
│   ├── services/
│   │   ├── evaluation_service.py          # 평가 완료 시 보고서 생성 호출
│   │   └── report_generation_service.py   # 보고서 생성 서비스 (신규)
│   └── templates/
│       └── adversarial_report/            # 보고서 템플릿 (신규)
│           ├── report_template.html
│           └── style.css
├── requirements.txt                        # 메인 의존성 (보고서 제외)
└── requirements-report.txt                 # 보고서 생성 의존성 (선택)

storage/
└── reports/                                # 생성된 PDF 보고서 저장
    ├── evaluation_xxx_20251229_140530.pdf
    └── temp_charts/                        # 임시 그래프 (자동 정리)
```

## 🔧 사용 방법

### 자동 생성 (기본)

평가 실행 시 자동으로 보고서가 생성됩니다:

```python
# 평가 실행 (Frontend 또는 API)
POST /api/evaluations/{eval_run_id}/execute

# 평가 완료 후 자동으로:
# 1. 평가 결과 저장
# 2. PDF 보고서 생성 (WeasyPrint 설치 시)
# 3. storage/reports/에 저장
```

**SSE 이벤트 로그:**
```
평가 완료!
PDF 보고서 생성 중...
보고서 생성 완료: reports/evaluation_xxx_20251229_140530.pdf
```

### 수동 생성 (Python 코드)

```python
from app.services.report_generation_service import report_generation_service
from sqlalchemy.ext.asyncio import AsyncSession

async def generate_report_manually(db: AsyncSession, evaluation_id: UUID):
    """평가 결과로부터 수동으로 보고서 생성"""

    # 의존성 확인
    deps = report_generation_service.check_dependencies()
    print(f"WeasyPrint: {deps['weasyprint']}")
    print(f"Matplotlib: {deps['matplotlib']}")

    if not deps['weasyprint']:
        print("WeasyPrint 미설치. pip install weasyprint 실행 필요")
        return

    # 보고서 생성
    report_path = await report_generation_service.generate_evaluation_report(
        db=db,
        evaluation_id=evaluation_id,
        include_charts=True  # 그래프 포함
    )

    print(f"보고서 생성 완료: {report_path}")
    return report_path
```

## 📊 생성되는 콘텐츠

### 1. 표지
- 실험 일시, 실험자, 시스템 정보

### 2. 모델 및 데이터셋 정보
- 모델: 이름, 아키텍처, 입력 크기, 클래스 수
- 데이터셋: 이름, 이미지 수, 타겟 클래스

### 3. 공격 기법 정보
- 공격 유형, 파라미터, 실험 환경

### 4. 정량적 결과
- **성능 비교 표**: Clean vs Attacked 메트릭
  - mAP@0.5, mAP@0.75, mAP@0.5:0.95
  - Precision, Recall, F1 Score
  - 탐지 성공률

- **공격 효과 지표**:
  - ΔmAP (감소율 %)
  - Attack Success Rate (ASR)
  - False Negative Rate 증가
  - Average Confidence 감소

- **그래프** (Matplotlib 설치 시):
  - Clean vs Attacked 막대 그래프 (6개 지표)
  - Precision-Recall Curve
  - 클래스별 AP 비교
  - Confidence Score 분포

### 5. 결론
- 주요 발견사항
- 권장사항

## 🔍 문제 해결

### WeasyPrint 설치 오류

```bash
# Cairo 라이브러리 확인
pkg-config --modversion cairo

# Pango 라이브러리 확인
pkg-config --modversion pango

# 재설치
sudo apt-get install --reinstall libcairo2-dev libpango1.0-dev
pip install --force-reinstall weasyprint
```

### 한글 폰트 문제

```bash
# 폰트 설치 확인
fc-list | grep -i nanum

# 폰트 캐시 재생성
fc-cache -f -v

# 폰트 없으면 설치
sudo apt-get install fonts-nanum fonts-nanum-coding
```

### 보고서 생성 건너뜀

로그에 `WeasyPrint 미설치로 보고서 생성 건너뜀`이 표시되면:

```bash
# 의존성 확인
python -c "from app.services.report_generation_service import report_generation_service; print(report_generation_service.check_dependencies())"

# WeasyPrint 설치
pip install -r requirements-report.txt
```

### Docker 환경

```dockerfile
# Dockerfile에 추가
RUN apt-get update && apt-get install -y \
    libcairo2-dev \
    libpango1.0-dev \
    fonts-nanum \
    fonts-nanum-coding \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements-report.txt
```

## 📝 커스터마이징

### 템플릿 수정

템플릿 파일 수정:
- `backend/app/templates/adversarial_report/report_template.html`
- `backend/app/templates/adversarial_report/style.css`

### 보고서 내용 추가

`backend/app/services/report_generation_service.py`의 `_prepare_report_data()` 메서드 수정:

```python
async def _prepare_report_data(self, db, evaluation, model, dataset, include_charts):
    data = {
        # 기존 필드...

        # 새로운 필드 추가
        'CUSTOM_FIELD': 'Custom Value',
    }
    return data
```

## 🎯 성능 최적화

### 그래프 생성 비활성화

그래프 생성 시간이 길 경우:

```python
report_path = await report_generation_service.generate_evaluation_report(
    db=db,
    evaluation_id=evaluation_id,
    include_charts=False  # 그래프 생성 비활성화
)
```

### 비동기 생성 (백그라운드)

```python
import asyncio

# 백그라운드에서 보고서 생성
asyncio.create_task(
    report_generation_service.generate_evaluation_report(
        db=db,
        evaluation_id=evaluation_id
    )
)
```

## 📞 지원

문제가 발생하면:
1. 로그 확인: 서버 콘솔에서 상세 에러 확인
2. 의존성 확인: `check_dependencies()` 메서드 실행
3. 시스템 패키지: Cairo, Pango 설치 확인
4. 폰트 설치: NanumGothic 폰트 확인

---

**Army AI Defense System - Report Generation Feature**
Version: 1.0
Last Updated: 2025-12-29
