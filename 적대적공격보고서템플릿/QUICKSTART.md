# 🚀 빠른 시작 가이드

## 1단계: 설치

```bash
cd 적대적공격보고서템플릿

# Python 라이브러리 설치
pip install -r requirements.txt

# 한글 폰트 설치 (Linux)
sudo apt-get install fonts-nanum fonts-nanum-coding
```

## 2단계: 예시 보고서 생성

```bash
# 예시 데이터로 PDF 보고서 생성
python generate_report.py --data example_data.json --output test_report.pdf
```

출력 예시:
```
============================================================
적대적 공격 보고서 생성 시작
============================================================

[1/4] 그래프 생성 중...
✓ 막대 그래프 생성: temp_charts/metrics_bar_chart.png
✓ PR Curve 생성: temp_charts/pr_curve.png
✓ 클래스별 AP 그래프 생성: temp_charts/per_class_ap.png
✓ Confidence 분포 그래프 생성: temp_charts/confidence_distribution.png

[2/4] 템플릿 로딩 중...
✓ 템플릿 로드 완료: report_template.html

[3/4] 데이터 치환 중...
✓ 데이터 치환 완료

[4/4] PDF 생성 중...
✓ PDF 생성 완료: test_report.pdf
  파일 크기: 1234.5 KB

============================================================
보고서 생성 완료!
============================================================

✅ 성공: test_report.pdf
```

## 3단계: 보고서 확인

```bash
# Linux
xdg-open test_report.pdf

# macOS
open test_report.pdf

# Windows
start test_report.pdf
```

## 커스터마이징

### 자신의 데이터로 보고서 생성

1. **데이터 준비** (`my_data.json`):

```json
{
  "EXPERIMENT_DATE": "2025-12-29",
  "EXPERIMENTER_NAME": "홍길동",
  "MODEL_NAME": "YOLOv8n",
  "ATTACK_TYPE": "PGD",

  "clean_metrics": {
    "f1": 0.85,
    "ap50": 0.82,
    "ap75": 0.71,
    "ap": 0.65,
    "precision": 0.87,
    "recall": 0.83
  },

  "attacked_metrics": {
    "f1": 0.34,
    "ap50": 0.31,
    "ap75": 0.22,
    "ap": 0.18,
    "precision": 0.42,
    "recall": 0.28
  },

  "CLEAN_MAP50": "0.82",
  "ATTACKED_MAP50": "0.31",
  "DELTA_MAP50": "-62.2%"
}
```

2. **보고서 생성**:

```bash
python generate_report.py --data my_data.json --output my_report.pdf
```

### Python 코드에서 사용

```python
from generate_report import AdversarialReportGenerator
import json

# 데이터 로드
with open('my_data.json', 'r') as f:
    data = json.load(f)

# 보고서 생성
generator = AdversarialReportGenerator()
generator.generate_report(
    data=data,
    output_pdf='my_report.pdf',
    generate_charts=True
)
generator.cleanup()
```

## 문제 해결

### 한글이 깨져서 나올 때

```bash
# 한글 폰트 설치 확인
fc-list | grep -i nanum

# 폰트가 없으면 설치
sudo apt-get install fonts-nanum fonts-nanum-coding
```

### WeasyPrint 설치 오류

```bash
# Cairo 라이브러리 설치 (Ubuntu/Debian)
sudo apt-get install libcairo2-dev libpango1.0-dev

# macOS
brew install cairo pango gdk-pixbuf libffi

# 그 후 다시 시도
pip install weasyprint
```

### 그래프가 생성되지 않을 때

```bash
# matplotlib 백엔드 확인
python -c "import matplotlib; print(matplotlib.get_backend())"

# Agg 백엔드로 설정되어 있어야 함 (GUI 불필요)
```

## 다음 단계

- 📖 전체 문서: [README.md](README.md)
- 💡 사용 예시: [example_usage.py](example_usage.py)
- 📊 예시 데이터: [example_data.json](example_data.json)
- 🌐 HTML 미리보기: [example_report.html](example_report.html)

---

**문의사항이 있으시면 프로젝트 관리자에게 연락하세요.**
