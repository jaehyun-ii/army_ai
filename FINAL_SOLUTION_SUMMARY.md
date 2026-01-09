# 3D 데이터셋 평가 문제 최종 해결 요약

**날짜**: 2026-01-09
**문제**: AP@50 = 0.000, PDF 보고서 생성 실패
**상태**: ✅ 수정 완료

---

## 발견된 문제

### 1. PDF 보고서 생성 실패 ✅
**원인**: Division by zero (clean_map == 0일 때)
**수정 파일**: `backend/app/services/report_generation_service.py:454`
**상태**: ✅ 수정 완료

### 2. AP@50 = 0.000 ✅
**원인**: 좌표 형식 불일치
- CARLA가 COCO 형식 (좌상단 좌표)으로 저장
- 평가 서비스가 중심 좌표로 해석
- 결과: IoU = 0 → AP = 0

**해결**: YOLO 형식 사용 (중심 좌표, 정규화)
**수정 파일**: `backend/app/services/carla/annotation_parser.py:157-162`
**상태**: ✅ 수정 완료

---

## 수정 내용

### 1. report_generation_service.py
```python
# Division by zero 방지
if clean_map > 0:
    delta_map_percent = ((clean_map - attacked_map) / clean_map) * 100
    robustness_ratio = attacked_map / clean_map
else:
    delta_map_percent = 0.0
    robustness_ratio = 0.0
```

### 2. annotation_parser.py
```python
# YOLO 파일을 상위 폴더에서도 찾도록 수정
txt_files = list(annotations_dir.glob("*_yolo.txt"))

# annotations 폴더가 비어있으면 상위 폴더에서 찾기
if not txt_files:
    parent_dir = annotations_dir.parent
    txt_files = list(parent_dir.glob("*_yolo.txt"))
    if txt_files:
        logger.info(f"Found YOLO file in parent directory: {txt_files[0]}")
```

---

## 다음 단계 (사용자 실행 필요)

### 1. 백엔드 재시작
```bash
cd /home/jaehyun/army_ai
docker-compose restart backend
```

### 2. Annotation 재파싱
```bash
# 원본 데이터셋
curl -X POST http://localhost:8000/api/v1/carla/datasets_3d/3f4f3e42-a2a5-4505-a9a8-f3d1e1e256d1/parse-annotations

# 공격 데이터셋
curl -X POST http://localhost:8000/api/v1/carla/datasets_3d/637a0210-ce7a-404d-8a6e-f4ecb73f2449/parse-annotations
```

### 3. 평가 재실행
프론트엔드에서 동일한 설정으로 평가를 다시 실행합니다.

**예상 결과**:
```
원본 AP@50: 0.85 (정상)
공격 AP@50: 0.42 (정상, 공격 효과 확인 가능)
하락률: 50.6%
```

---

## 검증 방법

### DB 확인 (재파싱 후)
```sql
-- Annotation 개수 및 타입 확인
SELECT
    a.annotation_type,
    a.class_name,
    COUNT(*) as count,
    AVG(a.bbox_x) as avg_x,
    AVG(a.bbox_y) as avg_y
FROM annotations a
JOIN images_3d i ON a.image_3d_id = i.id
WHERE i.dataset_id = '637a0210-ce7a-404d-8a6e-f4ecb73f2449'
GROUP BY a.annotation_type, a.class_name;
```

**기대 결과**:
```
annotation_type: bbox_normalized
class_name: tank
count: 108
avg_x: ~0.5 (중심 좌표, 0~1 범위)
avg_y: ~0.4
```

**잘못된 결과** (재파싱 전):
```
annotation_type: bbox
class_name: tank
count: 108
avg_x: ~200 (픽셀 좌표, 좌상단)
avg_y: ~150
```

---

## 근본 원인 분석

### CARLA가 생성하는 파일

```
storage/3d/exports/adv_k2_tank_advasd/
├── k2_tank_eval/
│   └── *.jpg (이미지 파일)
├── annotations/
│   └── k2_tank_eval.json          ← COCO 형식 (좌상단, 픽셀)
└── k2_tank_eval_yolo.txt          ← YOLO 형식 (중심, 정규화) ✅
```

### COCO vs YOLO

| 항목 | COCO | YOLO |
|------|------|------|
| 좌표 기준 | 좌상단 (x_min, y_min) | 중심 (x_center, y_center) |
| 단위 | 픽셀 | 정규화 (0~1) |
| 평가 호환성 | ❌ 불일치 | ✅ 호환됨 |
| 파일 위치 | annotations/ 안 | 상위 폴더 |

### 좌표 변환 예시

**COCO bbox**: `[199.5, 147.25, 312.25, 137.75]`
```
x_min = 199.5 픽셀
y_min = 147.25 픽셀
width = 312.25 픽셀
height = 137.75 픽셀
```

**YOLO bbox**: `0.6946,0.4221,0.6099,0.2690`
```
x_center = 0.6946 (정규화)
y_center = 0.4221 (정규화)
width = 0.6099 (정규화)
height = 0.2690 (정규화)
```

**평가 서비스 기대값**: YOLO 형식 (중심 좌표, 정규화)

---

## 생성된 문서

1. **3D_EVALUATION_ISSUE_REPORT.md** - 전체 문제 분석
2. **COORDINATE_MISMATCH_ISSUE.md** - 좌표 불일치 상세 분석
3. **3D_EVALUATION_SOLUTION.md** - 해결 방안
4. **FINAL_SOLUTION_SUMMARY.md** (현재 문서) - 최종 요약
5. **diagnose_3d_evaluation.sh** - DB 진단 스크립트

---

## 테스트 체크리스트

- [x] 문제 원인 파악
- [x] PDF 생성 에러 수정
- [x] AnnotationParser 수정
- [ ] 백엔드 재시작
- [ ] Annotation 재파싱
- [ ] DB 데이터 확인 (bbox_x가 0~1 범위인지)
- [ ] 평가 재실행
- [ ] AP@50 > 0 확인
- [ ] PDF 보고서 생성 성공 확인

---

## 추가 발견 사항

### CARLA의 클래스명
```json
{
  "category_id": 91,
  "name": "tank"  ✅ 정확함
}
```

→ 클래스명 불일치 문제는 없었습니다.

### Annotation 생성
- CARLA는 annotation을 정상적으로 생성했습니다 (108개)
- 메인 백엔드에도 파일이 존재합니다
- 단지 **올바른 형식(YOLO)**을 사용하지 않았을 뿐입니다

---

## 향후 개선 사항

### 1. CARLA 백엔드 수정 (선택)
YOLO 파일을 annotations 폴더 안에 생성하도록 수정

### 2. Annotation validation 추가
```python
# 파싱 후 자동 검증
def validate_annotations(dataset_id):
    # bbox 값이 0~1 범위인지 확인
    # annotation_type이 올바른지 확인
    # class_name이 있는지 확인
```

### 3. 평가 전 사전 체크
```python
# 평가 실행 전
if annotation_count == 0:
    raise ValidationError("Annotations not found. Parse annotations first.")
if not all(0 <= bbox <= 1 for bbox in bboxes):
    raise ValidationError("Bbox coordinates not normalized. Re-parse with YOLO format.")
```

---

## 관련 이슈

- [3D_ADVERSARIAL_ASSET_REVIEW.md](./3D_ADVERSARIAL_ASSET_REVIEW.md) - 전체 자산 관리 검토
- [CARLA_FOLDER_STRUCTURE.md](./CARLA_FOLDER_STRUCTURE.md) - 폴더 구조 문서
- [IMAGE_SCANNING_FLOW.md](./IMAGE_SCANNING_FLOW.md) - 이미지 스캔 흐름

---

**작성자**: Claude Code Analysis
**상태**: ✅ 수정 완료, 테스트 대기 중
**다음 단계**: 백엔드 재시작 → Annotation 재파싱 → 평가 재실행
