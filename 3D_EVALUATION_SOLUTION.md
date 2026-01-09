# 3D 데이터셋 평가 AP@50 = 0 문제 해결 방안

**날짜**: 2026-01-09
**상태**: ✅ 근본 원인 파악 완료, 수정 준비 완료

---

## 문제 요약

1. ✅ **PDF 보고서 생성 실패** - Division by zero (해결 완료)
2. 🔍 **AP@50 = 0.000** - 좌표 형식 불일치 (해결 방안 확인)

---

## 근본 원인

### 발견된 사실

#### 1. CARLA는 두 가지 형식으로 annotation 생성
- **COCO JSON**: `annotations/k2_tank_eval.json` (좌상단 좌표, 픽셀 단위)
- **YOLO TXT**: `k2_tank_eval_yolo.txt` (중심 좌표, 정규화) ✅

#### 2. 클래스명은 정확함
```json
{
  "category_id": 91,
  "categories": [{"id": 91, "name": "tank"}]
}
```
→ 클래스명 불일치 문제는 **아님**

#### 3. 좌표 형식 불일치
**COCO 형식** (현재 사용 중):
```
bbox: [199.5, 147.25, 312.25, 137.75]  // [x_min, y_min, width, height]
                                         // 좌상단 좌표, 픽셀 단위
```

DB에 저장: `bbox_x=199.5, bbox_y=147.25` (좌상단 좌표)
평가 시 해석: `bbox_x=중심X, bbox_y=중심Y` (잘못된 해석!)

→ **IoU = 0** → **AP = 0.000**

**YOLO 형식** (사용해야 할 것):
```csv
filename,class_id,x_center,y_center,width,height
k2_tank_3_0.jpg,91,0.6946,0.4221,0.6099,0.2690  // 중심 좌표, 정규화됨
```

#### 4. YOLO 파일 위치 문제
- **실제 위치**: `/storage/3d/exports/adv_k2_tank_advasd/k2_tank_eval_yolo.txt`
- **파서가 찾는 위치**: `/storage/3d/exports/adv_k2_tank_advasd/annotations/*_yolo.txt`

→ AnnotationParser가 YOLO 파일을 못 찾고 COCO JSON을 사용함

---

## 해결 방안

### 방법 1: AnnotationParser 수정 (권장 ✅)

**파일**: `backend/app/services/carla/annotation_parser.py`

**수정 위치**: Line 152-163

```python
# 수정 전
def parse_and_save(...):
    # YOLO 형식 TXT 파일 찾기 (우선순위)
    txt_files = list(annotations_dir.glob("*_yolo.txt"))  # ❌ annotations 폴더 안에서만 찾음
    if txt_files:
        ...

# 수정 후
def parse_and_save(...):
    # YOLO 형식 TXT 파일 찾기 (우선순위)
    # 1) annotations 폴더 안에서 찾기
    txt_files = list(annotations_dir.glob("*_yolo.txt"))

    # 2) annotations 폴더가 비어있으면 상위 폴더에서 찾기
    if not txt_files:
        parent_dir = annotations_dir.parent
        txt_files = list(parent_dir.glob("*_yolo.txt"))
        if txt_files:
            logger.info(f"Found YOLO file in parent directory: {txt_files[0]}")

    if txt_files:
        annotation_file = txt_files[0]
        logger.info(f"Found YOLO format annotation file: {annotation_file}")
        return await AnnotationParser.parse_yolo_txt_and_save(...)
```

---

### 방법 2: COCO bbox를 중심 좌표로 변환하여 저장 (차선책)

**파일**: `backend/app/services/carla/annotation_parser.py:227-246`

```python
# 수정 전
x, y, w, h = bbox
annotation_records.append(Annotation(
    bbox_x=x,           # 좌상단 X (픽셀)
    bbox_y=y,           # 좌상단 Y (픽셀)
    bbox_width=w,
    bbox_height=h,
    annotation_type="bbox",
))

# 수정 후
x, y, w, h = bbox
x_center = x + w / 2  # 중심 좌표 계산
y_center = y + h / 2

# 이미지 크기 가져오기
image_info = next((img for img in images if img["id"] == image_id_coco), None)
if image_info:
    img_width = image_info.get("width", 512)
    img_height = image_info.get("height", 512)

    # 정규화
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    w_norm = w / img_width
    h_norm = h / img_height

    annotation_records.append(Annotation(
        bbox_x=x_center_norm,      # 정규화된 중심 X
        bbox_y=y_center_norm,      # 정규화된 중심 Y
        bbox_width=w_norm,
        bbox_height=h_norm,
        annotation_type="bbox_normalized",  # 타입 명시
    ))
```

---

## 수정 코드 (방법 1)

```python
# backend/app/services/carla/annotation_parser.py

@staticmethod
async def parse_and_save(
    annotations_dir: Path,
    dataset_id: UUID,
    image_id_map: Dict[str, UUID],
    db: AsyncSession
) -> int:
    """
    CARLA에서 생성된 어노테이션 파일을 파싱하여 DB에 저장.
    JSON (COCO 형식) 또는 TXT (YOLO 형식)를 자동으로 감지하여 처리합니다.
    """
    try:
        # YOLO 형식 TXT 파일 찾기 (우선순위)
        # 1) annotations 폴더 안에서 찾기
        txt_files = list(annotations_dir.glob("*_yolo.txt"))

        # 2) annotations 폴더가 비어있으면 상위 폴더에서 찾기 (CARLA가 상위에 생성하는 경우)
        if not txt_files:
            parent_dir = annotations_dir.parent
            txt_files = list(parent_dir.glob("*_yolo.txt"))
            if txt_files:
                logger.info(f"Found YOLO file in parent directory: {txt_files[0]}")

        if txt_files:
            annotation_file = txt_files[0]
            logger.info(f"Found YOLO format annotation file: {annotation_file}")
            return await AnnotationParser.parse_yolo_txt_and_save(
                txt_file=annotation_file,
                dataset_id=dataset_id,
                image_id_map=image_id_map,
                db=db
            )

        # COCO 형식 JSON 파일 찾기 (YOLO가 없을 때만)
        json_files = list(annotations_dir.glob("*.json"))
        if not json_files:
            logger.warning(f"No annotation files (JSON or YOLO TXT) found in {annotations_dir}")
            return 0

        # ... (나머지 COCO 처리 로직)
```

---

## 테스트 계획

### 1단계: 코드 수정 적용
- AnnotationParser에 상위 폴더 검색 로직 추가

### 2단계: Annotation 재파싱
```bash
# 원본 데이터셋
curl -X POST http://localhost:8000/api/v1/carla/datasets_3d/3f4f3e42-a2a5-4505-a9a8-f3d1e1e256d1/parse-annotations

# 공격 데이터셋
curl -X POST http://localhost:8000/api/v1/carla/datasets_3d/637a0210-ce7a-404d-8a6e-f4ecb73f2449/parse-annotations
```

### 3단계: DB 확인
```sql
-- annotation_type이 bbox_normalized인지 확인
SELECT
    a.annotation_type,
    a.class_name,
    a.bbox_x,
    a.bbox_y,
    a.bbox_width,
    a.bbox_height
FROM annotations a
JOIN images_3d i ON a.image_3d_id = i.id
WHERE i.dataset_id = '637a0210-ce7a-404d-8a6e-f4ecb73f2449'
LIMIT 5;
```

**예상 결과**:
```
annotation_type: bbox_normalized
class_name: tank
bbox_x: 0.6946 (중심 X, 정규화)
bbox_y: 0.4221 (중심 Y, 정규화)
bbox_width: 0.6099 (정규화)
bbox_height: 0.2690 (정규화)
```

### 4단계: 평가 재실행
- 동일한 모델과 데이터셋으로 평가 재실행
- AP@50 > 0 확인

**예상 결과**:
```
원본 AP@50: 0.85  ← 정상
공격 AP@50: 0.42  ← 공격 효과 확인 가능
하락률: 50.6%
```

---

## 파일 구조 정리

```
storage/3d/exports/adv_k2_tank_advasd/
├── k2_tank_eval/               # 이미지 폴더
│   ├── k2_tank_3_0.jpg
│   └── ...
├── annotations/                # Annotation 폴더
│   └── k2_tank_eval.json      # COCO 형식 (좌상단 좌표)
└── k2_tank_eval_yolo.txt      # ✅ YOLO 형식 (중심 좌표, 정규화)
                                # ⚠️ annotations 폴더 밖에 있음!
```

---

## 추가 개선 사항

### 1. CARLA 백엔드 수정 (선택)
YOLO 파일을 annotations 폴더 안에 생성하도록 수정:

```python
# CARLA backend: utils/eval_dataset_gen.py
# save_yolo_csv() 함수 수정
save_path = os.path.join(self.base_save_dir, "annotations", f"{csv_filename}")
```

### 2. Dataset metadata에 annotation_format 추가
```python
dataset.metadata_ = {
    "annotation_format": "yolo",  # or "coco"
    "classes": ["tank"],
    ...
}
```

---

## 검증 체크리스트

- [ ] AnnotationParser 수정 완료
- [ ] 백엔드 재시작
- [ ] Annotation 재파싱 실행
- [ ] DB에서 bbox_x, bbox_y 값 확인 (0~1 범위인지)
- [ ] DB에서 annotation_type 확인 (bbox_normalized인지)
- [ ] DB에서 class_name 확인 (tank인지)
- [ ] 평가 재실행
- [ ] AP@50 > 0 확인
- [ ] PDF 보고서 생성 성공 확인

---

## 관련 파일

1. **COORDINATE_MISMATCH_ISSUE.md** - 상세 기술 분석
2. **3D_EVALUATION_ISSUE_REPORT.md** - 전체 문제 보고서
3. **diagnose_3d_evaluation.sh** - DB 진단 스크립트

---

**작성자**: Claude Code Analysis
**상태**: 수정 준비 완료
**다음 단계**: AnnotationParser 코드 수정
