# 3D 평가 AP@50 = 0.000 문제 - 좌표 형식 불일치

**날짜**: 2026-01-09
**심각도**: 🔴 **CRITICAL** - 모든 3D 데이터셋 평가에 영향

---

## 문제 요약

CARLA가 생성한 annotation이 DB에 저장될 때와 평가 시 읽을 때 **좌표 형식이 다르게 해석**되어 IoU = 0이 됩니다.

---

## 근본 원인

### 1. CARLA가 생성하는 COCO 형식 (확인 완료)

**파일**: `/workspace/exports/adv_k2_tank_advasd/annotations/k2_tank_eval.json`

```json
{
  "bbox": [199.5, 147.25, 312.25, 137.75],  // ← [x_min, y_min, width, height] (픽셀 좌표)
  "category_id": 91
}
{
  "categories": [
    {"id": 91, "name": "tank"}  // ← 클래스명은 "tank" (올바름)
  ]
}
```

**좌표 의미**:
- `bbox[0]` (199.5): 좌상단 X 좌표
- `bbox[1]` (147.25): 좌상단 Y 좌표
- `bbox[2]` (312.25): 박스 너비
- `bbox[3]` (137.75): 박스 높이

---

### 2. AnnotationParser가 DB에 저장하는 방식

**파일**: `backend/app/services/carla/annotation_parser.py:239-242`

```python
annotation_records.append(Annotation(
    bbox_x=x,           # ← COCO bbox[0] = 좌상단 X (199.5)
    bbox_y=y,           # ← COCO bbox[1] = 좌상단 Y (147.25)
    bbox_width=w,       # ← COCO bbox[2] = 너비 (312.25)
    bbox_height=h,      # ← COCO bbox[3] = 높이 (137.75)
    ...
))
```

**저장된 DB 값 (예시)**:
```
bbox_x: 199.5      (좌상단 X)
bbox_y: 147.25     (좌상단 Y)
bbox_width: 312.25 (너비)
bbox_height: 137.75 (높이)
```

---

### 3. EvaluationService가 DB에서 읽는 방식

**파일**: `backend/app/services/evaluation_service.py:881-915`

```python
# ❌ 문제: bbox_x를 중심 좌표로 가정!
x_center_val = float(ann.bbox_x)  # 199.5를 중심 X로 해석
y_center_val = float(ann.bbox_y)  # 147.25를 중심 Y로 해석
w_val = float(ann.bbox_width)     # 312.25
h_val = float(ann.bbox_height)    # 137.75

# 정규화 여부 체크
is_normalized = (
    0.0 <= x_center_val <= 1.0 and  # 199.5 > 1.0 → False
    0.0 <= y_center_val <= 1.0 and
    ...
)

if not is_normalized and gt_image_width and gt_image_height:
    # ❌ 잘못된 정규화: 좌상단 좌표를 중심 좌표로 착각하고 정규화
    x_center_norm = x_center_val / gt_image_width  # 199.5 / 512 = 0.389
    y_center_norm = y_center_val / gt_image_height # 147.25 / 512 = 0.287
    w_norm = w_val / gt_image_width                # 312.25 / 512 = 0.609
    h_norm = h_val / gt_image_height               # 137.75 / 512 = 0.269

# ❌ 잘못된 변환: 중심 좌표를 기준으로 좌표 계산
x1 = x_center_norm - w_norm / 2  # 0.389 - 0.305 = 0.084
y1 = y_center_norm - h_norm / 2  # 0.287 - 0.135 = 0.152
x2 = x_center_norm + w_norm / 2  # 0.389 + 0.305 = 0.694
y2 = y_center_norm + h_norm / 2  # 0.287 + 0.135 = 0.422
```

**실제 정답 (올바른 변환)**:
```python
# bbox_x, bbox_y를 좌상단 좌표로 해석해야 함
x_min = 199.5 / 512 = 0.389
y_min = 147.25 / 512 = 0.287
x_max = (199.5 + 312.25) / 512 = 1.0 (클리핑됨)
y_max = (147.25 + 137.75) / 512 = 0.556

# 정규화된 bbox: (0.389, 0.287, 1.0, 0.556)
```

**평가 코드가 계산한 잘못된 값**:
```
# bbox: (0.084, 0.152, 0.694, 0.422)
# → 실제 bbox와 완전히 다른 위치!
```

---

## 결과

**Ground Truth bbox**: 실제 tank 위치 (오른쪽 상단)
**Prediction bbox**: 모델이 탐지한 위치 (중간 어딘가)

**평가 시 계산된 GT bbox**: 왼쪽 상단의 작은 영역 (완전히 잘못된 위치)

→ **IoU = 0** → **AP = 0.000**

---

## 해결 방안

### Option 1: AnnotationParser를 YOLO 형식으로 수정 (권장 ✅)

CARLA가 이미 YOLO 형식 txt 파일도 생성하므로, 이것을 우선적으로 사용하도록 수정합니다.

**파일**: `backend/app/services/carla/annotation_parser.py:153-163`

현재 코드는 이미 YOLO 형식을 우선으로 처리합니다:
```python
# YOLO 형식 TXT 파일 찾기 (우선순위)
txt_files = list(annotations_dir.glob("*_yolo.txt"))
if txt_files:
    return await AnnotationParser.parse_yolo_txt_and_save(...)
```

**YOLO 형식 확인**:
```bash
# CARLA 컨테이너에서
cat /workspace/exports/adv_k2_tank_advasd/k2_tank_eval_yolo.txt | head -5
```

**YOLO 형식**:
```csv
filename,class_id,x_center,y_center,width,height
k2_tank_3_0.jpg,0,0.6992,0.4170,0.6103,0.2691
```

이미 **중심 좌표 + 정규화**되어 있으므로, evaluation_service와 호환됩니다.

**조치**:
1. CARLA가 `*_yolo.txt` 파일을 생성하는지 확인
2. 파일명 패턴이 정확한지 확인 (`glob("*_yolo.txt")`)

---

### Option 2: COCO bbox를 중심 좌표로 변환하여 저장

**파일**: `backend/app/services/carla/annotation_parser.py:227-242`

```python
# 수정 전
x, y, w, h = bbox
annotation_records.append(Annotation(
    bbox_x=x,           # 좌상단 X
    bbox_y=y,           # 좌상단 Y
    bbox_width=w,
    bbox_height=h,
    ...
))

# 수정 후
x, y, w, h = bbox
# COCO [x_min, y_min, w, h]를 중심 좌표로 변환
x_center = x + w / 2
y_center = y + h / 2

# 이미지 크기로 정규화 (images 정보에서 가져오기)
image_info = next((img for img in images if img["id"] == image_id_coco), None)
if image_info:
    img_width = image_info.get("width", 512)
    img_height = image_info.get("height", 512)

    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    w_norm = w / img_width
    h_norm = h / img_height

    annotation_records.append(Annotation(
        bbox_x=x_center_norm,      # 정규화된 중심 X
        bbox_y=y_center_norm,      # 정규화된 중심 Y
        bbox_width=w_norm,         # 정규화된 너비
        bbox_height=h_norm,        # 정규화된 높이
        annotation_type="bbox_normalized",  # 타입 명시
        ...
    ))
```

---

### Option 3: EvaluationService에서 COCO 형식 처리 추가

**파일**: `backend/app/services/evaluation_service.py:880-920`

```python
# annotation_type 확인 추가
annotation_type = ann.annotation_type or "bbox"  # 기본값: bbox

if annotation_type == "bbox":  # COCO 형식 (좌상단 좌표)
    # bbox_x, bbox_y를 좌상단 좌표로 해석
    x_min = float(ann.bbox_x)
    y_min = float(ann.bbox_y)
    w_val = float(ann.bbox_width)
    h_val = float(ann.bbox_height)

    # 정규화 여부 체크
    is_normalized = (0.0 <= x_min <= 1.0 and ...)

    if not is_normalized:
        x_min_norm = x_min / gt_image_width
        y_min_norm = y_min / gt_image_height
        w_norm = w_val / gt_image_width
        h_norm = h_val / gt_image_height
    else:
        x_min_norm = x_min
        y_min_norm = y_min
        w_norm = w_val
        h_norm = h_val

    # 좌상단 + 크기 → x1, y1, x2, y2 변환
    x1 = x_min_norm
    y1 = y_min_norm
    x2 = x_min_norm + w_norm
    y2 = y_min_norm + h_norm

elif annotation_type == "bbox_normalized":  # YOLO 형식 (중심 좌표)
    # 기존 로직 (중심 좌표로 처리)
    x_center_norm = float(ann.bbox_x)
    y_center_norm = float(ann.bbox_y)
    ...
```

---

## 권장 해결 순서

### 1단계: YOLO 파일 확인
```bash
# CARLA 컨테이너에서
ls -la /workspace/exports/adv_k2_tank_advasd/*.txt
cat /workspace/exports/adv_k2_tank_advasd/*_yolo.txt | head -5
```

### 2단계: 메인 백엔드에서도 확인
```bash
ls -la /home/jaehyun/army_ai/storage/3d/exports/adv_k2_tank_advasd/
```

### 3단계: annotation 재파싱
```bash
# YOLO 파일이 있다면
curl -X POST http://localhost:8000/api/v1/carla/datasets_3d/{dataset_id}/parse-annotations
```

### 4단계: 평가 재실행
- 동일한 설정으로 평가 재실행
- AP@50 > 0인지 확인

---

## 테스트 결과 예상

### 현재 (COCO 형식 사용 시)
```
원본 AP@50: 0.000  ← 좌표 불일치
공격 AP@50: 0.000  ← 좌표 불일치
```

### 수정 후 (YOLO 형식 사용 시)
```
원본 AP@50: 0.85   ← 정상
공격 AP@50: 0.42   ← 정상 (공격 효과 확인)
```

---

## 파일 위치 요약

| 구분 | 파일 경로 | 비고 |
|------|----------|------|
| CARLA JSON | `/workspace/exports/adv_k2_tank_advasd/annotations/k2_tank_eval.json` | COCO 형식 |
| CARLA YOLO | `/workspace/exports/adv_k2_tank_advasd/k2_tank_eval_yolo.txt` | YOLO 형식 (확인 필요) |
| 메인 백엔드 | `/home/jaehyun/army_ai/storage/3d/exports/adv_k2_tank_advasd/` | 볼륨 마운트 |

---

## 참고 자료

- **COCO 형식**: `[x_min, y_min, width, height]` (좌상단 좌표)
- **YOLO 형식**: `[x_center, y_center, width, height]` (중심 좌표, 정규화)
- **IoU 계산**: 두 bbox가 정확히 같은 좌표 형식이어야 함

---

**작성자**: Claude Code Analysis
**다음 단계**: YOLO 파일 확인 및 annotation 재파싱
