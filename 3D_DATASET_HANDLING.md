# 3D 데이터셋 처리 방식 문서

**날짜**: 2025-12-24
**목적**: 2D vs 3D 데이터셋의 차이점 및 평가 시스템 처리 방식 설명

---

## 📋 핵심 차이점

### 2D Attack Datasets vs 3D Attack Datasets

#### 2D Attack Datasets (`attack_datasets_2d`)
```sql
CREATE TABLE attack_datasets_2d (
  id uuid PRIMARY KEY,
  base_dataset_id uuid REFERENCES datasets_2d(id),  -- ✅ 있음
  output_dataset_id uuid REFERENCES datasets_2d(id),
  ...
);
```

**특징**:
- ✅ `base_dataset_id` 필드 **존재**
- ✅ 원본 데이터셋에서 공격 데이터셋 생성
- ✅ 1:1 매핑 관계 명확

#### 3D Attack Datasets (`attack_datasets_3d`)
```sql
CREATE TABLE attack_datasets_3d (
  id uuid PRIMARY KEY,
  -- ❌ base_dataset_id 필드 없음!
  output_dataset_id uuid REFERENCES datasets_3d(id),  -- ✅ 있음
  ...
);
```

**특징**:
- ❌ `base_dataset_id` 필드 **없음**
- ❌ CARLA 시뮬레이션 등으로 독립적으로 생성
- ❌ 원본 데이터셋과 직접적인 연결 없음

---

## 🎯 평가 시스템에서의 처리

### 시나리오별 처리 방식

#### 📌 시나리오 1: 베이스 데이터셋만 평가 (2D/3D 동일)

```
Input:
  - base_dataset_id (2D) 또는 base_dataset_3d_id (3D)
  - phase = "pre_attack"

Process:
  1. base dataset 로드
  2. 모델 추론
  3. GT와 비교하여 메트릭 계산

Output:
  - eval_dataset_results: 1개 (BASE)
```

#### 📌 시나리오 2: 공격 데이터셋만 평가

##### 🟦 2D의 경우
```
Input:
  - attack_dataset_id
  - phase = "post_attack"

Process:
  1. ✅ Auto-populate: base_dataset_id = attack_dataset.base_dataset_id
  2. 원본 데이터셋 평가 (BASE)
  3. 공격 데이터셋 평가 (ATTACK)
  4. Robustness 계산

Output:
  - eval_dataset_results: 2개 (BASE + ATTACK)
  - eval_runs.metrics_summary: robustness 정보 포함
```

##### 🟧 3D의 경우 (⭐ 중요!)

**Case A: base_dataset_3d_id 제공됨**
```
Input:
  - attack_dataset_3d_id
  - base_dataset_3d_id (사용자가 수동으로 선택)
  - phase = "post_attack"

Process:
  1. 원본 데이터셋 평가 (BASE)
  2. 공격 데이터셋 평가 (ATTACK)
     - GT는 원본 데이터셋에서 파일명 매핑으로 가져옴
  3. Robustness 계산

Output:
  - eval_dataset_results: 2개 (BASE + ATTACK)
  - eval_runs.metrics_summary: robustness 정보 포함
```

**Case B: base_dataset_3d_id 제공 안 됨** ⭐ NEW!
```
Input:
  - attack_dataset_3d_id
  - base_dataset_3d_id = null
  - phase = "post_attack"

Process:
  1. 원본 데이터셋 평가 생략
  2. 공격 데이터셋만 평가 (ATTACK)
     - GT는 output_dataset 자체의 annotations 사용
  3. Robustness 계산 생략 (비교할 원본이 없음)

Output:
  - eval_dataset_results: 1개 (ATTACK만)
  - eval_runs.metrics_summary: robustness 정보 없음
```

---

## 🔧 구현 상세

### 1. API Endpoint (`backend/app/api/v1/endpoints/evaluation.py`)

**수정 내용**: 3D auto-populate 로직 제거

```python
# ❌ 제거됨: 3D auto-populate (불가능)
# if eval_run.attack_dataset_3d_id and not eval_run.base_dataset_3d_id:
#     attack_dataset_3d = await ...
#     eval_run.base_dataset_3d_id = attack_dataset_3d.base_dataset_id  # 필드 없음!

# ✅ 유지됨: 2D auto-populate
if eval_run.attack_dataset_id and not eval_run.base_dataset_id:
    attack_dataset = await crud.attack_dataset_2d.get(db, id=eval_run.attack_dataset_id)
    if attack_dataset and attack_dataset.base_dataset_id:
        eval_run.base_dataset_id = attack_dataset.base_dataset_id
```

### 2. Evaluation Service (`backend/app/services/evaluation_service.py`)

**수정 내용**: 3D base_dataset_3d_id를 optional로 처리

```python
async def _evaluate_attack_dataset(...):
    if dimension == "3d":
        # 3D: base_dataset_3d_id is OPTIONAL
        original_dataset_id = eval_run.base_dataset_3d_id  # May be None
        attack_output_dataset_id = attack_dataset.output_dataset_id
    else:
        # 2D: base_dataset_id is auto-populated
        original_dataset_id = attack_dataset.base_dataset_id
        attack_output_dataset_id = attack_dataset.output_dataset_id

    # Step 1: Evaluate Original (if provided)
    if original_dataset_id:
        # Load original dataset
        # Run inference
        # Calculate metrics
        ...
    else:
        await eval_logger.info("원본 데이터셋 평가 생략")

    # Step 2: Evaluate Attack
    if original_images:
        # Use original GT (filename mapping)
        attack_metrics = await self._calculate_metrics_with_gt(
            images=original_images,
            use_filename_mapping=True,
            ...
        )
    else:
        # Use attack dataset's own GT
        attack_metrics = await self._calculate_metrics_with_gt(
            images=attack_images,
            use_filename_mapping=False,
            ...
        )

    # Step 3: Compare (if original provided)
    if original_metrics:
        robustness_metrics = calculate_robustness_metrics(...)
    else:
        await eval_logger.info("결과 비교 생략")

    # Save results
    if original_metrics:
        # Save BASE + ATTACK with robustness
        await self._save_evaluation_results(..., dataset_type=BASE)
        await self._save_evaluation_results(..., dataset_type=ATTACK, metrics with robustness)
    else:
        # Save ATTACK only without robustness
        await self._save_evaluation_results(..., dataset_type=ATTACK, metrics without robustness)
```

### 3. Database Schema (`database/complete_schema.sql`)

**추가된 문서**:

```sql
-- attack_datasets_3d 테이블
-- Important: Unlike 2D attack datasets, 3D attack datasets do NOT have a base_dataset_id field.
-- This is because 3D attacks are generated differently (e.g., CARLA simulations)
-- and may not have a direct 1:1 mapping to a base dataset.

-- eval_runs 테이블 constraint
-- Note for 2D: base_dataset_id is auto-populated from attack_dataset_2d.base_dataset_id
-- Note for 3D: base_dataset_3d_id must be manually selected (attack_datasets_3d has no base_dataset_id)
--   - If base_dataset_3d_id provided: Compare original vs attack (with robustness metrics)
--   - If base_dataset_3d_id NOT provided: Evaluate attack only (no comparison, no robustness)
```

---

## 🎯 사용자 가이드

### UI에서의 선택 방법

#### 2D 평가 생성
```
1. 베이스만: base_dataset_id 선택
2. 공격만: attack_dataset_id 선택 (base는 자동으로 채워짐)
3. 비교: base + attack 둘 다 선택
```

#### 3D 평가 생성 ⭐ 중요!
```
1. 베이스만: base_dataset_3d_id 선택

2. 공격만 (비교 없이):
   - attack_dataset_3d_id 선택
   - base_dataset_3d_id 선택 안 함
   → 공격 데이터셋만 평가, robustness 계산 없음

3. 공격 + 비교:
   - attack_dataset_3d_id 선택
   - base_dataset_3d_id 수동 선택 ⭐ 필수!
   → 원본 vs 공격 비교, robustness 계산 포함
```

---

## 📊 데이터베이스 저장 구조

### 2D Attack Evaluation
```
eval_runs (1개)
  ├─ base_dataset_id: auto-populated
  ├─ attack_dataset_id: user-selected
  └─ metrics_summary: {robustness, original_metrics}

eval_dataset_results (2개)
  ├─ result 1: dataset_type=BASE, dataset_id=base_dataset.id
  └─ result 2: dataset_type=ATTACK, dataset_id=attack_output_dataset.id

eval_items (N개)
  ├─ items for BASE (eval_dataset_result_id → result 1)
  └─ items for ATTACK (eval_dataset_result_id → result 2)

eval_class_metrics (M개)
  ├─ metrics for BASE (eval_dataset_result_id → result 1)
  └─ metrics for ATTACK (eval_dataset_result_id → result 2)
```

### 3D Attack Evaluation (with base)
```
eval_runs (1개)
  ├─ base_dataset_3d_id: user-selected (manual)
  ├─ attack_dataset_3d_id: user-selected
  └─ metrics_summary: {robustness, original_metrics}

eval_dataset_results (2개)
  ├─ result 1: dataset_type=BASE, dataset_id=base_dataset_3d.id
  └─ result 2: dataset_type=ATTACK, dataset_id=attack_output_dataset.id

...same as 2D...
```

### 3D Attack Evaluation (without base) ⭐ NEW!
```
eval_runs (1개)
  ├─ base_dataset_3d_id: null
  ├─ attack_dataset_3d_id: user-selected
  └─ metrics_summary: attack metrics only (no robustness)

eval_dataset_results (1개만!)
  └─ result 1: dataset_type=ATTACK, dataset_id=attack_output_dataset.id

eval_items (N개)
  └─ items for ATTACK (eval_dataset_result_id → result 1)

eval_class_metrics (M개)
  └─ metrics for ATTACK (eval_dataset_result_id → result 1)
```

---

## ⚠️ 주의사항

### 1. 3D 공격 데이터셋 GT

**3D 공격 데이터셋의 GT는 어디에 있나?**
- `output_dataset_id`가 가리키는 데이터셋의 `annotations` 테이블에 저장됨
- CARLA 시뮬레이션 등으로 생성 시 GT도 함께 생성됨

### 2. Filename Mapping (3D with base)

**원본 데이터셋과 비교 시**:
- 공격 데이터셋의 이미지 파일명과 원본 데이터셋의 이미지 파일명이 매칭되어야 함
- 예: `original_001.png` → `attacked_001.png`
- 매칭 실패 시 평가 오류 발생

### 3. UI 제약 사항

**프론트엔드에서 체크 필요**:
```typescript
// 3D 공격 데이터셋 선택 시
if (dimension === '3d' && selectedAttackDataset) {
  // 옵션 1: base 선택 강제 (비교 평가)
  if (!selectedBaseDataset) {
    showWarning("3D 공격 데이터셋 평가 시 원본 데이터셋을 선택하면 robustness 비교가 가능합니다.");
  }

  // 옵션 2: base 선택 optional (사용자 선택)
  if (selectedBaseDataset) {
    info = "원본 데이터셋과 비교하여 robustness를 계산합니다.";
  } else {
    info = "공격 데이터셋만 평가합니다 (robustness 계산 없음).";
  }
}
```

---

## 🔄 마이그레이션 가이드

### 기존 시스템에서 변경 사항

**Before** (문제):
```python
# 3D 평가 시 base_dataset_3d_id가 필수였음
if dimension == "3d" and not eval_run.base_dataset_3d_id:
    raise ValidationError("3D 평가에는 base_dataset_3d_id가 필요합니다")
```

**After** (수정):
```python
# 3D 평가 시 base_dataset_3d_id는 optional
original_dataset_id = eval_run.base_dataset_3d_id  # May be None

if original_dataset_id:
    # Compare evaluation (with robustness)
else:
    # Attack-only evaluation (without robustness)
```

---

## 📝 요약

| 항목 | 2D | 3D |
|------|----|----|
| **base_dataset_id 필드** | ✅ 있음 | ❌ 없음 |
| **Auto-populate** | ✅ 가능 | ❌ 불가능 |
| **공격만 선택 시** | base 자동 채워짐 | base 수동 선택 필요 (optional) |
| **Robustness 계산** | 항상 가능 | base 제공 시만 가능 |
| **GT 소스 (비교 시)** | 원본 dataset | 원본 dataset (filename mapping) |
| **GT 소스 (단독 시)** | N/A | output_dataset 자체 |

---

**결론**:
- 2D는 base_dataset_id가 자동으로 채워지므로 항상 비교 평가
- 3D는 base_dataset_3d_id를 수동으로 선택해야 하며, 선택하지 않으면 공격 데이터셋만 평가됨
