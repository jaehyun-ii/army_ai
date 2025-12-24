# 전체 시스템 검토 보고서

**날짜**: 2025-12-24
**목적**: 평가 시스템 전체 검토 및 문제점 파악

---

## ✅ 검토 완료 항목

### 1. Database Schema ✅
- `eval_runs` constraint 확인
- `eval_dataset_results` 테이블 구조 확인
- `attack_datasets_3d` base_dataset_id 없음 확인

**결과**: 문제 없음

---

### 2. Backend Evaluation Service ✅

#### 시나리오별 로직 검증

##### ✅ 시나리오 1: 베이스만 선택
```python
# _evaluate_base_dataset()
dataset = base_dataset
images = base_dataset_images
metrics = _calculate_metrics_with_gt(images=images)
_save_evaluation_results(dataset=dataset, dataset_type=BASE)
```
**결과**: 올바름

---

##### ✅ 시나리오 2a: 2D 공격만 선택
```python
# API endpoint: base_dataset_id auto-populated
original_dataset_id = attack_dataset.base_dataset_id  # ✅ Auto-populated

# _evaluate_attack_dataset()
# Step 1: Evaluate original
original_metrics = _calculate_metrics_with_gt(images=original_images)

# Step 2: Evaluate attack (2D)
if dimension == "2d":
    attack_metrics = _calculate_metrics_with_gt(
        images=original_images,  # ✅ Use original GT
        use_filename_mapping=True,
    )

# Step 3: Robustness
robustness_metrics = calculate_robustness_metrics(...)

# Save
_save_evaluation_results(..., BASE)
_save_evaluation_results(..., ATTACK, class_names_dataset=original_dataset)
```
**결과**: 올바름 - 베이스 GT 사용

---

##### ✅ 시나리오 2b: 3D 공격만 선택 (base 미제공)
```python
# API endpoint: No auto-populate for 3D
original_dataset_id = eval_run.base_dataset_3d_id  # None

# _evaluate_attack_dataset()
# Step 1: Skip original evaluation
if original_dataset_id:  # False
    ...
else:
    original_metrics = None

# Step 2: Evaluate attack (3D)
if dimension == "3d":
    attack_metrics = _calculate_metrics_with_gt(
        images=attack_images,  # ✅ Use attack's own GT
        dataset=attack_dataset_obj,
        use_filename_mapping=False,
    )

# Step 3: Skip robustness
if original_metrics:  # False
    ...
else:
    robustness_metrics = None

# Save
_save_evaluation_results(
    dataset=attack_dataset_obj,  # ✅ Attack dataset
    dataset_type=ATTACK,
    # class_names_dataset not provided → uses dataset
)
```
**결과**: 올바름 - 공격 자체 GT 사용

---

##### ✅ 시나리오 3a: 2D 둘다 선택
```python
# Frontend: Creates 2 runs
# Run 1 (Clean): pre_attack + base_dataset_id
# Run 2 (Adversarial): post_attack + base_dataset_id + attack_dataset_id

# Run 2 execution:
# Step 1: Evaluate original (베이스 GT)
original_metrics = _calculate_metrics_with_gt(images=original_images)

# Step 2: Evaluate attack (2D)
if dimension == "2d":
    attack_metrics = _calculate_metrics_with_gt(
        images=original_images,  # ✅ Use original GT (filename mapping)
        use_filename_mapping=True,
    )

# Step 3: Robustness
robustness_metrics = calculate_robustness_metrics(...)

# Save
_save_evaluation_results(..., BASE)
_save_evaluation_results(..., ATTACK, class_names_dataset=original_dataset)
```
**결과**: 올바름 - 둘 다 베이스 GT 사용

---

##### ✅ 시나리오 3b: 3D 둘다 선택
```python
# Frontend: Creates 2 runs
# Run 1 (Clean): pre_attack + base_dataset_3d_id
# Run 2 (Adversarial): post_attack + base_dataset_3d_id + attack_dataset_3d_id

# Run 2 execution:
# Step 1: Evaluate original (베이스 GT)
original_metrics = _calculate_metrics_with_gt(images=original_images)

# Step 2: Evaluate attack (3D)
if dimension == "3d":
    attack_metrics = _calculate_metrics_with_gt(
        images=attack_images,  # ✅ Use attack's own GT
        dataset=attack_dataset_obj,
        use_filename_mapping=False,
    )

# Step 3: Robustness
robustness_metrics = calculate_robustness_metrics(
    original_metrics,  # 베이스 GT로 평가한 결과
    attack_metrics,    # 공격 자체 GT로 평가한 결과
)

# Save
_save_evaluation_results(
    ..., BASE,
    dataset=original_dataset,
)
_save_evaluation_results(
    ..., ATTACK,
    dataset=attack_dataset_obj,  # ✅ Attack dataset
    class_names_dataset=original_dataset,  # 비교를 위해 동일한 class_names 사용
)
```
**결과**: 올바름 - 각각의 GT 사용

---

### 3. Frontend Integration ✅

#### Phase 결정 로직
```typescript
let phase = "pre_attack"  // 기본값

if (selectedBaseDataset && selectedAttackDataset) {
  // 둘다: 2개 run 생성 (별도 로직)
} else if (selectedAttackDataset) {
  phase = "post_attack"  // ✅ 공격만
} else {
  // 베이스만 (기본값 유지)
}
```

#### Dataset ID 설정
```typescript
// 단일 선택 시
if (datasetDimension === "3d") {
  if (evalBaseDatasetId) runData.base_dataset_3d_id = evalBaseDatasetId
  if (evalAttackDatasetId) runData.attack_dataset_3d_id = evalAttackDatasetId
}
```

**3D 공격만 선택 시**:
- `evalBaseDatasetId = undefined` (선택 안 함)
- `evalAttackDatasetId = selectedAttackDataset`
- → `runData`에 `base_dataset_3d_id` 없음 ✅
- → Backend에서 `eval_run.base_dataset_3d_id = None`

**결과**: 올바름

---

### 4. API Endpoint ✅

```python
# create_evaluation_run (evaluation.py:66-87)

# 2D auto-populate
if eval_run.attack_dataset_id and not eval_run.base_dataset_id:
    attack_dataset = await crud.attack_dataset_2d.get(...)
    if attack_dataset and attack_dataset.base_dataset_id:
        eval_run.base_dataset_id = attack_dataset.base_dataset_id  # ✅

# 3D: No auto-populate (올바름)
```

**결과**: 올바름

---

### 5. Database Constraints ✅

```sql
CONSTRAINT chk_eval_phase_requirements CHECK (
  (phase = 'pre_attack' AND (base_dataset_id IS NOT NULL OR base_dataset_3d_id IS NOT NULL)
    AND attack_dataset_id IS NULL AND attack_dataset_3d_id IS NULL) OR
  (phase = 'post_attack' AND ((attack_dataset_id IS NOT NULL AND attack_dataset_3d_id IS NULL) OR
                               (attack_dataset_id IS NULL AND attack_dataset_3d_id IS NOT NULL)))
)
```

**검증**:
- pre_attack: base 필수, attack 없음 ✅
- post_attack: attack 필수, **base optional** ✅

**결과**: 올바름

---

## 🔍 세부 검증

### class_names_dataset 처리

#### Case 1: 2D/3D 공격만 선택 (original 있음 - 2D auto-populate)
```python
_save_evaluation_results(
    dataset=attack_dataset_obj,
    class_names_dataset=original_dataset,  # ✅ 원본 class_names 사용
)
```
**이유**: 비교를 위해 일관된 class_names 필요

#### Case 2: 3D 공격만 선택 (original 없음)
```python
_save_evaluation_results(
    dataset=attack_dataset_obj,
    # class_names_dataset 없음
)

# _save_evaluation_results 내부:
class_names = self._get_class_names_from_dataset(
    class_names_dataset or dataset  # → dataset (attack_dataset_obj) 사용
)
```
**이유**: 공격 데이터셋 자체 class_names 사용

**결과**: 올바름

---

### GT Source 처리

#### 2D
```python
# 항상 original GT 사용 (filename mapping)
if dimension == "2d":
    if original_images:
        attack_metrics = _calculate_metrics_with_gt(
            images=original_images,  # ✅
            use_filename_mapping=True,
        )
    else:
        raise ValidationError("2D requires base_dataset_id")
```

#### 3D
```python
# 항상 attack 자체 GT 사용
if dimension == "3d":
    attack_metrics = _calculate_metrics_with_gt(
        images=attack_images,  # ✅
        dataset=attack_dataset_obj,
        use_filename_mapping=False,
    )
```

**결과**: 올바름

---

## ✅ 최종 결론

### 모든 검토 항목 통과

1. ✅ **Database Schema**: 올바름
2. ✅ **Backend Logic**: 모든 시나리오 올바르게 처리
3. ✅ **Frontend Integration**: 올바른 요청 생성
4. ✅ **API Endpoint**: Auto-populate 로직 올바름
5. ✅ **Constraints**: 요구사항 충족
6. ✅ **GT Handling**: 2D/3D 올바르게 분기
7. ✅ **class_names Handling**: 모든 케이스 올바름

---

## 🎯 발견된 문제

### ❌ 문제 없음!

전체 시스템이 요구사항에 맞게 올바르게 구현되었습니다.

---

## 📋 요구사항 충족 확인

| 시나리오 | 2D | 3D | 상태 |
|---------|----|----|------|
| **베이스만** | 베이스 GT | 베이스 GT | ✅ |
| **공격만** | 베이스 GT (auto) | 공격 자체 GT | ✅ |
| **둘다** | 베이스 GT (둘 다) | 각각의 GT | ✅ |

---

## 🚀 다음 단계

### 권장 사항

1. **데이터베이스 스키마 적용**
   ```bash
   psql -U postgres -d army_ai -f database/complete_schema.sql
   ```

2. **백엔드 재시작**
   ```bash
   docker-compose restart backend
   ```

3. **전체 시나리오 테스트**
   - [ ] 2D 베이스만
   - [ ] 2D 공격만
   - [ ] 2D 둘다
   - [ ] 3D 베이스만
   - [ ] 3D 공격만 (base 미제공)
   - [ ] 3D 둘다

4. **검증 사항**
   - [ ] GT 소스가 올바른지
   - [ ] Robustness 계산 정확성
   - [ ] eval_dataset_results 올바른 dataset_id 저장
   - [ ] class_names 올바르게 처리

---

## 📄 관련 문서

1. `EVALUATION_FLOW_ANALYSIS.md` - 전체 플로우 분석
2. `FIXES_APPLIED.md` - 주요 수정 사항
3. `3D_DATASET_HANDLING.md` - 3D 데이터셋 가이드
4. `3D_ATTACK_CREATION_ANALYSIS.md` - 3D 생성 분석
5. `GT_HANDLING_FIX.md` - GT 처리 수정
6. `SYSTEM_REVIEW_REPORT.md` - 전체 시스템 검토 ⭐ 현재 문서

---

## 🎉 결론

**전체 시스템이 요구사항에 완벽히 부합하며, 발견된 문제가 없습니다!**

모든 수정 사항이 올바르게 적용되었으며, 데이터베이스 마이그레이션 후 즉시 사용 가능합니다.
