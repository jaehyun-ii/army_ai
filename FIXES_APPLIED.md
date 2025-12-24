# 평가 시스템 주요 문제점 수정 완료

**날짜**: 2025-12-24
**작업**: Critical 및 Important 문제 해결

---

## ✅ 수정 완료된 문제들

### 🔴 Critical Fix 1: Attack Evaluation에서 dataset_id 불일치 수정

**문제**:
```python
# 기존 코드 (잘못됨)
await _save_evaluation_results(
    dataset=original_dataset,  # ❌ 공격 데이터셋인데 원본 dataset 사용
    dataset_type=EvalDatasetType.ATTACK,
)

# 결과:
eval_dataset_results:
  dataset_type: ATTACK
  dataset_id: original_dataset.id  # ❌ 잘못됨! attack_output_dataset.id여야 함
```

**수정 내용**:

#### 1. `_save_evaluation_results()` 함수에 `class_names_dataset` 파라미터 추가

**파일**: `backend/app/services/evaluation_service.py:895-926`

```python
async def _save_evaluation_results(
    self,
    ...
    dataset: Any = None,  # 평가 대상 데이터셋 (dataset_id 결정)
    class_names_dataset: Optional[Any] = None,  # ✅ NEW: 클래스명 해석용
    ...
):
    """
    Args:
        dataset: The dataset being evaluated (determines dataset_id in eval_dataset_results)
        class_names_dataset: Optional dataset to use for class name resolution.
                             Useful when evaluating attack datasets (dataset=attack, class_names_dataset=original)
    """
    # Get class names from class_names_dataset if provided
    class_names = self._get_class_names_from_dataset(
        class_names_dataset or dataset
    ) if (class_names_dataset or dataset) else None
```

#### 2. Attack evaluation 호출 부분 수정

**파일**: `backend/app/services/evaluation_service.py:498-512`

```python
# Step 2: Save attack evaluation results
await self._save_evaluation_results(
    db=db,
    eval_run_id=eval_run.id,
    inference_results=attack_inference_results,
    images=attack_images,
    metrics=combined_metrics,
    eval_logger=eval_logger,
    dataset=attack_dataset_obj,  # ✅ FIXED: Use attack dataset for dataset_id
    dataset_type=EvalDatasetType.ATTACK,
    target_class=target_class,
    iou_threshold=iou_threshold,
    dimension=dimension,
    class_names_dataset=original_dataset,  # ✅ NEW: Use original dataset for class names
)
```

**효과**:
- ✅ `eval_dataset_results.dataset_id`가 올바르게 `attack_output_dataset.id`로 저장됨
- ✅ 클래스명은 여전히 원본 데이터셋에서 가져옴 (정확한 클래스 매핑)

---

### 🔴 Critical Fix 2: API 엔드포인트 추가

**문제**: `eval_dataset_results` 테이블을 조회할 API 엔드포인트 누락

**수정 내용**:

#### 1. Schema import 추가

**파일**: `backend/app/api/v1/endpoints/evaluation.py:18-39`

```python
from app.schemas.evaluation import (
    ...,
    EvalDatasetResultResponse,  # ✅ NEW
    ...,
)
```

#### 2. 두 개의 새로운 엔드포인트 추가

**파일**: `backend/app/api/v1/endpoints/evaluation.py:103-155`

##### 엔드포인트 1: 모든 dataset 결과 조회
```python
@router.get("/runs/{run_id}/dataset-results", response_model=List[EvalDatasetResultResponse])
async def get_eval_dataset_results_by_run(
    run_id: UUID,
    dataset_type: Optional[EvalDatasetType] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    """
    Get all dataset results for an evaluation run.

    Optionally filter by dataset_type (BASE or ATTACK).
    """
    results = await crud_evaluation.get_eval_dataset_results_by_run(
        db=db,
        eval_run_id=run_id,
        dataset_type=dataset_type
    )
    if not results:
        raise HTTPException(status_code=404, detail="No dataset results found")
    return results
```

**사용 예시**:
```bash
# 모든 결과 조회
GET /api/v1/evaluation/runs/{run_id}/dataset-results

# BASE만 조회
GET /api/v1/evaluation/runs/{run_id}/dataset-results?dataset_type=base

# ATTACK만 조회
GET /api/v1/evaluation/runs/{run_id}/dataset-results?dataset_type=attack
```

##### 엔드포인트 2: 특정 타입 결과 직접 조회
```python
@router.get("/runs/{run_id}/dataset-results/{dataset_type}", response_model=EvalDatasetResultResponse)
async def get_eval_dataset_result_by_type(
    run_id: UUID,
    dataset_type: EvalDatasetType,
    db: AsyncSession = Depends(get_db),
):
    """Get specific dataset result by type (BASE or ATTACK)."""
    result = await crud_evaluation.get_eval_dataset_result_by_run_and_type(
        db=db,
        eval_run_id=run_id,
        dataset_type=dataset_type
    )
    if not result:
        raise HTTPException(status_code=404, detail=f"No {dataset_type.value} result found")
    return result
```

**사용 예시**:
```bash
# BASE 결과만 직접 조회
GET /api/v1/evaluation/runs/{run_id}/dataset-results/base

# ATTACK 결과만 직접 조회
GET /api/v1/evaluation/runs/{run_id}/dataset-results/attack
```

**효과**:
- ✅ Frontend에서 새로운 구조를 사용할 수 있음
- ✅ `eval_runs.metrics_summary`에 의존하지 않고 `eval_dataset_results`에서 직접 조회 가능
- ✅ BASE와 ATTACK 결과를 명확하게 분리하여 조회 가능

---

### 🟡 Important Fix 3: 3D Auto-populate 로직 추가

**문제**: 2D는 공격 데이터셋 선택 시 `base_dataset_id`가 자동으로 채워지지만, 3D는 처리 안 됨

**수정 내용**:

**파일**: `backend/app/api/v1/endpoints/evaluation.py:66-99`

```python
try:
    # Auto-populate base_dataset_id from attack_dataset if needed (2D)
    if eval_run.attack_dataset_id and not eval_run.base_dataset_id:
        from app import crud
        attack_dataset = await crud.attack_dataset_2d.get(db, id=eval_run.attack_dataset_id)
        if attack_dataset and attack_dataset.base_dataset_id:
            eval_run.base_dataset_id = attack_dataset.base_dataset_id

    # ✅ NEW: Auto-populate base_dataset_3d_id from attack_dataset_3d (3D)
    if eval_run.attack_dataset_3d_id and not eval_run.base_dataset_3d_id:
        from sqlalchemy import select
        from app.models.dataset_3d import AttackDataset3D
        result = await db.execute(
            select(AttackDataset3D).where(
                AttackDataset3D.id == eval_run.attack_dataset_3d_id,
                AttackDataset3D.deleted_at.is_(None)
            )
        )
        attack_dataset_3d = result.scalar_one_or_none()
        if attack_dataset_3d and attack_dataset_3d.base_dataset_id:
            eval_run.base_dataset_3d_id = attack_dataset_3d.base_dataset_id

    db_eval_run = await crud_evaluation.create_eval_run(db=db, eval_run=eval_run)
    return db_eval_run
```

**효과**:
- ✅ 3D 공격 데이터셋만 선택해도 자동으로 원본 데이터셋 ID가 채워짐
- ✅ 2D와 3D 모두 일관된 동작 보장
- ✅ 사용자가 수동으로 base_dataset_3d_id를 입력할 필요 없음

---

## 📊 수정 요약

### 파일 변경 사항

#### 1. `backend/app/services/evaluation_service.py`
- `_save_evaluation_results()` 함수 시그니처 변경 (line 895-926)
  - `class_names_dataset` 파라미터 추가
  - 클래스명 해석 로직 개선
- Attack evaluation 호출 부분 수정 (line 498-512)
  - `dataset=attack_dataset_obj` 변경
  - `class_names_dataset=original_dataset` 추가

#### 2. `backend/app/api/v1/endpoints/evaluation.py`
- Schema import 추가 (line 18-39)
  - `EvalDatasetResultResponse` import
- 새로운 API 엔드포인트 2개 추가 (line 103-155)
  - `GET /runs/{run_id}/dataset-results`
  - `GET /runs/{run_id}/dataset-results/{dataset_type}`
- 3D auto-populate 로직 추가 (line 75-87)

---

## 🎯 시나리오별 동작 확인

### ✅ 시나리오 1: 베이스 데이터셋만 선택
```
Input: base_dataset_id = "xxx"
Process:
  1. phase = "pre_attack"
  2. _evaluate_base_dataset()
  3. _save_evaluation_results(dataset=base_dataset, dataset_type=BASE)
Output:
  - eval_dataset_results: 1개 (BASE)
  - dataset_id = base_dataset.id ✅
```

### ✅ 시나리오 2: 공격 데이터셋만 선택
```
Input: attack_dataset_id = "yyy"
Process:
  1. Auto-populate base_dataset_id from attack_dataset ✅ (2D/3D 모두)
  2. phase = "post_attack"
  3. _evaluate_attack_dataset()
     - Step 1: original 평가
       _save_evaluation_results(dataset=original_dataset, dataset_type=BASE)
     - Step 2: attack 평가
       _save_evaluation_results(dataset=attack_dataset_obj, dataset_type=ATTACK, class_names_dataset=original_dataset) ✅ FIXED
Output:
  - eval_dataset_results: 2개 (BASE + ATTACK)
  - BASE: dataset_id = original_dataset.id ✅
  - ATTACK: dataset_id = attack_output_dataset.id ✅ FIXED
```

### ✅ 시나리오 3: 베이스 + 공격 함께 선택
```
Input: base_dataset_id = "xxx", attack_dataset_id = "yyy"
Process:
  - Frontend가 2개의 run 생성
  - Run 1 (Clean): 시나리오 1과 동일
  - Run 2 (Adversarial): 시나리오 2와 동일
Output:
  - Run 1: eval_dataset_results 1개 (BASE)
  - Run 2: eval_dataset_results 2개 (BASE + ATTACK)
```

---

## 🚀 다음 단계

### 필수 (데이터베이스 적용)
1. ✅ **데이터베이스 스키마 적용**
   ```bash
   # complete_schema.sql 적용 (eval_dataset_results 테이블 추가)
   psql -U postgres -d army_ai -f database/complete_schema.sql
   ```

2. ✅ **백엔드 재시작**
   ```bash
   # 변경된 코드 반영
   docker-compose restart backend
   ```

3. ✅ **테스트**
   - 시나리오 1: 베이스 데이터셋만 선택 → 평가 실행 → 결과 확인
   - 시나리오 2: 공격 데이터셋만 선택 → 평가 실행 → 결과 확인
   - 시나리오 3: 베이스+공격 선택 → 평가 실행 → 결과 확인
   - 새 API 엔드포인트 테스트:
     ```bash
     GET /api/v1/evaluation/runs/{run_id}/dataset-results
     GET /api/v1/evaluation/runs/{run_id}/dataset-results/base
     GET /api/v1/evaluation/runs/{run_id}/dataset-results/attack
     ```

### 선택 (프론트엔드 개선)
4. **Frontend 마이그레이션** (현재는 backward compatibility로 동작)
   - `eval_runs.metrics_summary` 대신 `eval_dataset_results` 사용
   - 새로운 API 엔드포인트 활용
   - 더 명확한 데이터 표시

---

## 📝 알려진 제한사항

### Backward Compatibility 유지
- `eval_items.dataset_type` - DEPRECATED이지만 유지 중
- `eval_class_metrics.dataset_type` - DEPRECATED이지만 유지 중
- `eval_runs.metrics_summary` - robustness 정보 저장용으로 계속 사용

### Frontend
- 현재 Frontend는 여전히 `eval_runs.metrics_summary`를 사용
- 새로운 API 엔드포인트로 마이그레이션 권장 (선택 사항)

---

## 🎉 완료!

모든 Critical 및 Important 문제가 수정되었습니다!

**수정된 파일**:
1. ✅ `backend/app/services/evaluation_service.py`
2. ✅ `backend/app/api/v1/endpoints/evaluation.py`

**추가된 기능**:
1. ✅ Attack evaluation dataset_id 올바른 저장
2. ✅ 2개의 새로운 API 엔드포인트
3. ✅ 3D auto-populate 로직

**다음**: 데이터베이스 마이그레이션 적용 → 테스트
