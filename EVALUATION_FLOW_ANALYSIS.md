# 평가 시스템 플로우 분석 및 스키마 검토

**날짜**: 2025-12-24
**목적**: 평가 생성부터 조회까지의 전체 플로우 분석 및 스키마 일관성 검증

---

## 📋 목차
1. [시스템 개요](#시스템-개요)
2. [데이터베이스 스키마](#데이터베이스-스키마)
3. [평가 생성 플로우](#평가-생성-플로우)
4. [데이터 저장 플로우](#데이터-저장-플로우)
5. [데이터 조회 플로우](#데이터-조회-플로우)
6. [발견된 Mismatch](#발견된-mismatch)
7. [권장 사항](#권장-사항)

---

## 시스템 개요

### 평가 시스템의 3가지 시나리오

#### ✅ 시나리오 1: **베이스 데이터셋만 선택**
```
사용자 → 베이스 데이터셋 선택 → 평가 실행
결과: 1개 eval_run + 1개 eval_dataset_result (BASE)
```

#### ✅ 시나리오 2: **공격 데이터셋만 선택**
```
사용자 → 공격 데이터셋 선택 → 평가 실행
결과: 1개 eval_run + 2개 eval_dataset_results (BASE + ATTACK)
주의: post_attack phase이므로 base_dataset_id가 필수였으나, 스키마 수정으로 optional로 변경
```

#### ✅ 시나리오 3: **베이스 + 공격 데이터셋 함께 선택**
```
사용자 → 베이스 + 공격 선택 → 2개 평가 생성
결과:
  - Run 1 (Clean): 1개 eval_run + 1개 eval_dataset_result (BASE)
  - Run 2 (Adversarial): 1개 eval_run + 2개 eval_dataset_results (BASE + ATTACK)
```

---

## 데이터베이스 스키마

### Phase 3 스키마 구조 (NEW)

```sql
eval_runs (평가 실행 이력)
  ├─ id, name, description, phase, dimension
  ├─ model_id, base_dataset_id, attack_dataset_id
  ├─ base_dataset_3d_id, attack_dataset_3d_id
  ├─ params, status
  ├─ metrics_summary (DEPRECATED - for robustness only)
  ├─ iou_distribution (DEPRECATED)
  └─ Constraint:
      - pre_attack: base_dataset_id 필수, attack_dataset_id NULL
      - post_attack: attack_dataset_id 필수, base_dataset_id OPTIONAL ✅ 수정됨

eval_dataset_results (데이터셋별 상세 결과) ⭐ NEW
  ├─ id, eval_run_id
  ├─ dataset_type (BASE | ATTACK)
  ├─ dataset_id (실제 데이터셋 UUID)
  ├─ dataset_dimension (2d | 3d)
  ├─ metrics_summary (이 데이터셋의 메트릭)
  └─ iou_distribution

eval_items (이미지별 결과)
  ├─ id, run_id (DEPRECATED)
  ├─ eval_dataset_result_id ⭐ NEW
  ├─ image_2d_id, image_3d_id
  ├─ dataset_type (DEPRECATED)
  └─ ground_truth, prediction, metrics

eval_class_metrics (클래스별 메트릭)
  ├─ id, run_id (DEPRECATED)
  ├─ eval_dataset_result_id ⭐ NEW
  ├─ class_name
  ├─ dataset_type (DEPRECATED)
  └─ metrics, iou_distribution
```

---

## 평가 생성 플로우

### Frontend → Backend API

#### 📍 위치
- **Frontend**: `frontend/components/evaluation/unified-evaluation-dashboard.tsx`
- **API Endpoint**: `POST /api/v1/evaluation/runs`
- **Handler**: `backend/app/api/v1/endpoints/evaluation.py:create_evaluation_run()`

#### 시나리오별 데이터 구조

##### 1️⃣ **베이스 데이터셋만 선택**
```typescript
// Frontend (line 558-583)
const runData = {
  name: evaluationName,
  phase: "pre_attack",
  dimension: datasetDimension,  // "2d" | "3d"
  model_id: selectedModel,
  base_dataset_id: selectedBaseDataset,  // 2D일 경우
  base_dataset_3d_id: selectedBaseDataset,  // 3D일 경우
  attack_dataset_id: null,
  params: { target_class: targetClass }
}
```

**Backend 처리**:
```python
# evaluation.py:53-84
db_eval_run = await crud_evaluation.create_eval_run(db=db, eval_run=eval_run)
# Constraint 검증: phase='pre_attack' → base_dataset_id 필수, attack_dataset_id NULL
```

##### 2️⃣ **공격 데이터셋만 선택** (⚠️ 문제 있음!)
```typescript
// Frontend (line 558-583)
const runData = {
  name: evaluationName,
  phase: "post_attack",  // ⚠️ 공격만 선택 시 post_attack
  dimension: datasetDimension,
  model_id: selectedModel,
  base_dataset_id: null,  // ⚠️ NULL!
  attack_dataset_id: selectedAttackDataset,
  params: { target_class: targetClass }
}
```

**Backend 처리** (⚠️ 이전 constraint 위반!):
```python
# evaluation.py:66-72
# AUTO-POPULATE: attack_dataset에서 base_dataset_id 추출 시도
if eval_run.attack_dataset_id and not eval_run.base_dataset_id:
    attack_dataset = await crud.attack_dataset_2d.get(db, id=eval_run.attack_dataset_id)
    if attack_dataset and attack_dataset.base_dataset_id:
        eval_run.base_dataset_id = attack_dataset.base_dataset_id  # ✅ 자동 설정
```

**Constraint 검증**:
```sql
-- 이전 (문제)
CONSTRAINT chk_eval_phase_requirements CHECK (
  (phase = 'post_attack' AND base_dataset_id IS NOT NULL ...)  -- ❌ 실패!
)

-- 현재 (수정됨)
CONSTRAINT chk_eval_phase_requirements CHECK (
  (phase = 'post_attack' AND attack_dataset_id IS NOT NULL)  -- ✅ base는 optional
)
```

##### 3️⃣ **베이스 + 공격 함께 선택**
```typescript
// Frontend (line 504-556) - 2개의 run 생성
// Run 1: Clean
const cleanRunData = {
  name: `${evaluationName} (Clean)`,
  phase: "pre_attack",
  model_id: selectedModel,
  base_dataset_id: selectedBaseDataset,  // 2D
  base_dataset_3d_id: selectedBaseDataset,  // 3D
  params: { target_class: targetClass }
}

// Run 2: Adversarial
const advRunData = {
  name: `${evaluationName} (Adversarial)`,
  phase: "post_attack",
  model_id: selectedModel,
  base_dataset_id: selectedBaseDataset,  // ✅ 둘 다 설정
  base_dataset_3d_id: selectedBaseDataset,
  attack_dataset_id: selectedAttackDataset,  // ✅ 둘 다 설정
  attack_dataset_3d_id: selectedAttackDataset,
  params: { target_class: targetClass }
}
```

---

## 데이터 저장 플로우

### Execution Flow

#### 📍 위치
- **API Endpoint**: `POST /api/v1/evaluation/runs/{run_id}/execute`
- **Service**: `backend/app/services/evaluation_service.py:execute_evaluation()`

#### 분기 로직
```python
# evaluation_service.py:92-189
async def execute_evaluation(eval_run_id, conf_threshold, iou_threshold, target_class):
    eval_run = await crud.evaluation.get_eval_run(db, eval_run_id)

    if eval_run.attack_dataset_id or eval_run.attack_dataset_3d_id:
        # Mode 2: Attack dataset evaluation
        await _evaluate_attack_dataset(...)  # 🔴 공격 데이터셋 평가
    else:
        # Mode 1: Base dataset evaluation
        await _evaluate_base_dataset(...)    # 🔵 베이스 데이터셋 평가
```

### Mode 1: Base Dataset Evaluation

**플로우**:
```
1. Load base dataset
2. Run inference on base dataset
3. Calculate metrics (compare predictions with GT)
4. Save results → _save_evaluation_results()
```

**저장되는 데이터**:
```python
# evaluation_service.py:274-286
await _save_evaluation_results(
    eval_run_id=eval_run.id,
    inference_results=[...],
    images=[...],
    metrics={'overall': {...}, 'per_class': {...}},
    dataset=base_dataset,
    dataset_type=EvalDatasetType.BASE,  # ✅ BASE로 표시
    target_class=target_class,
    dimension=dimension
)
```

**결과**:
```
eval_dataset_results 1개 생성:
  - dataset_type: BASE
  - dataset_id: base_dataset.id
  - metrics_summary: {'map50': 0.85, 'precision': 0.88, ...}

eval_items N개 생성 (각 이미지별):
  - eval_dataset_result_id: <위 result의 id>
  - image_2d_id: <image id>
  - dataset_type: BASE (deprecated)

eval_class_metrics M개 생성 (각 클래스별):
  - eval_dataset_result_id: <위 result의 id>
  - class_name: "person", "car", ...
  - dataset_type: BASE (deprecated)
```

### Mode 2: Attack Dataset Evaluation

**플로우**:
```
Step 1: Evaluate Original Dataset
  1.1. Load original dataset (from attack_dataset.base_dataset_id)
  1.2. Run inference on original
  1.3. Calculate original metrics
  1.4. Save original results → _save_evaluation_results(dataset_type=BASE)

Step 2: Evaluate Attack Dataset
  2.1. Load attack output dataset (from attack_dataset.output_dataset_id)
  2.2. Run inference on attack dataset
  2.3. Calculate attack metrics (using ORIGINAL GT!)
  2.4. Calculate robustness metrics (compare original vs attack)

Step 3: Save Attack Results
  3.1. Save attack results → _save_evaluation_results(dataset_type=ATTACK)
       - metrics에 robustness 정보 포함
```

**⭐ 중요: 2번 save_evaluation_results 호출!**

```python
# evaluation_service.py:484-511

# Step 1: Save BASE results
await _save_evaluation_results(
    eval_run_id=eval_run.id,
    inference_results=original_inference_results,
    images=original_images,
    metrics=original_metrics,  # {'overall': {...}, 'per_class': {...}}
    dataset=original_dataset,
    dataset_type=EvalDatasetType.BASE,  # ✅ BASE
    ...
)

# Step 2: Save ATTACK results
await _save_evaluation_results(
    eval_run_id=eval_run.id,
    inference_results=attack_inference_results,
    images=attack_images,
    metrics=combined_metrics,  # {'overall': {...}, 'robustness': {...}, 'original_metrics': {...}}
    dataset=original_dataset,  # ⚠️ original dataset 사용 (for class names)
    dataset_type=EvalDatasetType.ATTACK,  # ✅ ATTACK
    ...
)
```

**결과**:
```
eval_dataset_results 2개 생성:
  1. BASE result:
     - dataset_type: BASE
     - dataset_id: original_dataset.id
     - metrics_summary: {'map50': 0.85, ...}

  2. ATTACK result:
     - dataset_type: ATTACK
     - dataset_id: attack_output_dataset.id  ⚠️ 확인 필요!
     - metrics_summary: {'map50': 0.65, ...}

eval_runs 업데이트:
  - metrics_summary: {
      'robustness': {...},
      'original_metrics': {...}
    }
```

### _save_evaluation_results() 상세

**📍 위치**: `backend/app/services/evaluation_service.py:895-1113`

**새로운 로직 (Phase 3)**:
```python
async def _save_evaluation_results(...):
    # STEP 1: Create eval_dataset_result ⭐ NEW
    dataset_result = await crud.evaluation.create_eval_dataset_result(db,
        EvalDatasetResultCreate(
            eval_run_id=eval_run_id,
            dataset_type=dataset_type,  # BASE or ATTACK
            dataset_id=dataset.id,
            dataset_dimension=dimension_enum,
            metrics_summary=dataset_metrics_summary,
            iou_distribution=dataset_iou_distribution
        )
    )
    eval_dataset_result_id = dataset_result.id

    # STEP 2: Create eval_items (linked to dataset_result)
    for result in inference_results:
        eval_item = EvalItemCreate(
            run_id=eval_run_id,  # DEPRECATED
            eval_dataset_result_id=eval_dataset_result_id,  # ⭐ NEW
            image_2d_id=image_id,
            dataset_type=dataset_type,  # DEPRECATED
            ...
        )
        await crud.evaluation.create_eval_item(db, eval_item)

    # STEP 3: Create eval_class_metrics (linked to dataset_result)
    for class_name, class_metrics in per_class_metrics.items():
        class_metrics_create = EvalClassMetricsCreate(
            run_id=eval_run_id,  # DEPRECATED
            eval_dataset_result_id=eval_dataset_result_id,  # ⭐ NEW
            class_name=class_name,
            dataset_type=dataset_type,  # DEPRECATED
            ...
        )
        await crud.evaluation.create_eval_class_metrics(db, class_metrics_create)

    # STEP 4: Update eval_runs (backward compatibility)
    await crud.evaluation.update_eval_run(db, eval_run_id,
        EvalRunUpdate(
            metrics_summary=metrics_to_save,  # robustness info or overall
            iou_distribution=overall_iou_dist
        )
    )
```

---

## 데이터 조회 플로우

### 현재 조회 방식 (Frontend)

#### 📍 위치
- **Frontend**: `frontend/components/evaluation/unified-evaluation-dashboard.tsx:714-781`

```typescript
// handleViewResults()
const results = []
for (const evalId of completedEvaluationIds) {
  // 1. eval_run 조회
  const evalRun = await apiClient.getEvaluationRun(evalId)

  // 2. class_metrics 조회
  const classMetrics = await apiClient.getEvaluationClassMetrics(evalId)

  // 3. 결과 구성
  results.push({
    ...evalRun,
    classMetrics,
    targetClassName
  })
}

// 4. Comparison data 조회 (pre_attack + post_attack인 경우)
const preRun = results.find(r => r.phase === 'pre_attack')
const postRun = results.find(r => r.phase === 'post_attack')

if (preRun && postRun) {
  const comparisonData = await apiClient.compareRobustness({
    clean_run_id: preRun.id,
    adv_run_id: postRun.id
  })
  setComparisonData(comparisonData)
}
```

### ⚠️ Mismatch 발견!

**문제**: Frontend는 여전히 `eval_runs.metrics_summary`를 사용하고 있음

```typescript
// Frontend (line 1152-1160)
const baseAP50 = (preRun.metrics_summary.ap50 || preRun.metrics_summary.map50 || 0) * 100
const attackAP50 = (postRun.metrics_summary.ap50 || postRun.metrics_summary.map50 || 0) * 100
```

**이슈**:
- `eval_runs.metrics_summary`는 DEPRECATED
- 실제 데이터는 `eval_dataset_results.metrics_summary`에 저장됨
- **현재는 backward compatibility를 위해 동작하지만, 장기적으로는 `eval_dataset_results`를 사용해야 함**

---

## 발견된 Mismatch

### 🔴 Critical Mismatch

#### 1. **시나리오 2: 공격 데이터셋만 선택 시 dataset_id 불일치 가능성**

**문제 위치**: `evaluation_service.py:945-956`

```python
# _save_evaluation_results()
dataset_id = dataset.id if dataset else None  # ⚠️ dataset은 무엇인가?
```

**Attack dataset evaluation 시**:
```python
# Line 506: 공격 결과 저장
await _save_evaluation_results(
    dataset=original_dataset,  # ⚠️ original_dataset 전달!
    dataset_type=EvalDatasetType.ATTACK,
    ...
)
```

**예상 결과**:
```sql
eval_dataset_results:
  - dataset_type: ATTACK
  - dataset_id: original_dataset.id  -- ❌ 잘못됨! attack_output_dataset.id여야 함!
```

**올바른 동작이어야 할 것**:
```python
# 공격 결과 저장 시
await _save_evaluation_results(
    dataset=attack_dataset_obj,  # ✅ attack output dataset!
    dataset_type=EvalDatasetType.ATTACK,
    ...
)
```

#### 2. **Frontend 조회 로직이 새 구조를 사용하지 않음**

**현재**:
```typescript
// Frontend는 eval_runs.metrics_summary 직접 사용
const ap50 = evalRun.metrics_summary.ap50
```

**개선 필요**:
```typescript
// eval_dataset_results를 조회하여 사용
const datasetResults = await apiClient.getEvalDatasetResults(evalId)
const baseResult = datasetResults.find(r => r.dataset_type === 'BASE')
const attackResult = datasetResults.find(r => r.dataset_type === 'ATTACK')

const baseAP50 = baseResult.metrics_summary.ap50
const attackAP50 = attackResult.metrics_summary.ap50
```

### 🟡 Medium Mismatch

#### 3. **Deprecated 필드가 여전히 사용됨**

**eval_items**:
```python
# 여전히 run_id, dataset_type 설정
eval_item = EvalItemCreate(
    run_id=eval_run_id,  # DEPRECATED
    eval_dataset_result_id=eval_dataset_result_id,  # NEW
    dataset_type=dataset_type,  # DEPRECATED
)
```

**이슈**:
- Backward compatibility를 위해 유지
- 하지만 장기적으로는 제거 필요

#### 4. **API 엔드포인트 누락**

**필요한 엔드포인트**:
```python
# backend/app/api/v1/endpoints/evaluation.py에 추가 필요
@router.get("/runs/{run_id}/dataset-results", response_model=List[EvalDatasetResultResponse])
async def get_eval_dataset_results(run_id: UUID, db: AsyncSession = Depends(get_db)):
    """Get all dataset results for an evaluation run."""
    return await crud_evaluation.get_eval_dataset_results_by_run(db, run_id)

@router.get("/runs/{run_id}/dataset-results/{dataset_type}", response_model=EvalDatasetResultResponse)
async def get_eval_dataset_result_by_type(
    run_id: UUID,
    dataset_type: EvalDatasetType,
    db: AsyncSession = Depends(get_db)
):
    """Get specific dataset result by type."""
    return await crud_evaluation.get_eval_dataset_result_by_run_and_type(db, run_id, dataset_type)
```

### 🟢 Minor Mismatch

#### 5. **시나리오 2에서 auto-populate 로직**

**현재 API 엔드포인트** (evaluation.py:66-72):
```python
# 공격 데이터셋만 선택 시 자동으로 base_dataset_id 채움
if eval_run.attack_dataset_id and not eval_run.base_dataset_id:
    attack_dataset = await crud.attack_dataset_2d.get(db, id=eval_run.attack_dataset_id)
    if attack_dataset and attack_dataset.base_dataset_id:
        eval_run.base_dataset_id = attack_dataset.base_dataset_id
```

**이슈**:
- ✅ 2D는 동작함
- ❌ 3D는 처리 안 됨 (attack_dataset_3d_id는 체크 안 함)

**수정 필요**:
```python
# 3D도 처리
if eval_run.attack_dataset_3d_id and not eval_run.base_dataset_3d_id:
    from app.models.dataset_3d import AttackDataset3D
    from sqlalchemy import select
    result = await db.execute(
        select(AttackDataset3D).where(AttackDataset3D.id == eval_run.attack_dataset_3d_id)
    )
    attack_dataset_3d = result.scalar_one_or_none()
    if attack_dataset_3d and attack_dataset_3d.base_dataset_id:
        eval_run.base_dataset_3d_id = attack_dataset_3d.base_dataset_id
```

---

## 권장 사항

### 🚨 Immediate (Critical)

#### 1. **Fix dataset_id mismatch in attack evaluation**

**파일**: `backend/app/services/evaluation_service.py`

**수정**:
```python
# Line 498-511: Save attack evaluation results
await self._save_evaluation_results(
    db=db,
    eval_run_id=eval_run.id,
    inference_results=attack_inference_results,
    images=attack_images,
    metrics=combined_metrics,
    eval_logger=eval_logger,
    dataset=attack_dataset_obj,  # ✅ CHANGED: use attack_dataset_obj instead of original_dataset
    dataset_type=EvalDatasetType.ATTACK,
    target_class=target_class,
    iou_threshold=iou_threshold,
    dimension=dimension,
)
```

**추가 변경**: `_save_evaluation_results()`에서 class names 처리
```python
# Line 917: Get class names from dataset metadata
# For ATTACK datasets, we need class names from ORIGINAL dataset
if dataset_type == EvalDatasetType.ATTACK and original_dataset_for_class_names:
    class_names = self._get_class_names_from_dataset(original_dataset_for_class_names)
else:
    class_names = self._get_class_names_from_dataset(dataset) if dataset else None
```

**더 나은 방법**: `_save_evaluation_results()`에 `class_names_dataset` 파라미터 추가
```python
async def _save_evaluation_results(
    self,
    ...
    dataset: Any = None,
    class_names_dataset: Optional[Any] = None,  # NEW: for class name resolution
    ...
):
    # Use class_names_dataset if provided, otherwise use dataset
    class_names = self._get_class_names_from_dataset(
        class_names_dataset or dataset
    ) if (class_names_dataset or dataset) else None
```

#### 2. **Add API endpoints for eval_dataset_results**

**파일**: `backend/app/api/v1/endpoints/evaluation.py`

**추가**:
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
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No dataset results found for this evaluation run"
        )
    return results

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
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No {dataset_type} dataset result found for this evaluation run"
        )
    return result
```

### ⚠️ Short-term (Important)

#### 3. **Fix 3D auto-populate logic**

**파일**: `backend/app/api/v1/endpoints/evaluation.py`

**수정**:
```python
# Line 66-72: Auto-populate base_dataset_id from attack_dataset if needed
if eval_run.attack_dataset_id and not eval_run.base_dataset_id:
    from app import crud
    attack_dataset = await crud.attack_dataset_2d.get(db, id=eval_run.attack_dataset_id)
    if attack_dataset and attack_dataset.base_dataset_id:
        eval_run.base_dataset_id = attack_dataset.base_dataset_id

# ✅ ADD: 3D auto-populate
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
```

#### 4. **Update Frontend to use eval_dataset_results**

**파일**: `frontend/components/evaluation/unified-evaluation-dashboard.tsx`

**추가 API 클라이언트 함수** (`frontend/lib/api-client.ts`):
```typescript
async getEvalDatasetResults(runId: string): Promise<EvalDatasetResult[]> {
  const response = await fetch(`/api/v1/evaluation/runs/${runId}/dataset-results`)
  if (!response.ok) throw new Error('Failed to fetch dataset results')
  return response.json()
}

async getEvalDatasetResultByType(runId: string, datasetType: 'base' | 'attack'): Promise<EvalDatasetResult> {
  const response = await fetch(`/api/v1/evaluation/runs/${runId}/dataset-results/${datasetType}`)
  if (!response.ok) throw new Error('Failed to fetch dataset result')
  return response.json()
}
```

**Frontend 수정** (handleViewResults):
```typescript
// 기존
const evalRun = await apiClient.getEvaluationRun(evalId)
const metrics = evalRun.metrics_summary

// 개선
const evalRun = await apiClient.getEvaluationRun(evalId)
const datasetResults = await apiClient.getEvalDatasetResults(evalId)

// Get metrics from dataset_results
const baseResult = datasetResults.find(r => r.dataset_type === 'base')
const attackResult = datasetResults.find(r => r.dataset_type === 'attack')

const baseMetrics = baseResult?.metrics_summary
const attackMetrics = attackResult?.metrics_summary
```

### 📝 Long-term (Cleanup)

#### 5. **Remove deprecated fields**

**Phase 1** (현재):
- `eval_items.dataset_type` ✅ 유지 (backward compatibility)
- `eval_class_metrics.dataset_type` ✅ 유지
- `eval_runs.metrics_summary` ✅ 유지 (robustness 정보용)

**Phase 2** (모든 코드 마이그레이션 후):
- `eval_items.dataset_type` → 제거 고려
- `eval_class_metrics.dataset_type` → 제거 고려
- `eval_runs.metrics_summary` → robustness만 저장하도록 수정

#### 6. **Add database migration script**

**필요한 마이그레이션**:
1. 기존 데이터 마이그레이션 (eval_runs → eval_dataset_results)
2. eval_items, eval_class_metrics에 eval_dataset_result_id 채우기
3. 데이터 무결성 검증

---

## 요약

### ✅ 구현 완료
1. ✅ `eval_dataset_results` 테이블 추가
2. ✅ `eval_items.eval_dataset_result_id` 추가
3. ✅ `eval_class_metrics.eval_dataset_result_id` 추가
4. ✅ CRUD 함수 구현
5. ✅ `_save_evaluation_results()` 리팩토링
6. ✅ Constraint 완화 (post_attack에서 base optional)

### 🔴 Critical Fixes Needed
1. ❌ **Attack evaluation에서 dataset_id 불일치** → 수정 필요!
2. ❌ **API 엔드포인트 누락** → 추가 필요!

### 🟡 Important Improvements
3. ⚠️ **3D auto-populate 로직** → 추가 필요
4. ⚠️ **Frontend 마이그레이션** → 새 구조 사용

### 🟢 Long-term Cleanup
5. 📝 Deprecated 필드 제거 계획
6. 📝 데이터 마이그레이션 스크립트

---

**다음 단계**: Critical fixes 적용 → 데이터베이스 마이그레이션 → 테스트
