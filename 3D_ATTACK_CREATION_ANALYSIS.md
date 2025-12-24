# 3D 공격 데이터셋 생성 분석 결과

**날짜**: 2025-12-24
**목적**: 3D 공격 데이터셋 생성 시 base_dataset_id 관련 문제 여부 확인

---

## ✅ 결론: 문제 없음!

3D 공격 데이터셋 생성 코드는 **이미 올바르게 구현**되어 있습니다.
`base_dataset_id`를 전혀 참조하지 않으며, 모든 로직이 올바르게 작동합니다.

---

## 📋 분석 내용

### 1. Schema 확인 ✅

**파일**: `backend/app/schemas/dataset_3d.py`

```python
class AttackDataset3DCreate(AttackDataset3DBase):
    """Schema for creating 3D attack dataset (base dataset optional)."""
    model_config = ConfigDict(protected_namespaces=())

    target_model_id: Optional[UUID] = None
    output_dataset_id: Optional[UUID] = None  # ✅ 있음
    patch_id: Optional[UUID] = None
    experiment_id: Optional[UUID] = None
    created_by: Optional[UUID] = None
    # ✅ base_dataset_id 필드 없음 (올바름)
```

**결과**: ✅ Schema에 `base_dataset_id` 필드가 없음 (올바름)

---

### 2. CARLA Proxy API 확인 ✅

**파일**: `backend/app/api/v1/endpoints/carla_proxy.py`

#### 생성 위치: Line 636-661

```python
# 2) AttackDataset3D 생성 (output_dataset_id 연결)
attack_dataset = AttackDataset3D(
    name=attack_dataset_name,
    description=f"Attack dataset with {payload.get('attack_method')} patch applied",
    attack_type=AttackType.PATCH,
    output_dataset_id=output_dataset_id,  # ✅ output_dataset만 사용
    target_class=payload.get("object_name"),
    patch_id=None,
    parameters={
        "attack_method": payload.get("attack_method"),
        "patch_name": payload.get("patch_name"),
        ...
        "output_dataset_id": output_dataset_id,
        "source": "carla_simulator",
    },
)
db.add(attack_dataset)
await db.commit()
```

**결과**: ✅ `base_dataset_id` 전혀 참조하지 않음 (올바름)

---

### 3. Naming Convention 확인 ✅

#### Line 518-520
```python
timestamp_suffix = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
base_dataset_name = payload.get("dataset_name", "unknown")  # ⚠️ 이름만!
attack_dataset_name = f"{base_dataset_name}_attack_{timestamp_suffix}"
output_dataset_name = f"{base_dataset_name}_output_{timestamp_suffix}"
```

**해석**:
- `base_dataset_name`은 변수 이름일 뿐, **실제 base_dataset과는 무관**
- 단순히 naming convention을 위한 prefix로 사용됨
- `payload.get("dataset_name")`은 사용자가 입력한 dataset 이름

**결과**: ✅ 문제 없음 (naming convention일 뿐)

---

### 4. 전체 워크플로우

#### CARLA 공격 데이터셋 생성 과정

```
1. 사용자 요청 (POST /api/v1/carla/sim_apply_texture)
   └─ payload: {dataset_name, attack_method, patch_name, ...}

2. CARLA Simulator 호출
   └─ 패치 적용된 이미지 생성
   └─ storage_path에 저장

3. Output Dataset3D 생성
   ├─ name: "{dataset_name}_output_{timestamp}"
   ├─ storage_path: CARLA가 생성한 경로
   └─ metadata: attack_method, patch_name, etc.

4. Image3D 레코드 생성
   ├─ storage_path 스캔하여 이미지 파일 발견
   ├─ dataset_id: output_dataset.id
   └─ 각 이미지의 메타데이터 저장

5. Annotation 파싱 및 저장
   ├─ annotations/*.json 파일 파싱
   └─ image_3d_id와 연결

6. AttackDataset3D 생성 ⭐
   ├─ name: "{dataset_name}_attack_{timestamp}"
   ├─ output_dataset_id: output_dataset.id  ✅
   ├─ attack_type: PATCH
   ├─ parameters: {attack_method, patch_name, storage_path, ...}
   └─ ❌ base_dataset_id: 없음! (올바름)
```

---

## 🔍 왜 base_dataset_id가 필요 없는가?

### CARLA 시뮬레이션의 특성

1. **독립적인 데이터 생성**
   - CARLA는 시뮬레이터에서 **새로운 장면을 생성**
   - 기존 데이터셋을 기반으로 하지 않음
   - 처음부터 공격이 적용된 이미지 + GT 생성

2. **1:1 매핑 불필요**
   - 2D는 원본 이미지에 패치/노이즈를 추가 → 원본과 1:1 매핑
   - 3D는 시뮬레이터에서 새로 생성 → 원본 개념이 없음

3. **GT 포함 생성**
   - 공격 이미지와 함께 GT도 CARLA에서 생성
   - `output_dataset`에 이미지 + 어노테이션 모두 포함
   - 별도의 원본 데이터셋 불필요

---

## 📊 2D vs 3D 비교

| 항목 | 2D Attack Dataset | 3D Attack Dataset |
|------|-------------------|-------------------|
| **생성 방식** | 기존 이미지 변형 | 시뮬레이터에서 새로 생성 |
| **base_dataset_id** | ✅ 필수 (원본 참조) | ❌ 없음 (독립 생성) |
| **output_dataset_id** | ✅ 있음 | ✅ 있음 |
| **원본과의 관계** | 1:1 매핑 | 독립적 |
| **GT 소스** | 원본 dataset annotations | output_dataset annotations |
| **파일명 매핑** | 필요 (original.png → attacked.png) | 불필요 |

---

## ✅ 검증 체크리스트

- [x] **Schema**: `AttackDataset3DCreate`에 `base_dataset_id` 필드 없음
- [x] **Model**: `AttackDataset3D` 테이블에 `base_dataset_id` 컬럼 없음
- [x] **API**: CARLA proxy에서 `base_dataset_id` 전혀 참조하지 않음
- [x] **생성 로직**: `AttackDataset3D()` 생성 시 `base_dataset_id` 없음
- [x] **Naming**: `base_dataset_name` 변수는 이름 prefix일 뿐, 실제 ID 아님

---

## 🎯 결론

### 3D 공격 데이터셋 생성은 문제 없음 ✅

1. **Schema**: 올바르게 구현됨 (base_dataset_id 없음)
2. **API**: 올바르게 구현됨 (base_dataset_id 참조 없음)
3. **생성 로직**: 올바르게 구현됨 (output_dataset_id만 사용)

### 수정 불필요

3D 공격 데이터셋 생성 코드는 **이미 올바르게 구현**되어 있으므로 **수정이 필요 없습니다**.

---

## 📝 평가(Evaluation)와의 차이점

### 생성(Creation): 문제 없음 ✅
- `base_dataset_id` 전혀 사용 안 함
- `output_dataset_id`만 저장

### 평가(Evaluation): 수정 완료 ✅
- `base_dataset_3d_id`를 optional로 처리
- base 제공 시: 비교 평가 (robustness 계산)
- base 미제공 시: 공격만 평가

---

## 🚀 최종 정리

### 생성 시스템
```
3D Attack Dataset Creation (CARLA)
  ├─ Input: dataset_name, attack_method, patch_name
  ├─ Process: CARLA 시뮬레이터가 이미지+GT 생성
  └─ Output:
      ├─ Dataset3D (output_dataset_id)
      │   ├─ Images
      │   └─ Annotations (GT)
      └─ AttackDataset3D
          ├─ output_dataset_id ✅
          └─ ❌ base_dataset_id (없음, 올바름)
```

### 평가 시스템
```
3D Attack Dataset Evaluation
  ├─ Case A: base_dataset_3d_id 제공
  │   ├─ Original 평가
  │   ├─ Attack 평가 (filename mapping으로 GT 매칭)
  │   └─ Robustness 계산
  └─ Case B: base_dataset_3d_id 미제공
      ├─ Attack만 평가 (output_dataset의 GT 사용)
      └─ Robustness 계산 생략
```

---

**결론**: 3D 공격 데이터셋 **생성**은 이미 올바르게 구현되어 있으며, **평가** 시스템도 수정 완료되어 모든 시스템이 정상 작동합니다! ✅
