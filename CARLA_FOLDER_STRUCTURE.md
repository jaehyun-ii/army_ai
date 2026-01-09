# CARLA 백엔드 폴더 구조 및 네이밍 규칙

## 개요
CARLA 백엔드 컨테이너와 메인 백엔드 간의 볼륨 마운트 구조 및 파일 경로 규칙을 정리한 문서입니다.

## 볼륨 마운트 구조
```
호스트: /data/workspace/jh/army_ai/storage/3d/exports
  ↓
컨테이너: /workspace/exports (CARLA 백엔드)
  ↓
메인 백엔드: /storage/3d/exports (STORAGE_ROOT 기준)
```

## 1. 데이터셋 생성 (sim_generate)

### 프론트엔드 → 메인 백엔드
- **요청 파라미터**: `dataset_name` (예: `"k2_tank_tmpkkkkksad"`)

### CARLA 백엔드 생성 경로
- **컨테이너 경로**: `/workspace/exports/{dataset_name}/{vehicle_name}_eval/`
- **예시**:
  ```
  /workspace/exports/k2_tank_tmpkkkkksad/k2_tank_eval/image_001.jpg
  /workspace/exports/k2_tank_tmpkkkkksad/k2_tank_eval/image_002.jpg
  /workspace/exports/k2_tank_tmpkkkkksad/annotations/annotations.json
  ```

### 메인 백엔드 저장 경로
- **Storage Path**: `/storage/3d/exports/{dataset_name}`
- **Image Path**: `/storage/3d/exports/{dataset_name}/{vehicle_name}_eval/image.jpg`
- **예시**: `/storage/3d/exports/k2_tank_tmpkkkkksad`

### DB 저장 구조
- **테이블**: `dataset_3d`
- **storage_path**: `/storage/3d/exports/{dataset_name}`
- **metadata**: map_name, weather_name, time_name, object_name, loc_total, batch_total 등

---

## 2. 패치 생성 (sim_gen_patch)

### 프론트엔드 → 메인 백엔드
- **요청 파라미터**:
  - `patch_name` (예: `"3d_k2_tank"`)
  - `object_name` (예: `"K2 전차"`)
  - `attack_method_name` (예: `"ACTIVE"` 또는 `"DTA"`)

### CARLA 백엔드 생성 경로
- **컨테이너 경로**: `/workspace/exports/{adv_method_name}_{vehicle_name}_{patch_name}/`
- **adv_method_name**: `"Active"` (ACTIVE) 또는 `"Dta"` (DTA)
- **vehicle_name**: 차량 영문명 (예: `"k2_tank"`)
- **패치 파일 경로**:
  ```
  /workspace/exports/Active_{vehicle_name}_{patch_name}/attack_pattern/{vehicle_name}_bestval.png
  /workspace/exports/Active_{vehicle_name}_{patch_name}/attack_pattern/{vehicle_name}_besttrain.png
  ```
- **예시**:
  ```
  /workspace/exports/Active_k2_tank_3d_k2_tank/attack_pattern/k2_tank_bestval.png
  /workspace/exports/Active_k2_tank_3d_k2_tank/attack_pattern/k2_tank_besttrain.png
  /workspace/exports/Active_k2_tank_3d_k2_tank/plot/loss_curve.png
  ```

### 메인 백엔드 저장 경로
- **Storage Key**: `3d/exports/Active_{vehicle_name}_{patch_name}/attack_pattern/{vehicle_name}_bestval.png`
- **예시**: `/storage/3d/exports/Active_k2_tank_3d_k2_tank/attack_pattern/k2_tank_bestval.png`

### DB 저장 구조
- **테이블**: `patch_3d`
- **storage_key**: `3d/exports/Active_{vehicle_name}_{patch_name}/attack_pattern/{vehicle_name}_bestval.png`
- **name**: `patch_name`
- **method**: `attack_method_name` (ACTIVE, DTA)
- **target_class**: `object_name`
- **target_model_id**: 사용된 모델 ID (UUID)

---

## 3. 공격 데이터셋 생성 (sim_apply_texture)

### 프론트엔드 → 메인 백엔드
- **요청 파라미터**:
  - `dataset_name` (예: `"k2_tank"`)
  - `attack_method` (예: `"ACTIVE"`)
  - `patch_name` (예: `"3d_k2_tank"`)

### CARLA 백엔드 생성 경로
- **컨테이너 경로**: `/workspace/exports/adv_{dataset_name}/{vehicle_name}_eval/`
- **예시**:
  ```
  /workspace/exports/adv_k2_tank/k2_tank_eval/image_001.jpg
  /workspace/exports/adv_k2_tank/k2_tank_eval/image_002.jpg
  /workspace/exports/adv_k2_tank/annotations/annotations.json
  ```

### 메인 백엔드 저장 경로
- **Storage Path**: `/storage/3d/exports/adv_{dataset_name}`
- **Image Path**: `/storage/3d/exports/adv_{dataset_name}/{vehicle_name}_eval/image.jpg`
- **예시**: `/storage/3d/exports/adv_k2_tank`

### DB 저장 구조
- **테이블**: `attack_dataset_3d` (공격 메타데이터)
- **테이블**: `dataset_3d` (출력 이미지 데이터셋, output_dataset_id로 연결)
- **storage_path**: `/storage/3d/exports/adv_{dataset_name}`
- **parameters**: attack_method, patch_name, map_name, weather_name 등

---

## 경로 변환 로직

### CARLA → 메인 백엔드 경로 변환
`Dataset3DService.convert_carla_path_to_storage()` 함수가 담당:

```python
# CARLA 경로 → 메인 백엔드 경로
"/static/exports/k2_tank/image.jpg" → "/storage/3d/exports/k2_tank/image.jpg"
"/exports/Active_k2_tank_3d_k2_tank/..." → "/storage/3d/exports/Active_k2_tank_3d_k2_tank/..."
"/workspace/exports/adv_k2_tank/..." → "/storage/3d/exports/adv_k2_tank/..."
```

### 폴더 경로 추출 로직
`carla_proxy.py`의 `_proxy_stream_post()` 함수에서 처리:

1. **데이터셋 생성**:
   ```python
   # /storage/3d/exports/k2_tank_tmpkkkkksad/k2_tank_eval/image.jpg
   # → /storage/3d/exports/k2_tank_tmpkkkkksad
   path_parts = image_path.split("/")
   if path_parts[3] == "exports":
       storage_path = "/".join(path_parts[:5])  # exports 폴더 포함
   ```

2. **패치 생성**:
   ```python
   # 이미지 경로 자체를 사용 (파일 단위 저장)
   image_path = "/storage/3d/exports/Active_k2_tank_3d_k2_tank/attack_pattern/k2_tank_bestval.png"
   ```

3. **공격 데이터셋 생성**:
   ```python
   # /storage/3d/exports/adv_k2_tank/k2_tank_eval/image.jpg
   # → /storage/3d/exports/adv_k2_tank
   path_parts = image_path.split("/")
   if path_parts[3] == "exports":
       storage_path = "/".join(path_parts[:5])  # exports 폴더 포함
   ```

---

## 프론트엔드에서 이름 생성 규칙

### 1. 데이터셋 이름
프론트엔드에서 사용자가 입력하거나 자동 생성:
- **패턴**: `{object_name}_{random_suffix}`
- **예시**: `"k2_tank_tmpkkkkksad"`, `"etron_car_eval_20240109"`

### 2. 패치 이름
프론트엔드에서 사용자가 입력:
- **패턴**: `{prefix}_{object_name}`
- **예시**: `"3d_k2_tank"`, `"efficient_k2_tank"`
- **CARLA 생성 폴더**: `Active_{vehicle_name}_{patch_name}` → `"Active_k2_tank_3d_k2_tank"`

### 3. 공격 데이터셋 이름
프론트엔드에서 원본 데이터셋 이름 사용:
- **패턴**: CARLA가 자동으로 `adv_` 접두사 추가
- **예시**: `"k2_tank"` → CARLA가 `"adv_k2_tank"` 폴더 생성

---

## 주의사항

1. **폴더명 중복 방지**:
   - CARLA 백엔드는 같은 이름의 폴더가 이미 존재하면 409 Conflict 에러 반환
   - 프론트엔드에서 유니크한 이름 생성 필요

2. **경로 변환 필수**:
   - CARLA 스트림에서 받은 모든 `image_path`는 반드시 `convert_carla_path_to_storage()` 처리
   - 변환 실패 시 로그 남기고 원본 경로 사용하지 않음

3. **exports 폴더 체크**:
   - 모든 경로에서 `exports` 폴더 포함 여부 확인
   - `path_parts[3] == "exports"` 체크로 정확한 storage_path 추출

4. **어노테이션 폴링**:
   - CARLA가 annotations JSON을 생성할 때까지 최대 30초 대기
   - 타임아웃 시 나중에 `/parse-annotations` 엔드포인트로 수동 파싱 가능

---

## 트러블슈팅

### 문제: 이미지가 DB에 저장되지 않음
- **확인**: storage_path가 `/storage/3d/exports/...` 형식인지 확인
- **해결**: `convert_carla_path_to_storage()` 함수가 제대로 호출되었는지 확인

### 문제: 폴더 경로가 잘못 추출됨
- **확인**: `path_parts[3] == "exports"` 체크가 누락되지 않았는지 확인
- **해결**: `carla_proxy.py:509-510` 로직 참고

### 문제: 어노테이션이 저장되지 않음
- **확인**: `annotations` 폴더가 생성되었는지 확인
- **해결**: 30초 폴링 타임아웃 후 수동 파싱 엔드포인트 사용

---

## 참고 파일
- `backend/app/api/v1/endpoints/carla_proxy.py` - 메인 프록시 엔드포인트
- `backend/app/services/carla/dataset_3d_service.py` - 데이터셋 서비스
- `backend/app/services/carla/patch_3d_service.py` - 패치 서비스
- `backend/app/services/carla/attack_dataset_3d_service.py` - 공격 데이터셋 서비스
- `backend/app/services/carla/annotation_parser.py` - 어노테이션 파서

---

**작성일**: 2026-01-09
**작성자**: Claude Code Analysis
