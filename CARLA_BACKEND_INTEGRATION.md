# CARLA 백엔드 통합 가이드

## 개요

메인 백엔드 (`army_ai_backend_dev`)와 CARLA 백엔드 (`army_ai_carla_backend_dev`) 간의 3D 패치 생성 연동을 위한 수정사항 문서입니다.

---

## 1. 메인 백엔드 수정사항 (✅ 완료)

### 파일: `backend/app/api/v1/endpoints/carla_proxy.py`

#### 1.1 요청 스키마 업데이트 (line 50-63)

**변경사항**: `config_path` 필드 추가

```python
class PatchGenerationRequest(BaseModel):
    """Request schema for patch generation."""
    patch_name: str
    object_name: str
    attack_method_name: str
    model_id: Optional[UUID] = None
    object_detection_name: Optional[str] = None

    # 자동으로 채워지는 필드
    model_path: str = ""
    config_path: str = ""  # ← 추가됨 (EfficientDet 등 MMDet 모델용)
    model_type: str = ""
    class_name: List[str] = Field(default_factory=lambda: ["car"])
    input_size: List[int] = Field(default_factory=lambda: [640, 640])
    device_type: str = ""
```

#### 1.2 CONFIG artifact 조회 로직 추가 (line 832-845)

**변경사항**: WEIGHTS artifact와 함께 CONFIG artifact도 조회하여 payload에 추가

```python
# CONFIG artifact 조회 (EfficientDet 등 MMDetection 모델용)
config_stmt = select(ODModelArtifact).where(
  ODModelArtifact.model_id == request.model_id,
  ODModelArtifact.artifact_type == ArtifactType.CONFIG
)
config_result = await db.execute(config_stmt)
config_artifact = config_result.scalar_one_or_none()

if config_artifact:
  payload["config_path"] = config_artifact.storage_path
  logger.info(f"Added config_path to payload: {config_artifact.storage_path}")
else:
  logger.debug(f"No CONFIG artifact found for model {request.model_id}")
  payload["config_path"] = ""
```

#### 1.3 input_size 추출 로직 수정 (line 856-869)

**변경 전**:
```python
# ❌ 잘못된 키
if model.input_spec and "size" in model.input_spec:
    input_size_val = model.input_spec["size"]
```

**변경 후**:
```python
# ✅ 올바른 키 ("shape")
if model.input_spec and "shape" in model.input_spec:
  shape = model.input_spec["shape"]
  if isinstance(shape, list) and len(shape) >= 2:
    # [768, 768, 3] → [768, 768]
    payload["input_size"] = shape[:2]
    logger.info(f"Extracted input_size from input_spec.shape: {payload['input_size']}")
```

**이유**: DB에 저장된 형식은 `{"shape": [768, 768, 3], "dtype": "float32"}` 이므로 "size" 키가 아닌 "shape" 키를 사용해야 함

---

## 2. CARLA 백엔드 수정사항 (⚠️ 필요)

### 2.1 파일: `/workspace/utils/tasks.py`

**위치**: `run_attack_generation` 함수 (line 77-135)

**현재 상태**:
```python
def run_attack_generation(
    carla_host: str,
    carla_port: int,
    carla_timeout: float,
    patch_name: str,
    object_name: str,
    attack_method_name: str,
    object_detection_name: str,
    model_path: str,
    model_type: str,
    class_name: list,
    input_size: list,
    device_type: str,
    stream_queue: queue.Queue,
) -> None:
```

**수정 필요**:
```python
def run_attack_generation(
    carla_host: str,
    carla_port: int,
    carla_timeout: float,
    patch_name: str,
    object_name: str,
    attack_method_name: str,
    object_detection_name: str,
    model_path: str,
    config_path: str,  # ← 추가 필요
    model_type: str,
    class_name: list,
    input_size: list,
    device_type: str,
    stream_queue: queue.Queue,
) -> None:
```

**custom_od_load_parameter 구성 수정**:

**변경 전** (line 120-128):
```python
custom_od_load_parameter = {
    "model_path": model_path,
    "model_type": model_type,
    "class_name": class_name,
    "input_size": input_size,
    "device_type": device_type,
}
```

**변경 후**:
```python
custom_od_load_parameter = {
    "model_path": model_path,
    "config_path": config_path,  # ← 추가
    "model_type": model_type,
    "class_name": class_name,
    "input_size": input_size,
    "device_type": device_type,
}
```

### 2.2 파일: `/workspace/main.py`

**위치**: `sim_gen_patch` 엔드포인트 (line 230-268)

**수정 필요**:

**변경 전**:
```python
@app.post("/sim_gen_patch")
async def sim_gen_patch(
    patch_name: str = Body("3d_k2_tank"),
    object_name: str = Body("K2 전차"),
    attack_method_name: str = Body("ACTIVE"),
    object_detection_name: str = Body("efficientdet_d2_tank"),
    model_path: str = Body(""),
    model_type: str = Body(""),
    class_name: list = Body(["car"]),
    input_size: list = Body([640, 640]),
    device_type: str = Body(""),
):
```

**변경 후**:
```python
@app.post("/sim_gen_patch")
async def sim_gen_patch(
    patch_name: str = Body("3d_k2_tank"),
    object_name: str = Body("K2 전차"),
    attack_method_name: str = Body("ACTIVE"),
    object_detection_name: str = Body("efficientdet_d2_tank"),
    model_path: str = Body(""),
    config_path: str = Body(""),  # ← 추가
    model_type: str = Body(""),
    class_name: list = Body(["car"]),
    input_size: list = Body([640, 640]),
    device_type: str = Body(""),
):
```

**run_attack_generation 호출 수정**:

**변경 전** (line 248-262):
```python
job = job_manager.create_job(
    "sim_gen_patch",
    run_attack_generation,
    global_args.carla_host,
    global_args.carla_port,
    config.TIMEOUT_DEFAULT,
    patch_name,
    object_name,
    attack_method_name,
    object_detection_name,
    model_path,
    model_type,
    class_name,
    input_size,
    device_type,
    timeout_seconds=None,
)
```

**변경 후**:
```python
job = job_manager.create_job(
    "sim_gen_patch",
    run_attack_generation,
    global_args.carla_host,
    global_args.carla_port,
    config.TIMEOUT_DEFAULT,
    patch_name,
    object_name,
    attack_method_name,
    object_detection_name,
    model_path,
    config_path,  # ← 추가
    model_type,
    class_name,
    input_size,
    device_type,
    timeout_seconds=None,
)
```

### 2.3 파일: `/workspace/object_detection/framework/pt/custom_od/main.py`

**위치**: `CustomPTModel.load_config` 메서드 (line 18-45)

**변경 목적**: config_path가 제공되면 glob 대신 직접 사용

**변경 전** (line 23-32):
```python
if self.custom_od_load_parameter['model_type'] == 'efficientdet':
    checkpoint = os.path.abspath(self.custom_od_load_parameter['model_path'])
    config = glob.glob(f"{os.path.dirname(checkpoint)}/*.py")[0]  # ← glob으로 찾음
    config_form = {
        'model_name': model_name,
        'labels': self.custom_od_load_parameter['class_name'],
        'config': config,
        'checkpoint': checkpoint
    }
```

**변경 후**:
```python
if self.custom_od_load_parameter['model_type'] == 'efficientdet':
    checkpoint = os.path.abspath(self.custom_od_load_parameter['model_path'])

    # config_path가 제공되면 사용, 없으면 glob으로 찾기 (하위호환성)
    if 'config_path' in self.custom_od_load_parameter and self.custom_od_load_parameter['config_path']:
        config = os.path.abspath(self.custom_od_load_parameter['config_path'])
        print(f"Using provided config_path: {config}")
    else:
        config = glob.glob(f"{os.path.dirname(checkpoint)}/*.py")[0]
        print(f"Config path not provided, found via glob: {config}")

    config_form = {
        'model_name': model_name,
        'labels': self.custom_od_load_parameter['class_name'],
        'config': config,
        'checkpoint': checkpoint
    }
```

---

## 3. 전달되는 파라미터 전체 흐름

### 3.1 프론트엔드 → 메인 백엔드

**요청 예시**:
```json
{
  "patch_name": "ultrathink_tank_patch",
  "object_name": "K2 전차",
  "attack_method_name": "ACTIVE",
  "model_id": "uuid-here"
}
```

### 3.2 메인 백엔드 → CARLA 백엔드

**DB 조회 후 구성된 payload**:
```json
{
  "patch_name": "ultrathink_tank_patch",
  "object_name": "K2 전차",
  "attack_method_name": "ACTIVE",
  "object_detection_name": "Ultrathink Efficientdet D2",
  "model_path": "/storage/models/ultrathink_efficientdet_d2/model.pt",
  "config_path": "/storage/models/ultrathink_efficientdet_d2/config.py",
  "model_type": "pytorch",
  "class_name": ["tank", "soldier", "vehicle"],
  "input_size": [768, 768],
  "device_type": ""
}
```

### 3.3 CARLA 백엔드 처리

1. **main.py**: 요청 받기
2. **utils/tasks.py**: `custom_od_load_parameter` 구성
3. **dta/active_attack.py**: ACTIVE 프레임워크 초기화
4. **object_detection/framework/pt/custom_od/main.py**:
   - `config_path` 사용하여 MMDetection 모델 로드
   - `input_size` [768, 768] 사용

---

## 4. 테스트 방법

### 4.1 메인 백엔드 수정사항 확인

```bash
# 메인 백엔드 컨테이너 재시작
docker restart army_ai_backend_dev

# 로그 확인
docker logs -f army_ai_backend_dev | grep "config_path\|input_size"
```

### 4.2 CARLA 백엔드 수정 후 테스트

```bash
# SSH 접속
ssh oem@192.168.0.4

# CARLA 백엔드 컨테이너 접속
docker exec -it 50300fe91848 bash

# 파일 수정 (위의 2.1, 2.2, 2.3 참고)
vi /workspace/utils/tasks.py
vi /workspace/main.py
vi /workspace/object_detection/framework/pt/custom_od/main.py

# CARLA 백엔드 재시작
exit
docker restart 50300fe91848

# 로그 확인
docker logs -f 50300fe91848
```

### 4.3 통합 테스트

1. 프론트엔드에서 Ultrathink Efficientdet D2 모델 선택
2. 패치 생성 요청
3. 메인 백엔드 로그에서 확인:
   ```
   INFO: Added config_path to payload: /storage/models/ultrathink_efficientdet_d2/config.py
   INFO: Extracted input_size from input_spec.shape: [768, 768]
   ```
4. CARLA 백엔드 로그에서 확인:
   ```
   Using provided config_path: /storage/models/ultrathink_efficientdet_d2/config.py
   ```

---

## 5. 문제 해결

### 5.1 "Config file not found" 에러

**원인**: config.py 파일이 storage에 없음

**해결**:
1. 모델 업로드 시 config.py 파일도 함께 업로드했는지 확인
2. DB의 `od_model_artifacts` 테이블에서 CONFIG artifact 존재 확인
3. storage 경로에 실제 파일 존재 확인

### 5.2 "Invalid input size" 에러

**원인**: input_size가 [640, 640]으로 전달됨 (768이 아님)

**해결**:
1. `od_models` 테이블의 `input_spec` 확인
2. 메인 백엔드 로그에서 "Extracted input_size" 메시지 확인
3. CARLA 백엔드로 전달된 payload 로그 확인

---

## 6. 주의사항

1. **하위 호환성**: 내장 모델 ("yolov11", "yolov8", "rt-detr", "efficientdet_d2_tank")은 기존 방식대로 작동
2. **파일 경로**: storage 경로가 메인 백엔드와 CARLA 백엔드 간에 공유되어야 함 (docker volume mount)
3. **config.py 필수**: EfficientDet 모델은 config.py 없이는 로드 불가능

---

## 7. 관련 파일 목록

### 메인 백엔드
- `backend/app/api/v1/endpoints/carla_proxy.py` ✅ 수정 완료
- `backend/app/api/v1/endpoints/models.py` (모델 업로드)
- `backend/app/models/model_repo.py` (DB 모델)

### CARLA 백엔드 (192.168.0.4:50300fe91848)
- `/workspace/main.py` ⚠️ 수정 필요
- `/workspace/utils/tasks.py` ⚠️ 수정 필요
- `/workspace/object_detection/framework/pt/custom_od/main.py` ⚠️ 수정 필요
- `/workspace/dta/active_attack.py` (수정 불필요)
- `/workspace/dta/framework/active_framework.py` (수정 불필요)

---

## 8. 참고 자료

- EfficientDet config 파일 예시: `/storage/models/ultrathink_efficientdet_d2/config.py`
- MMDetection 문서: https://mmdetection.readthedocs.io/
- CARLA 백엔드 API 문서: `http://192.168.0.4:8888/docs`
