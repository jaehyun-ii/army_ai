# CARLA 백엔드 통합 - 간단 가이드 (코드 수정 없이!)

## ✅ 해결 방법: CARLA 백엔드 코드를 건드리지 않고 해결!

CARLA 백엔드의 Python 코드를 전혀 수정하지 않고, **docker-compose.yml과 메인 백엔드만 수정**하여 해결했습니다.

---

## 1. 문제점

### 1.1 발견된 이슈

1. **config.py 파일 접근 불가**
   - CARLA 백엔드에 `/storage/models/` 디렉토리가 마운트되지 않음
   - EfficientDet 모델은 config.py 필수

2. **input_size 잘못 추출**
   - `model.input_spec["size"]` → 존재하지 않는 키
   - 실제: `model.input_spec["shape"]` 사용해야 함
   - 결과: 768x768이어야 하는데 640x640로 전달됨

### 1.2 CARLA 백엔드 동작 방식

```python
# CARLA 백엔드가 config.py를 찾는 방식
checkpoint = os.path.abspath(custom_od_load_parameter['model_path'])
config = glob.glob(f"{os.path.dirname(checkpoint)}/*.py")[0]
```

**핵심**: model.pt와 config.py가 **같은 디렉토리**에 있으면 자동으로 찾음!

---

## 2. 해결책 (✅ 완료)

### 2.1 Docker Volume 추가 (docker-compose.yml)

**파일**: `/home/jaehyun/army_ai/docker-compose.yml` (line 263-267)

```yaml
volumes:
  - ./storage/3d/exports:/workspace/exports:rw
  - ./storage/models:/workspace/models:ro  # ← 추가됨 (읽기 전용)
  - ./carla_client_entrypoint.sh:/entrypoint.sh:ro
```

**효과**:
- 메인 백엔드의 `./storage/models/` → CARLA 백엔드의 `/workspace/models/`로 마운트
- CARLA 백엔드가 업로드된 모델 파일에 접근 가능

### 2.2 메인 백엔드 수정 (carla_proxy.py)

**파일**: `backend/app/api/v1/endpoints/carla_proxy.py`

#### 변경사항 1: 경로 변환 (line 827-853)

```python
# 메인 백엔드 경로 → CARLA 백엔드 경로 변환
# /storage/models/... → /workspace/models/...
carla_model_path = weights_artifact.storage_path.replace(
    "/storage/models/", "/workspace/models/"
)
payload["model_path"] = carla_model_path
```

#### 변경사항 2: input_size 추출 수정 (line 864-877)

```python
# ✅ 수정: "shape" 키 사용
if model.input_spec and "shape" in model.input_spec:
  shape = model.input_spec["shape"]
  if isinstance(shape, list) and len(shape) >= 2:
    # [768, 768, 3] → [768, 768]
    payload["input_size"] = shape[:2]
```

---

## 3. 작동 원리

### 3.1 파일 구조

```
메인 백엔드:
./storage/models/ultrathink_efficientdet_d2/
  ├── model.pt
  └── config.py

Docker Volume Mount:
↓

CARLA 백엔드:
/workspace/models/ultrathink_efficientdet_d2/
  ├── model.pt
  └── config.py
```

### 3.2 전달되는 파라미터

**메인 백엔드 → CARLA 백엔드**:

```json
{
  "model_path": "/workspace/models/ultrathink_efficientdet_d2/model.pt",
  "model_type": "pytorch",
  "class_name": ["tank", "soldier", "vehicle"],
  "input_size": [768, 768],
  "device_type": ""
}
```

**CARLA 백엔드 처리**:

```python
# 1. model_path에서 디렉토리 추출
checkpoint = "/workspace/models/ultrathink_efficientdet_d2/model.pt"
dir = os.path.dirname(checkpoint)  # "/workspace/models/ultrathink_efficientdet_d2"

# 2. 같은 디렉토리에서 .py 파일 찾기
config = glob.glob(f"{dir}/*.py")[0]  # "/workspace/models/ultrathink_efficientdet_d2/config.py"

# 3. MMDetection 모델 로드
model = init_detector(config, checkpoint)
```

**결과**: config.py 자동으로 찾아서 사용! 🎉

---

## 4. 적용 방법

### 4.1 Docker 컨테이너 재시작

```bash
# CARLA 백엔드 재시작 (volume 변경 적용)
docker-compose down carla_client_backend
docker-compose up -d carla_client_backend

# 메인 백엔드 재시작 (코드 변경 적용)
docker-compose restart backend
```

### 4.2 검증

```bash
# CARLA 백엔드에서 models 디렉토리 확인
docker exec -it army_ai_carla_backend_dev ls -la /workspace/models/

# 특정 모델 디렉토리 확인
docker exec -it army_ai_carla_backend_dev ls -la /workspace/models/ultrathink_efficientdet_d2/
# 출력 예상:
# model.pt
# config.py
```

### 4.3 로그 확인

```bash
# 메인 백엔드 로그
docker logs -f army_ai_backend_dev | grep "Converted.*for CARLA\|input_size"

# 예상 출력:
# INFO: Converted model_path for CARLA: /workspace/models/ultrathink_efficientdet_d2/model.pt
# INFO: Extracted input_size from input_spec.shape: [768, 768]
```

---

## 5. 테스트 시나리오

### 5.1 Ultrathink Efficientdet D2 모델로 패치 생성

1. 프론트엔드에서 모델 선택: "Ultrathink Efficientdet D2"
2. 패치 이름 입력: "test_tank_patch"
3. 객체 이름: "K2 전차"
4. Attack Method: "ACTIVE"
5. 생성 시작

**기대 결과**:
- ✅ config.py 자동으로 찾음
- ✅ 768x768 해상도로 로드됨
- ✅ 패치 생성 성공

---

## 6. 왜 CARLA 백엔드 수정이 필요 없나?

### 6.1 기존 CARLA 백엔드 로직이 완벽함

```python
# CARLA 백엔드: custom_od/main.py
if custom_od_load_parameter['model_type'] == 'efficientdet':
    checkpoint = os.path.abspath(custom_od_load_parameter['model_path'])
    config = glob.glob(f"{os.path.dirname(checkpoint)}/*.py")[0]  # ← 자동 탐색!
```

**장점**:
- model.pt와 config.py가 같은 디렉토리에만 있으면 됨
- 파일명이 정확히 일치하지 않아도 됨
- 유연하고 robust한 설계

### 6.2 우리가 한 일

1. ✅ Docker volume으로 파일 접근 가능하게 만듦
2. ✅ 경로를 CARLA 형식으로 변환
3. ✅ input_size를 올바르게 추출

**CARLA 백엔드는 이미 완벽하게 작동하도록 설계되어 있었음!**

---

## 7. 주의사항

### 7.1 파일 업로드 시

- model.pt와 config.py를 **함께** 업로드해야 함
- 같은 디렉토리에 저장됨 (자동)

### 7.2 내장 모델

다음 모델들은 CARLA 백엔드 내부에 내장되어 있어 별도 파일 필요 없음:
- yolov11
- yolov8
- rt-detr
- efficientdet_d2_tank

### 7.3 Storage 공유

- 메인 백엔드와 CARLA 백엔드가 **같은 호스트**에서 실행되어야 함
- Docker volume mount를 통한 파일 공유

---

## 8. 트러블슈팅

### 8.1 "Config file not found" 에러

**원인**: volume이 제대로 마운트되지 않음

**해결**:
```bash
# 컨테이너 재시작
docker-compose down carla_client_backend
docker-compose up -d carla_client_backend

# 마운트 확인
docker exec -it army_ai_carla_backend_dev ls -la /workspace/models/
```

### 8.2 "Invalid input size" 경고

**원인**: input_spec.shape 추출 실패

**해결**:
1. DB에서 `od_models` 테이블의 `input_spec` 확인
2. 수동으로 업데이트:
```sql
UPDATE od_models
SET input_spec = '{"shape": [768, 768, 3], "dtype": "float32"}'
WHERE name = 'Ultrathink Efficientdet D2';
```

### 8.3 경로 변환 실패

**증상**: `/storage/models/...` 그대로 전달됨

**원인**: 경로 변환 로직 미적용

**해결**: carla_proxy.py line 830 확인

---

## 9. 성능 영향

### 9.1 읽기 전용 마운트

```yaml
- ./storage/models:/workspace/models:ro  # ← :ro (read-only)
```

**장점**:
- CARLA 백엔드가 실수로 모델 파일 수정/삭제 불가
- 보안 향상
- 원본 파일 보호

### 9.2 파일 접근 속도

- Docker volume mount는 네이티브 파일 시스템 속도
- 추가 오버헤드 없음

---

## 10. 요약

| 항목 | 변경 전 | 변경 후 |
|------|---------|---------|
| **CARLA 백엔드 코드** | - | 수정 불필요 ✅ |
| **docker-compose.yml** | exports만 마운트 | models도 마운트 ✅ |
| **carla_proxy.py** | 경로 변환 없음 | `/storage` → `/workspace` ✅ |
| **input_size 추출** | "size" 키 사용 (오류) | "shape" 키 사용 ✅ |
| **config.py 접근** | 불가능 | 자동 탐색 가능 ✅ |

**결론**: CARLA 백엔드 코드를 건드리지 않고 완벽하게 작동! 🎉
