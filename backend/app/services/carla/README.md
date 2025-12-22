# CARLA Services Layer

CARLA 백엔드 프록시 로직을 서비스 레이어로 분리하여 코드의 재사용성과 유지보수성을 향상시킵니다.

## 디렉토리 구조

```
backend/app/services/carla/
├── __init__.py                    # 서비스 export
├── annotation_parser.py           # 어노테이션 파싱 서비스
├── dataset_3d_service.py          # Dataset3D 생성/관리 서비스
├── patch_3d_service.py            # Patch3D 생성/관리 서비스
├── attack_dataset_3d_service.py   # AttackDataset3D 생성/관리 서비스
└── README.md                      # 이 파일
```

## 서비스 설명

### 1. AnnotationParser

CARLA에서 생성된 COCO 형식 어노테이션을 DB에 저장합니다.

**주요 메서드:**
- `parse_and_save()`: 어노테이션 파일 파싱 및 DB 저장

**사용 예:**
```python
from app.services.carla import AnnotationParser

annotation_count = await AnnotationParser.parse_and_save(
    annotations_dir=Path("/storage/3d/k2_tank/annotations"),
    dataset_id=dataset_id,
    image_id_map={"image1.jpg": uuid1, "image2.jpg": uuid2},
    db=db
)
```

### 2. Dataset3DService

3D 데이터셋 생성 및 이미지/어노테이션 관리를 담당합니다.

**주요 메서드:**
- `create_dataset()`: Dataset3D 레코드 생성
- `save_images_and_annotations()`: 이미지 파일 스캔 및 어노테이션 파싱
- `convert_carla_path_to_storage()`: CARLA 경로를 storage 경로로 변환
- `infer_image_count()`: 메타데이터에서 이미지 개수 추론

**사용 예:**
```python
from app.services.carla import Dataset3DService

# 데이터셋 생성
dataset = await Dataset3DService.create_dataset(
    dataset_name="k2_tank_dataset",
    storage_path="/storage/3d/k2_tank",
    metadata={"map_name": "Town01", ...},
    db=db
)

# 이미지 및 어노테이션 저장
image_count, annotation_count = await Dataset3DService.save_images_and_annotations(
    dataset=dataset,
    storage_path="/storage/3d/k2_tank",
    db=db,
    parse_annotations=True
)
```

### 3. Patch3DService

3D 적대적 패치 생성 및 관리를 담당합니다.

**주요 메서드:**
- `save_patch_metadata()`: 패치 메타데이터 DB 저장

**사용 예:**
```python
from app.services.carla import Patch3DService

patch = await Patch3DService.save_patch_metadata(
    image_path="/storage/3d/Active_3d_k2_tank/attack_pattern/k2_tank_bestval.png",
    patch_name="k2_tank_active_patch",
    method="ACTIVE",
    target_class="k2_tank",
    model_id=model_id,
    db=db,
    training_metadata={"epoch": 99, "loss": 0.001}
)
```

### 4. AttackDataset3DService

3D 적대적 공격 데이터셋 생성 및 관리를 담당합니다.

**주요 메서드:**
- `create_attack_dataset()`: AttackDataset3D 및 출력 Dataset3D 생성

**사용 예:**
```python
from app.services.carla import AttackDataset3DService

attack_dataset = await AttackDataset3DService.create_attack_dataset(
    payload={
        "dataset_name": "k2_tank",
        "attack_method": "ACTIVE",
        "patch_name": "k2_tank_patch",
        "object_name": "k2_tank",
        ...
    },
    storage_path="/storage/3d/adv_k2_tank",
    processed_images=100,
    loc_total=10,
    batch_total=10,
    db=db
)
```

## 통합 흐름

### 3D 데이터 생성 흐름
```
1. CARLA 백엔드에서 데이터 생성
2. Dataset3DService.create_dataset() - 데이터셋 생성
3. Dataset3DService.save_images_and_annotations() - 이미지/어노테이션 저장
   └─> AnnotationParser.parse_and_save() - 어노테이션 파싱
```

### 3D 패치 생성 흐름
```
1. CARLA 백엔드에서 패치 학습
2. Patch3DService.save_patch_metadata() - 패치 메타데이터 저장
```

### 3D 공격 데이터셋 생성 흐름
```
1. CARLA 백엔드에서 패치 적용
2. AttackDataset3DService.create_attack_dataset()
   ├─> Dataset3DService.create_dataset() - 출력 데이터셋 생성
   ├─> Dataset3DService.save_images_and_annotations() - 이미지/어노테이션 저장
   │   └─> AnnotationParser.parse_and_save() - 어노테이션 파싱
   └─> AttackDataset3D 레코드 생성
```

## 리팩토링 전후 비교

### Before (carla_proxy.py)
- 1400+ 줄의 단일 파일
- 비즈니스 로직이 엔드포인트에 혼재
- 코드 재사용 어려움
- 테스트 작성 어려움

### After (서비스 레이어 분리)
- 각 서비스가 단일 책임 원칙 준수
- 비즈니스 로직과 API 레이어 분리
- 서비스 간 조합 가능
- 단위 테스트 작성 용이

## 향후 개선 사항

1. **엔드포인트 분리**: carla_proxy.py를 여러 파일로 분리
   - `simulator.py`: 시뮬레이터 설정 (맵, 날씨 등)
   - `datasets_3d.py`: 3D 데이터셋 CRUD
   - `patches_3d.py`: 3D 패치 CRUD
   - `attack_datasets_3d.py`: 3D 공격 데이터셋 CRUD

2. **프록시 서비스 추가**: CARLA 백엔드 통신을 별도 서비스로 분리
   - `proxy_service.py`: HTTP 프록시 로직
   - SSE 스트리밍 처리
   - 에러 핸들링

3. **테스트 추가**: 각 서비스에 대한 단위 테스트 작성

## 데이터베이스 마이그레이션

이 서비스 레이어는 `annotations` 테이블에 `image_3d_id` 컬럼을 필요로 합니다.

**마이그레이션 실행:**
```bash
bash /home/jaehyun/army_ai/apply_migration.sh
```

또는:

```bash
docker exec army_ai_postgres psql -U admin -d armydb -f /database/migrations/add_image_3d_annotations.sql
```
