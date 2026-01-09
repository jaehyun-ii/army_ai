# DB Storage 구조 상세 설명

## 개요
CARLA 백엔드에서 생성된 파일들이 메인 백엔드 DB에 어떻게 저장되는지 상세히 설명합니다.

---

## 1. 패치 생성 (Patch3D)

### 파일 시스템 구조
```
storage/3d/exports/Active_k2_tank_3d_k2_tankasdasd/
├── attack_pattern/
│   ├── k2_tank_bestval.png      ← 검증 세트 최고 성능 패치 (DB 저장 O)
│   └── k2_tank_besttrain.png    ← 훈련 세트 최고 성능 패치 (DB 저장 X)
└── plot/
    └── loss_curve.png
```

### DB 저장 (Patch3D 테이블)
| 필드 | 값 | 설명 |
|------|-----|------|
| `name` | `"3d_k2_tankasdasd"` | 프론트엔드가 보낸 patch_name |
| `storage_key` | `"3d/exports/Active_k2_tank_3d_k2_tankasdasd/attack_pattern/k2_tank_bestval.png"` | **bestval.png만 저장** |
| `file_name` | `"k2_tank_bestval.png"` | 파일명 |
| `method` | `"ACTIVE"` | 공격 방법 |
| `target_class` | `"K2 전차"` | 타겟 클래스 |
| `target_model_id` | `UUID` | 사용된 모델 ID |

### 왜 bestval.png만 저장하나?
- CARLA가 학습 과정에서 **두 가지 패치**를 생성:
  - `besttrain.png`: 훈련 세트에서 최고 성능
  - `bestval.png`: **검증 세트에서 최고 성능** (일반화 성능이 더 좋음)
- 실전 사용 시 `bestval.png`가 더 안정적이므로 DB에는 이것만 저장
- 파일 시스템에는 둘 다 존재하지만, DB 조회 시 `bestval.png`만 반환

---

## 2. 데이터셋 생성 (Dataset3D + Image3D)

### 파일 시스템 구조
```
storage/3d/exports/k2_tank_tmpkkkkksad/          ← Dataset3D.storage_path
├── k2_tank_eval/                                ← 하위 폴더 (평가용 이미지)
│   ├── image_loc0001_batch00001.jpg            ← Image3D 레코드
│   ├── image_loc0001_batch00002.jpg
│   ├── image_loc0002_batch00001.jpg
│   └── ...
└── annotations/
    └── annotations.json                         ← Annotation 레코드
```

### DB 저장 (Dataset3D 테이블)
| 필드 | 값 | 설명 |
|------|-----|------|
| `name` | `"k2_tank_tmpkkkkksad"` | 데이터셋 이름 |
| `storage_path` | `"/storage/3d/exports/k2_tank_tmpkkkkksad"` | **폴더 경로 (파일명 제외)** |
| `metadata_` | `{map_name: "Town03", ...}` | 생성 메타데이터 |

### DB 저장 (Image3D 테이블)
각 이미지마다 별도 레코드 생성:

| 필드 | 값 | 설명 |
|------|-----|------|
| `dataset_id` | `UUID` | Dataset3D의 ID (외래키) |
| `file_name` | `"image_loc0001_batch00001.jpg"` | 파일명만 |
| `storage_key` | `"3d/exports/k2_tank_tmpkkkkksad/k2_tank_eval/image_loc0001_batch00001.jpg"` | **전체 경로** |
| `width` | `512` | 이미지 너비 |
| `height` | `512` | 이미지 높이 |

### DB 저장 (Annotation 테이블)
각 바운딩 박스마다 별도 레코드 생성:

| 필드 | 값 | 설명 |
|------|-----|------|
| `image_3d_id` | `UUID` | Image3D의 ID (외래키) |
| `bbox_x`, `bbox_y`, `bbox_width`, `bbox_height` | `float` | 바운딩 박스 좌표 |
| `class_name` | `"car"` | 객체 클래스 |

### 왜 storage_path에 k2_tank_eval이 없나?
- **Dataset3D.storage_path**: 데이터셋의 **루트 폴더**만 저장
  - 목적: 데이터셋 전체를 대표하는 경로
  - 예: `/storage/3d/exports/k2_tank_tmpkkkkksad`
- **Image3D.storage_key**: 개별 이미지의 **전체 경로** 저장
  - 목적: 실제 파일 위치
  - 예: `3d/exports/k2_tank_tmpkkkkksad/k2_tank_eval/image.jpg`
- 프론트엔드에서 이미지 다운로드 시 `Image3D.storage_key` 사용

---

## 3. 공격 데이터셋 생성 (AttackDataset3D + Dataset3D + Image3D)

### 파일 시스템 구조
```
storage/3d/exports/adv_k2_tank/                  ← Dataset3D.storage_path
├── k2_tank_eval/                                ← 하위 폴더 (패치 적용된 이미지)
│   ├── image_loc0001_batch00001.jpg            ← Image3D 레코드
│   ├── image_loc0001_batch00002.jpg
│   └── ...
└── annotations/
    └── annotations.json                         ← Annotation 레코드
```

### DB 저장 (AttackDataset3D 테이블)
| 필드 | 값 | 설명 |
|------|-----|------|
| `name` | `"k2_tank_attack_20260109_123456"` | 자동 생성 이름 |
| `attack_type` | `"PATCH"` | 공격 타입 |
| `output_dataset_id` | `UUID` | 출력 Dataset3D의 ID (외래키) |
| `patch_id` | `NULL` | 패치 이름만 문자열로 parameters에 저장 |
| `parameters` | `{attack_method: "ACTIVE", patch_name: "3d_k2_tank", ...}` | 공격 메타데이터 |

### DB 저장 (Dataset3D 테이블 - output_dataset)
| 필드 | 값 | 설명 |
|------|-----|------|
| `name` | `"k2_tank_output_20260109_123456"` | 자동 생성 이름 |
| `storage_path` | `"/storage/3d/exports/adv_k2_tank"` | **폴더 경로** |
| `metadata_` | `{attack_method: "ACTIVE", ...}` | 생성 메타데이터 |

### DB 저장 (Image3D 테이블)
| 필드 | 값 | 설명 |
|------|-----|------|
| `dataset_id` | `UUID` | output_dataset의 ID |
| `storage_key` | `"3d/exports/adv_k2_tank/k2_tank_eval/image.jpg"` | **전체 경로** |

---

## 경로 규칙 요약

### 1. 절대 경로 vs 상대 경로
- **Dataset3D.storage_path**: 절대 경로 (`/storage/3d/...`)
- **Image3D.storage_key**: 상대 경로 (`3d/...`, `/storage/` 제거)
- **Patch3D.storage_key**: 상대 경로 (`3d/...`, `/storage/` 제거)

### 2. 폴더 vs 파일
- **Dataset3D.storage_path**: 폴더까지만 (`/storage/3d/exports/k2_tank_tmpkkkkksad`)
- **Image3D.storage_key**: 파일까지 (`3d/exports/k2_tank_tmpkkkkksad/k2_tank_eval/image.jpg`)
- **Patch3D.storage_key**: 파일까지 (`3d/exports/Active_k2_tank_3d_k2_tank/attack_pattern/k2_tank_bestval.png`)

### 3. exports 폴더 포함
- ✅ **모든 경로에 exports 폴더 포함**
- 예: `3d/exports/k2_tank_tmpkkkkksad` (O)
- 예: `3d/k2_tank_tmpkkkkksad` (X)

---

## 프론트엔드 사용 방법

### 1. 데이터셋 이미지 표시
```typescript
// 1. Dataset3D 조회
GET /api/v1/carla/datasets_3d/{dataset_id}
→ { storage_path: "/storage/3d/exports/k2_tank_tmpkkkkksad" }

// 2. Image3D 목록 조회
GET /api/v1/carla/datasets_3d/{dataset_id}/images
→ { images: [{ storage_key: "3d/exports/k2_tank_tmpkkkkksad/k2_tank_eval/image.jpg" }] }

// 3. 이미지 표시
<img src="/api/storage/{storage_key}" />
→ <img src="/api/storage/3d/exports/k2_tank_tmpkkkkksad/k2_tank_eval/image.jpg" />
```

### 2. 패치 이미지 표시
```typescript
// 1. Patch3D 조회
GET /api/v1/carla/patches_3d/{patch_id}
→ { storage_key: "3d/exports/Active_k2_tank_3d_k2_tank/attack_pattern/k2_tank_bestval.png" }

// 2. 패치 이미지 표시
<img src="/api/storage/{storage_key}" />
→ <img src="/api/storage/3d/exports/Active_k2_tank_3d_k2_tank/attack_pattern/k2_tank_bestval.png" />
```

### 3. 공격 데이터셋 이미지 표시
```typescript
// 1. AttackDataset3D 조회
GET /api/v1/carla/attack_datasets_3d/{attack_dataset_id}
→ { output_dataset_id: "xxx-xxx-xxx" }

// 2. output_dataset의 이미지 조회
GET /api/v1/carla/datasets_3d/{output_dataset_id}/images
→ { images: [{ storage_key: "3d/exports/adv_k2_tank/k2_tank_eval/image.jpg" }] }

// 3. 이미지 표시
<img src="/api/storage/3d/exports/adv_k2_tank/k2_tank_eval/image.jpg" />
```

---

## 트러블슈팅

### Q: 패치 파일이 2개 생성되는데 하나만 DB에 저장되나요?
A: 네, CARLA가 `besttrain.png`와 `bestval.png` 두 개를 생성하지만, **검증 성능이 더 좋은 `bestval.png`만 DB에 저장**됩니다. 이는 실전 사용 시 일반화 성능이 더 중요하기 때문입니다.

### Q: Dataset3D.storage_path에 왜 하위 폴더가 포함되지 않나요?
A: Dataset3D는 **데이터셋의 루트 폴더**를 나타냅니다. 하위 폴더 정보는 Image3D.storage_key에 포함됩니다. 이렇게 분리하면 데이터셋 전체 이동 시 storage_path만 변경하면 됩니다.

### Q: storage_key에 `/storage/`가 붙는 경우와 안 붙는 경우가 있나요?
A: **DB 저장 규칙**:
- Dataset3D.storage_path: `/storage/` 포함 (절대 경로)
- Image3D.storage_key: `/storage/` 제거 (상대 경로)
- Patch3D.storage_key: `/storage/` 제거 (상대 경로)

프론트엔드에서 사용 시 `/api/storage/{storage_key}` 형식으로 요청하므로, storage_key는 상대 경로여야 합니다.

---

**작성일**: 2026-01-09
**테스트 완료**: test_path_parsing.py 실행 결과 모든 경로 정상 파싱 확인
