#!/bin/bash
# 3D 평가 문제 진단 스크립트
# 사용법: ./diagnose_3d_evaluation.sh [dataset_id]

# 데이터셋 ID (기본값: 로그에서 확인한 원본 데이터셋)
DATASET_ID="${1:-3f4f3e42-a2a5-4505-a9a8-f3d1e1e256d1}"
ATTACK_DATASET_ID="637a0210-ce7a-404d-8a6e-f4ecb73f2449"

echo "========================================================================"
echo "3D 데이터셋 평가 문제 진단"
echo "========================================================================"
echo "원본 데이터셋 ID: $DATASET_ID"
echo "공격 데이터셋 ID: $ATTACK_DATASET_ID"
echo ""

# PostgreSQL 연결 정보 (환경에 맞게 수정)
DB_NAME="${POSTGRES_DB:-army_ai}"
DB_USER="${POSTGRES_USER:-postgres}"
DB_HOST="${POSTGRES_HOST:-localhost}"
DB_PORT="${POSTGRES_PORT:-5432}"

# psql 명령어 기본 옵션
PSQL="psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME"

echo "=== 1. 데이터셋 기본 정보 ==="
$PSQL -c "
SELECT
    id,
    name,
    storage_path,
    metadata_->'classes' as classes,
    metadata_->'object_name' as object_name,
    metadata_->'source' as source,
    created_at
FROM datasets_3d
WHERE id IN ('$DATASET_ID', '$ATTACK_DATASET_ID')
ORDER BY created_at;
"

echo ""
echo "=== 2. 이미지 개수 확인 ==="
$PSQL -c "
SELECT
    d.name as dataset_name,
    d.id as dataset_id,
    COUNT(i.id) as image_count
FROM datasets_3d d
LEFT JOIN images_3d i ON i.dataset_id = d.id AND i.deleted_at IS NULL
WHERE d.id IN ('$DATASET_ID', '$ATTACK_DATASET_ID')
GROUP BY d.id, d.name
ORDER BY d.created_at;
"

echo ""
echo "=== 3. Annotation 개수 및 클래스 분포 ==="
$PSQL -c "
SELECT
    d.name as dataset_name,
    a.class_name,
    COUNT(a.id) as annotation_count,
    AVG(a.bbox_width * a.bbox_height) as avg_bbox_area
FROM datasets_3d d
JOIN images_3d i ON i.dataset_id = d.id AND i.deleted_at IS NULL
LEFT JOIN annotations a ON a.image_3d_id = i.id
WHERE d.id IN ('$DATASET_ID', '$ATTACK_DATASET_ID')
GROUP BY d.name, a.class_name
ORDER BY d.name, annotation_count DESC;
"

echo ""
echo "=== 4. 샘플 Annotation 상세 (각 데이터셋 첫 5개) ==="
echo "--- 원본 데이터셋 ---"
$PSQL -c "
SELECT
    i.file_name,
    a.class_name,
    a.bbox_x,
    a.bbox_y,
    a.bbox_width,
    a.bbox_height,
    CASE
        WHEN a.bbox_x <= 1.0 AND a.bbox_y <= 1.0 THEN 'normalized'
        ELSE 'absolute'
    END as format
FROM images_3d i
JOIN annotations a ON a.image_3d_id = i.id
WHERE i.dataset_id = '$DATASET_ID' AND i.deleted_at IS NULL
ORDER BY i.file_name
LIMIT 5;
"

echo ""
echo "--- 공격 데이터셋 ---"
$PSQL -c "
SELECT
    i.file_name,
    a.class_name,
    a.bbox_x,
    a.bbox_y,
    a.bbox_width,
    a.bbox_height,
    CASE
        WHEN a.bbox_x <= 1.0 AND a.bbox_y <= 1.0 THEN 'normalized'
        ELSE 'absolute'
    END as format
FROM images_3d i
JOIN annotations a ON a.image_3d_id = i.id
WHERE i.dataset_id = '$ATTACK_DATASET_ID' AND i.deleted_at IS NULL
ORDER BY i.file_name
LIMIT 5;
"

echo ""
echo "=== 5. 파일 시스템 확인 ==="
STORAGE_PATH_ORIG=$($PSQL -t -A -c "SELECT storage_path FROM datasets_3d WHERE id = '$DATASET_ID';")
STORAGE_PATH_ATTACK=$($PSQL -t -A -c "SELECT storage_path FROM datasets_3d WHERE id = '$ATTACK_DATASET_ID';")

BASE_DIR="/home/jaehyun/army_ai"

echo "원본 데이터셋 경로: $BASE_DIR$STORAGE_PATH_ORIG"
if [ -d "$BASE_DIR$STORAGE_PATH_ORIG" ]; then
    echo "✅ 디렉토리 존재"
    echo "이미지 개수: $(find "$BASE_DIR$STORAGE_PATH_ORIG" -name "*.jpg" -o -name "*.png" | wc -l)"
    echo "Annotations 폴더:"
    ls -lh "$BASE_DIR$STORAGE_PATH_ORIG/annotations/" 2>/dev/null || echo "❌ annotations 폴더 없음"
else
    echo "❌ 디렉토리 없음"
fi

echo ""
echo "공격 데이터셋 경로: $BASE_DIR$STORAGE_PATH_ATTACK"
if [ -d "$BASE_DIR$STORAGE_PATH_ATTACK" ]; then
    echo "✅ 디렉토리 존재"
    echo "이미지 개수: $(find "$BASE_DIR$STORAGE_PATH_ATTACK" -name "*.jpg" -o -name "*.png" | wc -l)"
    echo "Annotations 폴더:"
    ls -lh "$BASE_DIR$STORAGE_PATH_ATTACK/annotations/" 2>/dev/null || echo "❌ annotations 폴더 없음"
else
    echo "❌ 디렉토리 없음"
fi

echo ""
echo "=== 6. AttackDataset3D 정보 ==="
$PSQL -c "
SELECT
    ad.name,
    ad.attack_type,
    ad.target_class,
    ad.patch_id,
    ad.parameters->'patch_name' as patch_name,
    ad.parameters->'attack_method' as attack_method,
    d.name as output_dataset_name
FROM attack_datasets_3d ad
LEFT JOIN datasets_3d d ON d.id = ad.output_dataset_id
WHERE ad.output_dataset_id = '$ATTACK_DATASET_ID';
"

echo ""
echo "========================================================================"
echo "진단 완료"
echo "========================================================================"
echo ""
echo "📊 결과 해석:"
echo ""
echo "1. Annotation 개수가 0이면:"
echo "   → CARLA가 annotations.json을 생성하지 못했거나"
echo "   → 폴링 타임아웃으로 파싱이 실행되지 않았을 가능성"
echo ""
echo "2. class_name이 'tank'가 아니면:"
echo "   → target_class='tank'로 필터링되어 GT가 0개가 됨"
echo "   → 평가 시 target_class를 실제 class_name으로 변경 필요"
echo ""
echo "3. metadata의 classes가 없으면:"
echo "   → class_X 형식의 annotation을 변환할 수 없음"
echo "   → CARLA 백엔드에서 classes 정보를 metadata에 추가 필요"
echo ""
echo "다음 단계:"
echo "  - annotation이 없으면: POST /api/v1/carla/datasets_3d/{id}/parse-annotations"
echo "  - class_name 불일치: 평가 요청 시 target_class 수정"
echo "  - 상세 보고서: 3D_EVALUATION_ISSUE_REPORT.md 참고"
echo ""
