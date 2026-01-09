#!/usr/bin/env python3
"""
이미지 스캔 로직 테스트 스크립트
k2_tank_eval 하위 폴더의 이미지가 제대로 스캔되는지 확인
"""
import os
import tempfile
from pathlib import Path
from PIL import Image
import io

# 테스트용 가상 파일 시스템 생성
def create_test_dataset(base_dir: Path):
    """테스트용 데이터셋 폴더 구조 생성"""
    # k2_tank_eval 폴더 생성
    eval_dir = base_dir / "k2_tank_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # annotations 폴더 생성
    annotations_dir = base_dir / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)

    # 테스트 이미지 파일 생성
    image_files = []
    for i in range(1, 6):
        # k2_tank_eval 폴더 안에 이미지 생성
        img_path = eval_dir / f"image_loc000{i}_batch00001.jpg"

        # 간단한 테스트 이미지 생성 (100x100 빨간색)
        img = Image.new('RGB', (100, 100), color='red')
        img.save(img_path, 'JPEG')

        image_files.append(img_path)

    # annotations.json 생성
    annotations_file = annotations_dir / "annotations.json"
    annotations_file.write_text('{"annotations": []}')

    return image_files


def test_image_scanning(storage_root: Path, dataset_name: str):
    """이미지 스캔 로직 테스트"""
    print("=" * 80)
    print("이미지 스캔 로직 테스트")
    print("=" * 80)

    # Dataset3D.storage_path 시뮬레이션
    storage_path = f"/storage/3d/exports/{dataset_name}"
    print(f"\nDataset3D.storage_path: {storage_path}")

    # storage_key_base 계산
    storage_key_base = storage_path.replace("/storage/", "")
    print(f"storage_key_base:       {storage_key_base}")

    # base_dir 계산
    base_dir = storage_root / storage_key_base
    print(f"base_dir:               {base_dir}")
    print(f"base_dir exists:        {base_dir.exists()}")

    # 이미지 파일 스캔 (실제 코드와 동일한 로직)
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
    image_records = []

    print("\n" + "-" * 80)
    print("스캔된 이미지 파일:")
    print("-" * 80)

    for file_path in base_dir.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in image_exts:
            continue

        # storage_key 계산 (실제 코드와 동일)
        storage_key = str(file_path.relative_to(storage_root))

        # file_name 추출
        file_name = file_path.name

        # 이미지 크기 읽기
        try:
            with Image.open(file_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"❌ 이미지 읽기 실패: {file_path.name} - {e}")
            continue

        image_records.append({
            'file_name': file_name,
            'storage_key': storage_key,
            'width': width,
            'height': height,
            'full_path': str(file_path)
        })

        print(f"✅ {file_name}")
        print(f"   storage_key: {storage_key}")
        print(f"   size: {width}x{height}")
        print(f"   full_path: {file_path}")
        print()

    # 결과 요약
    print("-" * 80)
    print(f"총 {len(image_records)}개의 이미지 파일 발견")
    print("-" * 80)

    # storage_key 검증
    print("\n" + "=" * 80)
    print("storage_key 검증")
    print("=" * 80)

    for record in image_records:
        expected_pattern = f"3d/exports/{dataset_name}/k2_tank_eval/"
        actual_key = record['storage_key']

        if expected_pattern in actual_key:
            print(f"✅ {record['file_name']}")
            print(f"   예상: {expected_pattern}xxx.jpg")
            print(f"   실제: {actual_key}")
        else:
            print(f"❌ {record['file_name']}")
            print(f"   예상: {expected_pattern}xxx.jpg")
            print(f"   실제: {actual_key}")
            print(f"   ⚠️  경로에 k2_tank_eval이 누락되었습니다!")

    # annotations 디렉토리 확인
    print("\n" + "=" * 80)
    print("annotations 디렉토리 확인")
    print("=" * 80)

    annotations_dir = base_dir / "annotations"
    print(f"annotations_dir: {annotations_dir}")
    print(f"exists:          {annotations_dir.exists()}")

    if annotations_dir.exists():
        json_files = list(annotations_dir.glob("*.json"))
        print(f"JSON 파일 개수:  {len(json_files)}")
        for json_file in json_files:
            print(f"  - {json_file.name}")

    return image_records


# 메인 테스트 실행
if __name__ == "__main__":
    # 임시 디렉토리 생성 (테스트용)
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_root = Path(tmpdir)
        dataset_name = "k2_tank_tmpkkkkksad"

        # 테스트 데이터셋 생성
        dataset_path = storage_root / "3d" / "exports" / dataset_name
        print(f"테스트 데이터셋 생성 중: {dataset_path}")
        create_test_dataset(dataset_path)

        # 생성된 파일 구조 출력
        print("\n생성된 파일 구조:")
        print("-" * 80)
        for root, dirs, files in os.walk(dataset_path):
            level = root.replace(str(dataset_path), '').count(os.sep)
            indent = ' ' * 2 * level
            print(f'{indent}{os.path.basename(root)}/')
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f'{subindent}{file}')
        print()

        # 이미지 스캔 테스트
        records = test_image_scanning(storage_root, dataset_name)

        # 최종 결과
        print("\n" + "=" * 80)
        print("최종 결과")
        print("=" * 80)

        if len(records) == 5:
            print("✅ 모든 이미지 파일이 정상적으로 스캔되었습니다!")
        else:
            print(f"❌ 예상: 5개, 실제: {len(records)}개")

        # k2_tank_eval 경로 포함 확인
        all_have_eval = all("k2_tank_eval" in r['storage_key'] for r in records)
        if all_have_eval:
            print("✅ 모든 storage_key에 k2_tank_eval 폴더가 포함되었습니다!")
        else:
            print("❌ 일부 storage_key에 k2_tank_eval 폴더가 누락되었습니다!")
