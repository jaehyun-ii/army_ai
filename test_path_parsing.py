#!/usr/bin/env python3
"""
CARLA 경로 파싱 로직 테스트 스크립트
"""

def convert_carla_path_to_storage(image_path: str) -> str:
    """CARLA 경로를 메인 백엔드 storage 경로로 변환"""
    if image_path.startswith("/static/exports/"):
        relative_path = image_path.replace("/static/exports/", "")
        storage_path = f"/storage/3d/exports/{relative_path}"
        return storage_path
    elif image_path.startswith("/exports/"):
        relative_path = image_path.replace("/exports/", "")
        storage_path = f"/storage/3d/exports/{relative_path}"
        return storage_path
    elif image_path.startswith("/workspace/exports/"):
        relative_path = image_path.replace("/workspace/exports/", "")
        storage_path = f"/storage/3d/exports/{relative_path}"
        return storage_path
    elif image_path.startswith("/storage/3d/"):
        return image_path
    else:
        return None


def extract_dataset_storage_path(image_path: str) -> str:
    """데이터셋/공격 데이터셋의 storage_path 추출"""
    if image_path and image_path.startswith("/storage/3d/"):
        path_parts = image_path.split("/")
        if len(path_parts) >= 4:
            # exports 폴더인 경우 한 단계 더 깊이 포함
            if len(path_parts) >= 5 and path_parts[3] == "exports":
                storage_path = "/".join(path_parts[:5])  # /storage/3d/exports/folder_name
            else:
                storage_path = "/".join(path_parts[:4])  # /storage/3d/folder_name
            return storage_path
    return None


def extract_patch_storage_key(image_path: str) -> str:
    """패치의 storage_key 추출 (파일 경로 전체)"""
    storage_key = image_path.replace("/storage/", "")
    return storage_key


print("=" * 80)
print("CARLA 경로 파싱 테스트")
print("=" * 80)

# 테스트 케이스 1: 패치 생성
print("\n[테스트 1] 패치 생성 (Active_k2_tank_3d_k2_tankasdasd)")
print("-" * 80)
carla_patch_path = "/static/exports/Active_k2_tank_3d_k2_tankasdasd/attack_pattern/k2_tank_bestval.png"
print(f"CARLA 경로:        {carla_patch_path}")

storage_patch_path = convert_carla_path_to_storage(carla_patch_path)
print(f"변환된 경로:       {storage_patch_path}")

storage_key = extract_patch_storage_key(storage_patch_path)
print(f"DB storage_key:    {storage_key}")
print(f"✅ 예상 경로와 일치: {'3d/exports/Active_k2_tank_3d_k2_tankasdasd/attack_pattern/k2_tank_bestval.png' == storage_key}")

# besttrain도 테스트
carla_patch_train = "/static/exports/Active_k2_tank_3d_k2_tankasdasd/attack_pattern/k2_tank_besttrain.png"
storage_patch_train = convert_carla_path_to_storage(carla_patch_train)
storage_key_train = extract_patch_storage_key(storage_patch_train)
print(f"\nbesttrain 경로:    {storage_key_train}")
print(f"💡 참고: DB에는 bestval.png만 저장되고, besttrain.png는 저장되지 않습니다.")

# 테스트 케이스 2: 데이터셋 생성
print("\n[테스트 2] 데이터셋 생성 (k2_tank_tmpkkkkksad)")
print("-" * 80)
carla_dataset_path = "/static/exports/k2_tank_tmpkkkkksad/k2_tank_eval/image_001.jpg"
print(f"CARLA 경로:        {carla_dataset_path}")

storage_dataset_path = convert_carla_path_to_storage(carla_dataset_path)
print(f"변환된 경로:       {storage_dataset_path}")

dataset_storage_path = extract_dataset_storage_path(storage_dataset_path)
print(f"DB storage_path:   {dataset_storage_path}")
print(f"✅ 예상 경로와 일치: {'/storage/3d/exports/k2_tank_tmpkkkkksad' == dataset_storage_path}")

# Image3D storage_key 추출
image_storage_key = storage_dataset_path.replace("/storage/", "")
print(f"\nImage3D storage_key: {image_storage_key}")

# 테스트 케이스 3: 공격 데이터셋 생성
print("\n[테스트 3] 공격 데이터셋 생성 (adv_k2_tank)")
print("-" * 80)
carla_attack_path = "/static/exports/adv_k2_tank/k2_tank_eval/image_001.jpg"
print(f"CARLA 경로:        {carla_attack_path}")

storage_attack_path = convert_carla_path_to_storage(carla_attack_path)
print(f"변환된 경로:       {storage_attack_path}")

attack_storage_path = extract_dataset_storage_path(storage_attack_path)
print(f"DB storage_path:   {attack_storage_path}")
print(f"✅ 예상 경로와 일치: {'/storage/3d/exports/adv_k2_tank' == attack_storage_path}")

# 테스트 케이스 4: 다양한 CARLA 경로 형식
print("\n[테스트 4] 다양한 CARLA 경로 형식")
print("-" * 80)
test_paths = [
    "/static/exports/k2_tank/image.jpg",
    "/exports/k2_tank/image.jpg",
    "/workspace/exports/k2_tank/image.jpg",
    "/storage/3d/exports/k2_tank/image.jpg",
]

for path in test_paths:
    converted = convert_carla_path_to_storage(path)
    print(f"{path:50} → {converted}")

# 테스트 케이스 5: path_parts 분석
print("\n[테스트 5] path_parts 분석")
print("-" * 80)
test_image_path = "/storage/3d/exports/k2_tank_tmpkkkkksad/k2_tank_eval/image_001.jpg"
path_parts = test_image_path.split("/")
print(f"image_path:  {test_image_path}")
print(f"path_parts:  {path_parts}")
print(f"len:         {len(path_parts)}")
print(f"path_parts[3]: '{path_parts[3]}'")
print(f"path_parts[:5]: {path_parts[:5]}")
print(f"결과:        {'/'.join(path_parts[:5])}")

print("\n" + "=" * 80)
print("테스트 완료")
print("=" * 80)
