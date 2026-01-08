#!/usr/bin/env python3
from ultralytics import YOLO, RTDETR
from pathlib import Path
import yaml

def extract_and_create_config(pt_path, model_name, model_loader=YOLO):
    print(f"\n{'='*80}")
    print(f"Processing: {model_name}")
    print(f"File: {pt_path}")
    print('='*80)

    try:
        model = model_loader(str(pt_path))
        config = {}

        if hasattr(model, 'names') and model.names:
            config['names'] = model.names
            print(f"\n✓ Class names: {model.names}")
            print(f"✓ Number of classes: {len(model.names)}")

        if hasattr(model, 'model') and hasattr(model.model, 'nc'):
            nc = model.model.nc
            config['nc'] = nc
            print(f"✓ nc: {nc}")

        if hasattr(model, 'overrides') and 'imgsz' in model.overrides:
            imgsz = model.overrides['imgsz']
            config['imgsz'] = imgsz
            print(f"✓ imgsz: {imgsz}")

        if hasattr(model, 'task'):
            config['task'] = model.task
            print(f"✓ task: {model.task}")

        config_path = pt_path.parent / "config.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        print(f"\n✅ Created: {config_path}")
        print(f"\nConfig contents:")
        print("-" * 60)
        print(yaml.dump(config, default_flow_style=False, allow_unicode=True, sort_keys=False))
        print("-" * 60)

        return config

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

base_dir = Path("/home/jaehyun/army_ai/test/test_model")

models = [
    (base_dir / "rt_detr_tank" / "best.pt", "RT-DETR", RTDETR),
    (base_dir / "yolov11_tank" / "best.pt", "YOLOv11", YOLO),
    (base_dir / "yolov8_tank" / "best.pt", "YOLOv8", YOLO),
]

print("="*80)
print("Extracting model information")
print("="*80)

results = {}
for pt_path, model_name, loader in models:
    if pt_path.exists():
        config = extract_and_create_config(pt_path, model_name, loader)
        results[model_name] = config
    else:
        print(f"\n❌ File not found: {pt_path}")
        results[model_name] = None

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
for model_name, config in results.items():
    print(f"\n{model_name}:")
    if config:
        print(f"  ✓ nc: {config.get('nc', 'N/A')}")
        print(f"  ✓ imgsz: {config.get('imgsz', 'N/A')}")
        print(f"  ✓ task: {config.get('task', 'N/A')}")
        print(f"  ✓ classes: {len(config.get('names', []))} classes")
    else:
        print("  ✗ Failed")

print("\n" + "="*80)
print("Done!")
print("="*80)
