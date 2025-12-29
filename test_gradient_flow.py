#!/usr/bin/env python3
"""
Test script to verify gradient flow in NaturalisticPatchPyTorch.

This script checks if gradients properly flow from loss to latent_shift
through the BigGAN generator.

Usage:
    python test_gradient_flow.py
"""
import sys
import torch
import numpy as np
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

from app.ai.attacks.evasion import NaturalisticPatchPyTorch
from app.ai.estimators.object_detection import PyTorchYolo


def create_dummy_estimator():
    """Create a minimal dummy estimator for testing."""
    from ultralytics import YOLO

    # Use a small YOLO model for testing
    model_path = "yolov8n.pt"  # Nano model (smallest)

    print(f"Loading YOLO model: {model_path}")
    yolo_model = YOLO(model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    yolo_model.model.to(device)

    estimator = PyTorchYolo(
        model=yolo_model.model,
        input_shape=(3, 640, 640),
        channels_first=True,
        clip_values=(0, 255),
        attack_losses=("loss_total", "loss_cls", "loss_box", "loss_dfl"),
        device_type=device,
        is_ultralytics=True,
        model_name="yolov8",
    )

    return estimator


def test_gradient_flow():
    """Test gradient flow through NaturalisticPatchPyTorch."""
    print("=" * 80)
    print("Gradient Flow Test for NaturalisticPatchPyTorch")
    print("=" * 80)

    try:
        # Create dummy estimator
        estimator = create_dummy_estimator()
        print("\n✓ Estimator created")

        # Create attack instance with minimal iterations
        print("\nCreating NaturalisticPatchPyTorch...")
        attack = NaturalisticPatchPyTorch(
            estimator=estimator,
            gan_class_id=259,  # Pomeranian
            tv_weight=0.0,
            alpha_latent=1.0,
            patch_scale=0.2,
            learning_rate=0.02,
            max_iter=2,  # Just 2 iterations for testing
            batch_size=2,
            max_latent_value=8.0,
            enable_deformator=False,
            summary_writer=False,
            verbose=True,
        )
        print("✓ Attack instance created")

        # Save initial latent shift
        initial_latent = attack.latent_shift.clone().detach()
        print(f"\nInitial latent_shift norm: {initial_latent.norm().item():.4f}")

        # Test 1: Check if latent_shift requires gradient
        print("\n" + "-" * 80)
        print("Test 1: Check latent_shift.requires_grad")
        print("-" * 80)
        assert attack.latent_shift.requires_grad, "❌ latent_shift.requires_grad is False!"
        print("✓ latent_shift.requires_grad = True")

        # Test 2: Generate patch and check gradient flow
        print("\n" + "-" * 80)
        print("Test 2: Generate patch and verify gradient computation")
        print("-" * 80)

        attack.optimizer.zero_grad()
        patch = attack._generate_patch_from_latent()

        # Create dummy loss
        dummy_loss = patch.mean()
        print(f"Dummy loss: {dummy_loss.item():.4f}")

        # Backward pass
        dummy_loss.backward()

        # Check if gradient exists
        if attack.latent_shift.grad is None:
            print("❌ FAILED: No gradient on latent_shift!")
            print("   This means gradient flow is still broken.")
            return False
        else:
            grad_norm = attack.latent_shift.grad.norm().item()
            print(f"✓ Gradient computed successfully!")
            print(f"  Gradient norm: {grad_norm:.6f}")

            if grad_norm == 0:
                print("⚠️  WARNING: Gradient norm is zero (might indicate dead neurons)")
            else:
                print("✓ Gradient norm is non-zero (healthy)")

        # Test 3: Run one optimizer step
        print("\n" + "-" * 80)
        print("Test 3: Optimizer step and check latent_shift update")
        print("-" * 80)

        attack.optimizer.step()

        # Check if latent changed
        latent_change = (attack.latent_shift - initial_latent).norm().item()
        print(f"Latent change after one step: {latent_change:.6f}")

        if latent_change == 0:
            print("❌ FAILED: latent_shift did not change!")
            print("   Optimizer step had no effect.")
            return False
        else:
            print(f"✓ latent_shift updated successfully!")

        # Test 4: Run actual training for 2 iterations
        print("\n" + "-" * 80)
        print("Test 4: Run 2 training iterations")
        print("-" * 80)

        # Create dummy training data
        x_train = np.random.randint(0, 255, (2, 640, 640, 3), dtype=np.uint8)
        y_train = None

        print("Generating adversarial patch...")
        patch_np, mask_np = attack.generate(x=x_train, y=y_train)

        print(f"✓ Patch generated successfully!")
        print(f"  Patch shape: {patch_np.shape}")
        print(f"  Patch dtype: {patch_np.dtype}")
        print(f"  Patch value range: [{patch_np.min()}, {patch_np.max()}]")

        # Check final latent change
        final_latent_change = (attack.latent_shift - initial_latent).norm().item()
        print(f"\nFinal latent_shift change: {final_latent_change:.6f}")

        if final_latent_change == 0:
            print("❌ FAILED: latent_shift did not change during training!")
            return False
        else:
            print("✓ latent_shift evolved during training!")

        # All tests passed
        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED!")
        print("=" * 80)
        print("\nGradient flow is working correctly. The fix is successful!")
        return True

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_gradient_flow()
    sys.exit(0 if success else 1)
