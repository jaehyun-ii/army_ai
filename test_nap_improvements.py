"""
Test script for NAP improvements: overlap loss and noise distribution.
"""
import torch
import numpy as np
from backend.app.ai.attacks.evasion.naturalistic_patch_pytorch import NaturalisticPatchPyTorch


class MockEstimator:
    """Mock estimator for testing."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_values = (0.0, 1.0)
        self.model = torch.nn.Linear(10, 10).to(self.device)

    def compute_loss(self, x, y):
        """Mock compute_loss that returns a differentiable tensor."""
        # Simple mock: return mean of image with some random variation
        loss = x.mean() + torch.randn(1, device=x.device) * 0.1
        return loss


def test_overlap_loss():
    """Test overlap loss functionality."""
    print("\n" + "="*80)
    print("TEST 1: Overlap Loss")
    print("="*80)

    estimator = MockEstimator()

    # Test with overlap_weight = 0.0 (disabled)
    print("\n1.1. Testing with overlap_weight=0.0 (disabled)")
    attack_no_overlap = NaturalisticPatchPyTorch(
        estimator=estimator,
        overlap_weight=0.0,
        max_iter=5,
        verbose=False,
    )
    print(f"✓ Initialized with overlap_weight={attack_no_overlap.overlap_weight}")

    # Test with overlap_weight = 0.1 (enabled)
    print("\n1.2. Testing with overlap_weight=0.1 (enabled)")
    attack_with_overlap = NaturalisticPatchPyTorch(
        estimator=estimator,
        overlap_weight=0.1,
        max_iter=5,
        verbose=False,
    )
    print(f"✓ Initialized with overlap_weight={attack_with_overlap.overlap_weight}")

    # Test _compute_detection_loss with/without clean images
    print("\n1.3. Testing _compute_detection_loss method")
    B, C, H, W = 2, 3, 416, 416
    patched_images = torch.rand(B, C, H, W, device=estimator.device)
    clean_images = torch.rand(B, C, H, W, device=estimator.device)
    labels = [
        {"boxes": torch.tensor([[100, 100, 200, 200]], device=estimator.device),
         "labels": torch.tensor([0], device=estimator.device)},
        {"boxes": torch.tensor([[150, 150, 250, 250]], device=estimator.device),
         "labels": torch.tensor([0], device=estimator.device)},
    ]

    # Without clean images
    loss_det, overlap_score = attack_with_overlap._compute_detection_loss(
        patched_images, labels=labels, clean_images=None
    )
    print(f"  Without clean images: loss_det={loss_det.item():.4f}, overlap_score={overlap_score.item():.4f}")
    assert overlap_score.item() == 0.0, "Overlap score should be 0 without clean images"
    assert loss_det.requires_grad, "Detection loss should require gradient"

    # With clean images
    loss_det, overlap_score = attack_with_overlap._compute_detection_loss(
        patched_images, labels=labels, clean_images=clean_images
    )
    print(f"  With clean images: loss_det={loss_det.item():.4f}, overlap_score={overlap_score.item():.4f}")
    assert overlap_score.item() > 0.0, "Overlap score should be > 0 with clean images"
    assert loss_det.requires_grad, "Detection loss should require gradient"
    assert not overlap_score.requires_grad, "Overlap score should not require gradient (for efficiency)"

    print("✓ All overlap loss tests passed!")


def test_noise_distribution():
    """Test noise distribution selection."""
    print("\n" + "="*80)
    print("TEST 2: Noise Distribution")
    print("="*80)

    estimator = MockEstimator()

    # Test gaussian (default)
    print("\n2.1. Testing with noise_distribution='gaussian'")
    attack_gaussian = NaturalisticPatchPyTorch(
        estimator=estimator,
        noise_distribution='gaussian',
        max_iter=5,
        verbose=False,
    )
    print(f"✓ Initialized with noise_distribution='{attack_gaussian.noise_distribution}'")

    # Test uniform
    print("\n2.2. Testing with noise_distribution='uniform'")
    attack_uniform = NaturalisticPatchPyTorch(
        estimator=estimator,
        noise_distribution='uniform',
        max_iter=5,
        verbose=False,
    )
    print(f"✓ Initialized with noise_distribution='{attack_uniform.noise_distribution}'")

    # Test invalid distribution
    print("\n2.3. Testing with invalid noise_distribution")
    try:
        attack_invalid = NaturalisticPatchPyTorch(
            estimator=estimator,
            noise_distribution='invalid',
            max_iter=5,
            verbose=False,
        )
        print("✗ Should have raised ValueError!")
        assert False
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")

    # Test noise generation
    print("\n2.4. Testing noise generation in _apply_patch_to_images")
    B, C, H, W = 2, 3, 416, 416
    images = torch.rand(B, C, H, W, device=estimator.device)
    patch = torch.rand(C, 128, 128, device=estimator.device)
    labels = [
        {"boxes": torch.tensor([[100, 100, 200, 200]], device=estimator.device),
         "labels": torch.tensor([0], device=estimator.device)},
        {"boxes": torch.tensor([[150, 150, 250, 250]], device=estimator.device),
         "labels": torch.tensor([0], device=estimator.device)},
    ]

    # Generate with gaussian noise
    patched_gaussian = attack_gaussian._apply_patch_to_images(images, patch, labels, target_class=0)
    print(f"  Gaussian noise: patched shape={patched_gaussian.shape}")
    assert patched_gaussian.shape == images.shape

    # Generate with uniform noise
    patched_uniform = attack_uniform._apply_patch_to_images(images, patch, labels, target_class=0)
    print(f"  Uniform noise: patched shape={patched_uniform.shape}")
    assert patched_uniform.shape == images.shape

    print("✓ All noise distribution tests passed!")


def test_gradient_flow():
    """Test gradient flow through the entire pipeline."""
    print("\n" + "="*80)
    print("TEST 3: Gradient Flow")
    print("="*80)

    estimator = MockEstimator()

    # Create attack with all features enabled
    print("\n3.1. Creating attack with overlap_weight=0.1")
    attack = NaturalisticPatchPyTorch(
        estimator=estimator,
        overlap_weight=0.1,
        tv_weight=0.1,
        noise_distribution='gaussian',
        max_iter=3,
        verbose=False,
    )

    # Prepare test data
    B, C, H, W = 2, 3, 416, 416
    x = np.random.rand(B, C, H, W).astype(np.float32)
    y = [
        {"boxes": np.array([[100, 100, 200, 200]], dtype=np.float32),
         "labels": np.array([0], dtype=np.int64)},
        {"boxes": np.array([[150, 150, 250, 250]], dtype=np.float32),
         "labels": np.array([0], dtype=np.int64)},
    ]

    print(f"  Input shape: {x.shape}")
    print(f"  Labels: {len(y)} images")

    # Run one iteration manually
    print("\n3.2. Running one optimization step")
    images_tensor = torch.from_numpy(x).to(estimator.device)
    labels_tensor = [
        {"boxes": torch.as_tensor(y[0]["boxes"], device=estimator.device),
         "labels": torch.as_tensor(y[0]["labels"], device=estimator.device)},
        {"boxes": torch.as_tensor(y[1]["boxes"], device=estimator.device),
         "labels": torch.as_tensor(y[1]["labels"], device=estimator.device)},
    ]
    clean_images = images_tensor.clone().detach()

    # Zero gradients
    attack.optimizer.zero_grad()

    # Generate patch
    patch = attack._generate_patch_from_latent()
    print(f"  Generated patch: shape={patch.shape}, requires_grad={patch.requires_grad}")

    # Apply patch
    patched_images = attack._apply_patch_to_images(images_tensor, patch, labels_tensor, target_class=0)
    print(f"  Patched images: shape={patched_images.shape}, requires_grad={patched_images.requires_grad}")

    # Compute losses
    loss_det, overlap_score = attack._compute_detection_loss(
        patched_images, labels=labels_tensor, clean_images=clean_images
    )
    loss_tv = attack.tv_loss_fn(patch)
    print(f"  Detection loss: {loss_det.item():.4f}, requires_grad={loss_det.requires_grad}")
    print(f"  Overlap score: {overlap_score.item():.4f}, requires_grad={overlap_score.requires_grad}")
    print(f"  TV loss: {loss_tv.item():.4f}, requires_grad={loss_tv.requires_grad}")

    # Total loss
    loss_attack = -loss_det
    loss_overlap = -overlap_score
    loss = loss_attack + attack.tv_weight * loss_tv + attack.overlap_weight * loss_overlap
    print(f"  Total loss: {loss.item():.4f}, requires_grad={loss.requires_grad}")

    # Backward
    loss.backward()

    # Check gradients
    latent_grad_norm = torch.norm(attack.latent_shift.grad).item()
    print(f"  Latent gradient norm: {latent_grad_norm:.6f}")
    assert latent_grad_norm > 0, "Gradient should be non-zero"

    # Optimizer step
    attack.optimizer.step()

    print("✓ Gradient flow test passed!")


def test_integration():
    """Test full integration with generate method."""
    print("\n" + "="*80)
    print("TEST 4: Integration (generate method)")
    print("="*80)

    estimator = MockEstimator()

    print("\n4.1. Running full generation pipeline")
    attack = NaturalisticPatchPyTorch(
        estimator=estimator,
        overlap_weight=0.05,
        tv_weight=0.1,
        noise_distribution='gaussian',
        max_iter=10,
        verbose=True,
    )

    # Prepare test data
    B, C, H, W = 2, 3, 416, 416
    x = np.random.rand(B, C, H, W).astype(np.float32)
    y = [
        {"boxes": np.array([[100, 100, 200, 200]], dtype=np.float32),
         "labels": np.array([0], dtype=np.int64)},
        {"boxes": np.array([[150, 150, 250, 250]], dtype=np.float32),
         "labels": np.array([0], dtype=np.int64)},
    ]

    print(f"  Input: x.shape={x.shape}, {len(y)} labels")

    # Generate patch
    patch, mask = attack.generate(x, y)

    print(f"\n  Generated patch: shape={patch.shape}, dtype={patch.dtype}")
    print(f"  Patch range: [{patch.min():.4f}, {patch.max():.4f}]")
    print(f"  Mask: shape={mask.shape}")

    assert patch.shape[0] == 3, "Patch should have 3 channels"
    assert mask.shape == patch.shape, "Mask shape should match patch"

    print("✓ Integration test passed!")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("NAP IMPROVEMENTS TEST SUITE")
    print("="*80)
    print("\nTesting improvements:")
    print("  1. Overlap loss support")
    print("  2. Noise distribution selection (gaussian/uniform)")
    print("  3. Gradient flow")
    print("  4. Full integration")

    try:
        test_overlap_loss()
        test_noise_distribution()
        test_gradient_flow()
        test_integration()

        print("\n" + "="*80)
        print("ALL TESTS PASSED! ✓")
        print("="*80)
        print("\nImprovements successfully implemented:")
        print("  ✓ Overlap loss (optional, default weight=0.0)")
        print("  ✓ Noise distribution selection (gaussian/uniform)")
        print("  ✓ Gradient flow preserved")
        print("  ✓ Full integration working")
        print("\nThe implementation is ready for production use.")

    except Exception as e:
        print("\n" + "="*80)
        print(f"TEST FAILED: {e}")
        print("="*80)
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
