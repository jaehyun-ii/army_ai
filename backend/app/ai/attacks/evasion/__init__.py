"""
Module providing evasion attacks.
Cleaned up version with only actively used implementations.
"""

# Noise attacks - FGSM/PGD (PyTorch implementations)
from app.ai.attacks.evasion.fast_gradient_pytorch import FastGradientMethodPyTorch
from app.ai.attacks.evasion.projected_gradient_descent_pytorch import ProjectedGradientDescentPyTorch

# Patch attacks - PyTorch implementations
from app.ai.attacks.evasion.dpatch_pytorch import DPatchPyTorch
from app.ai.attacks.evasion.dpatch_robust_pytorch import RobustDPatchPyTorch
from app.ai.attacks.evasion.adversarial_patch_pytorch import AdversarialPatchPyTorch

# Universal noise attacks - PyTorch implementations
from app.ai.attacks.evasion.universal_noise_pytorch import UniversalNoiseAttackPyTorch
from app.ai.attacks.evasion.noise_osfd_pytorch import NoiseOSFDPyTorch

# Backward compatibility aliases
UniversalNoiseAttack = UniversalNoiseAttackPyTorch
NoiseOSFD = NoiseOSFDPyTorch

__all__ = [
    "FastGradientMethodPyTorch",
    "ProjectedGradientDescentPyTorch",
    "DPatchPyTorch",
    "RobustDPatchPyTorch",
    "AdversarialPatchPyTorch",
    "UniversalNoiseAttackPyTorch",
    "NoiseOSFDPyTorch",
    "UniversalNoiseAttack",
    "NoiseOSFD",
]