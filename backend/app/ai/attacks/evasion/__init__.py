"""
Module providing evasion attacks.
Version with PGD and patch attacks for YOLO.
"""

from app.ai.attacks.evasion.fast_gradient import FastGradientMethod
from app.ai.attacks.evasion.projected_gradient_descent.projected_gradient_descent import ProjectedGradientDescent
from app.ai.attacks.evasion.projected_gradient_descent.projected_gradient_descent_numpy import (
    ProjectedGradientDescentNumpy,
)
from app.ai.attacks.evasion.projected_gradient_descent.projected_gradient_descent_pytorch import (
    ProjectedGradientDescentPyTorch,
)

# Patch attacks
from app.ai.attacks.evasion.dpatch import DPatch
from app.ai.attacks.evasion.dpatch_pytorch import DPatchPyTorch
from app.ai.attacks.evasion.dpatch_robust import RobustDPatch
from app.ai.attacks.evasion.dpatch_robust_pytorch import RobustDPatchPyTorch
from app.ai.attacks.evasion.adversarial_patch.adversarial_patch import AdversarialPatch
from app.ai.attacks.evasion.adversarial_patch.adversarial_patch_pytorch import AdversarialPatchPyTorch
from app.ai.attacks.evasion.adversarial_patch.adversarial_patch_numpy import AdversarialPatchNumpy
from app.ai.attacks.evasion.dynamic_object_patch import DynamicObjectPatchPyTorch

# Noise attacks - PyTorch versions (primary implementations with full features)
from app.ai.attacks.evasion.universal_noise_pytorch import UniversalNoiseAttackPyTorch
from app.ai.attacks.evasion.noise_osfd_pytorch import NoiseOSFDPyTorch

# Legacy: Export PyTorch versions under old names for backward compatibility
UniversalNoiseAttack = UniversalNoiseAttackPyTorch
NoiseOSFD = NoiseOSFDPyTorch