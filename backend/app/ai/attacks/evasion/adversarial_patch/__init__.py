"""
Adversarial Patch attacks.
"""
from app.ai.attacks.evasion.adversarial_patch.adversarial_patch_pytorch import AdversarialPatchPyTorch
from app.ai.attacks.evasion.adversarial_patch.adversarial_patch import AdversarialPatch

__all__ = [
    'AdversarialPatchPyTorch',
    'AdversarialPatch',
]
