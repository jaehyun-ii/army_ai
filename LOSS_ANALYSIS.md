# NAP Loss 문제 분석

## 🔴 관찰된 문제점

### 1. Detection Loss가 비정상적으로 높음
```
Iteration 0:   det_loss=483.3477
Iteration 10:  det_loss=460.4140
Iteration 100: det_loss=462.7949
```

**문제**:
- Detection loss가 **460-483 범위**로 매우 높음
- 정상적인 YOLO loss는 **0-10 범위**여야 함
- 100 iterations 동안 거의 감소하지 않음 (483 → 462, 약 4% 감소)

### 2. Loss가 수렴하지 않음
```
Loss 변화: 483.35 → 460.41 → 464.01 → 462.94 → 460.87 → ...
```

**문제**:
- Loss가 460-465 사이에서 진동만 함
- 명확한 감소 추세가 없음
- Gradient가 제대로 흐르지 않을 가능성

### 3. TV Loss는 정상
```
tv_loss=0.0187 (초기)
tv_loss=0.0976 (10번째 이후)
```

**정상**:
- TV loss는 0.01-0.1 범위로 정상
- tv_weight=0.0이므로 전체 loss에 영향 없음

---

## 🔍 원인 분석

### 가능한 원인 1: `compute_loss()` 반환값 문제

**가설**: RT-DETR의 `compute_loss()`가 **총 loss의 합계**가 아닌 **batch 평균이 아닌 값**을 반환

```python
# 예상: loss per image (정상)
loss = estimator.compute_loss(x, y)  # 예: 5.2

# 실제: loss × batch_size × num_boxes? (비정상)
loss = estimator.compute_loss(x, y)  # 예: 483.3
```

### 가능한 원인 2: Gradient Flow 문제

**가설**: Latent → Patch → Image 경로에서 gradient가 끊김

확인 필요:
1. `_generate_patch_from_latent()`에서 gradient 보존 여부
2. `_apply_patch_to_images()`에서 in-place 연산 여부
3. `estimator.compute_loss()` 내부에서 detach 여부

### 가능한 원인 3: Loss Scale 문제

**가설**: 이미지나 라벨의 스케일이 잘못됨

```python
# 예: 이미지가 [0, 255] 범위인데 모델은 [0, 1] 기대
# 또는 BBox 좌표가 normalized되지 않음
```

---

## 🔬 진단 방법

### 1. Loss 값 자체 확인

```python
# naturalistic_patch_pytorch.py의 generate() 메서드에 추가

# 첫 iteration에 상세 로깅
if iteration == 0:
    logger.info(f"=== First Iteration Debug ===")
    logger.info(f"Images range: [{images_tensor.min():.2f}, {images_tensor.max():.2f}]")
    logger.info(f"Patch range: [{patch.min():.2f}, {patch.max():.2f}]")
    logger.info(f"Patched images range: [{patched_images.min():.2f}, {patched_images.max():.2f}]")
    logger.info(f"Loss det: {loss_det.item():.4f}")
    logger.info(f"Loss det requires_grad: {loss_det.requires_grad}")

    # Gradient check
    if hasattr(self, 'latent_shift') and self.latent_shift.grad is not None:
        logger.info(f"Latent gradient norm: {torch.norm(self.latent_shift.grad).item():.6f}")
```

### 2. Estimator의 compute_loss() 검증

```python
# Test script to check what compute_loss returns

import torch
from backend.app.ai.estimators.object_detection.pytorch_object_detector import PyTorchObjectDetector

# Load estimator
estimator = ...

# Create dummy data
B, C, H, W = 2, 3, 640, 640
x = torch.rand(B, C, H, W, device=estimator.device)
y = [
    {"boxes": torch.tensor([[100, 100, 200, 200]], device=estimator.device),
     "labels": torch.tensor([0], device=estimator.device)},
    {"boxes": torch.tensor([[150, 150, 250, 250]], device=estimator.device),
     "labels": torch.tensor([0], device=estimator.device)},
]

# Compute loss
loss = estimator.compute_loss(x, y)
print(f"Loss value: {loss.item():.4f}")
print(f"Loss requires_grad: {loss.requires_grad}")
print(f"Loss dtype: {loss.dtype}")
print(f"Loss shape: {loss.shape}")
```

### 3. Gradient Flow 검증

```python
# After first backward pass
if iteration == 0:
    # Check if gradient flows to latent
    logger.info(f"=== Gradient Check ===")
    logger.info(f"latent_shift.requires_grad: {self.latent_shift.requires_grad}")
    logger.info(f"latent_shift.grad is not None: {self.latent_shift.grad is not None}")

    if self.latent_shift.grad is not None:
        grad_norm = torch.norm(self.latent_shift.grad).item()
        grad_max = torch.max(torch.abs(self.latent_shift.grad)).item()
        logger.info(f"Gradient norm: {grad_norm:.6f}")
        logger.info(f"Gradient max abs: {grad_max:.6f}")

        if grad_norm < 1e-10:
            logger.error("Gradient is too small! Possible gradient flow issue.")
```

---

## 🔧 해결 방안

### 방안 1: Loss Normalization

If `compute_loss()` returns unnormalized loss:

```python
# In _compute_detection_loss()
loss_det_patched = self.estimator.compute_loss(x=patched_images, y=labels)

# Normalize by batch size
batch_size = patched_images.size(0)
loss_det_patched = loss_det_patched / batch_size

# Or normalize by number of boxes
total_boxes = sum([len(label["boxes"]) for label in labels])
if total_boxes > 0:
    loss_det_patched = loss_det_patched / total_boxes
```

### 방안 2: 직접 Detection Loss 계산

RT-DETR의 `compute_loss()` 대신 직접 계산:

```python
def _compute_detection_loss_direct(
    self,
    patched_images: torch.Tensor,
    labels: list[dict],
) -> torch.Tensor:
    """
    Compute detection loss directly from model predictions.
    """
    # Get predictions
    with torch.enable_grad():
        predictions = self.estimator.model(patched_images)

    # Extract objectness scores for target class
    # (RT-DETR specific implementation needed)

    # Return mean objectness as surrogate loss
    return objectness_scores.mean()
```

### 방안 3: Learning Rate 조정

Loss scale이 크면 learning rate를 낮춰야 함:

```python
# Current: learning_rate=0.02 (from logs)
# Try: learning_rate=0.001 or 0.0001

attack = NaturalisticPatchPyTorch(
    estimator=estimator,
    learning_rate=0.001,  # 20배 감소
    # ...
)
```

### 방안 4: Loss Clipping

Extreme loss 값을 clipping:

```python
# In training loop
loss_det = self._compute_detection_loss(...)

# Clip to reasonable range
loss_det = torch.clamp(loss_det, min=0.0, max=100.0)

loss_tv = self.tv_loss_fn(patch)
loss = -loss_det + self.tv_weight * loss_tv
```

---

## 📊 예상되는 정상 값

### YOLO/RT-DETR Loss 범위
```
Total Loss:        0.5 - 10.0  (학습 초기)
                   0.1 - 2.0   (수렴 후)

Classification:    0.1 - 5.0
BBox Regression:   0.1 - 3.0
Objectness:        0.1 - 2.0
```

### NAP에서 기대되는 Loss
```
Iteration 0:   det_loss=5.0-15.0  (초기)
Iteration 50:  det_loss=2.0-8.0   (중간)
Iteration 100: det_loss=0.5-3.0   (수렴)
```

**현재 관찰값 (460-483)은 약 100배 높음!**

---

## 🚨 즉시 확인할 사항

### 1. Estimator의 compute_loss() 구현 확인

```bash
# PyTorchObjectDetector의 compute_loss 메서드 찾기
grep -n "def compute_loss" backend/app/ai/estimators/object_detection/pytorch_object_detector.py
```

### 2. RT-DETR 모델의 loss 반환 방식

RT-DETR이 반환하는 loss가:
- **Sum of losses**: 전체 배치의 loss 합계 → 이 경우 batch_size로 나눠야 함
- **Mean of losses**: 배치 평균 → 정상
- **Dict of losses**: {'total': ..., 'cls': ..., 'bbox': ...} → 'total' 추출

### 3. 이미지 범위 확인

```python
# In generate() method, before training loop
logger.info(f"Input images range: [{images_tensor.min():.2f}, {images_tensor.max():.2f}]")
logger.info(f"Estimator clip_values: {self.estimator.clip_values}")
```

---

## 🎯 권장 조치

### 단계 1: 진단 로깅 추가
위의 "진단 방법" 섹션의 로깅 코드를 추가하여 실제 원인 파악

### 단계 2: compute_loss() 검증
Estimator의 compute_loss() 메서드가 올바른 값을 반환하는지 확인

### 단계 3: 임시 해결책 적용
Loss normalization 또는 learning rate 조정으로 학습 진행 확인

### 단계 4: 근본 원인 해결
compute_loss() 구현을 수정하거나 직접 loss 계산 구현

---

## 📝 다음 단계

1. ✅ `PyTorchObjectDetector.compute_loss()` 구현 확인
2. 🔬 첫 iteration의 상세 로그 수집
3. 🔧 원인에 따라 적절한 해결책 적용
4. 📊 정상 범위(0-10)의 loss로 재학습 확인

**현재 상태**: Loss 값이 비정상적으로 높아 학습이 진행되지 않음
**목표**: Loss를 정상 범위로 낮추고 iteration마다 감소 추세 확인
