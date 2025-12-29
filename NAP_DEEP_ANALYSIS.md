# NAP 마이그레이션 심층 분석 - 검증 및 개선 방안

## 📌 개요

이 문서는 NAP_MIGRATION_COMPARISON.md의 **9.2 "필요 시 추가 고려 사항"** 및 **9.3 "검증 필요 사항"**에 대한 원본 코드의 실제 구현을 분석하고, 마이그레이션에 적용하는 구체적인 방안을 제시합니다.

---

## 1. 🔍 Overlap Loss 상세 분석

### 1.1 원본 구현 (pytorchYOLOv4/demo.py:130-149)

```python
# get overlap_score
if not(clear_imgs == None):
    # resize image
    clear_imgs = F.interpolate(clear_imgs, size=self.m.width).to(self.device)

    # detections_tensor
    output_clear = self.m(clear_imgs)
    detections_tensor_xy_clear    = output_clear[0]    # xy1xy2
    detections_tensor_class_clear = output_clear[1]    # conf

    # Extract max probability for attacked class
    max_prob_obj_cls_clear = self.max_prob_extractor(detections_tensor_class_clear)

    # Count overlap: 패치 있을 때와 없을 때의 탐지 확률 차이
    output_score       = max_prob_obj_cls        # 패치 있는 이미지
    output_score_clear = max_prob_obj_cls_clear  # 패치 없는 원본 이미지
    overlap_score      = torch.abs(output_score - output_score_clear)
else:
    overlap_score = torch.tensor(0).to(self.device)
```

### 1.2 핵심 발견 사항 ⚠️

**중요**: 원본 코드의 `overlap_score`는 **BBox IoU가 아닙니다!**

실제로는:
```
overlap_score = |P(detection | with_patch) - P(detection | without_patch)|
```

즉, **패치 적용 전후의 탐지 확률 변화량**을 측정합니다.

### 1.3 목적 및 의미

```python
# Loss 계산 (ensemble_tool/model.py:321)
loss_overlap = -torch.mean(overlap_score)  # 음수를 취함

# 전체 Loss
loss = loss_det + (weight_loss_overlap * loss_overlap) + (weight_loss_tv * loss_tv)
```

**의미**:
- `overlap_score`가 **클수록** → 패치가 탐지 확률을 크게 변화시킴 → **공격 효과 큼**
- `loss_overlap = -overlap_score` → overlap_score를 **최대화**하는 방향으로 학습
- 즉, **패치가 탐지 확률을 최대한 많이 감소시키도록** 유도

### 1.4 마이그레이션 적용 방안

#### 방안 1: 원본과 동일하게 구현 (권장)

```python
# naturalistic_patch_pytorch.py의 _compute_detection_loss 수정

def _compute_detection_loss(
    self,
    patched_images: torch.Tensor,
    labels: Optional[list[dict[str, torch.Tensor]]] = None,
    target_class: Optional[int] = None,
    clean_images: Optional[torch.Tensor] = None,  # 추가
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute detection loss with optional overlap score.

    Returns:
        Tuple of (detection_loss, overlap_score)
    """
    # Detection loss on patched images
    if labels is not None:
        loss_det_patched = self.estimator.compute_loss(x=patched_images, y=labels)
    else:
        loss_det_patched = patched_images.mean()

    # Overlap score: 패치 전후 탐지 확률 차이
    overlap_score = torch.tensor(0.0, device=patched_images.device)
    if clean_images is not None and labels is not None:
        loss_det_clean = self.estimator.compute_loss(x=clean_images, y=labels)
        overlap_score = torch.abs(loss_det_patched - loss_det_clean)

    return loss_det_patched, overlap_score

# generate() 메서드 수정
def generate(self, x, y=None, **kwargs):
    # ...
    clean_images_tensor = images_tensor.clone().detach()  # 원본 저장

    for iteration in iterator:
        self.optimizer.zero_grad()

        patch = self._generate_patch_from_latent()
        patched_images = self._apply_patch_to_images(...)

        # Loss 계산
        loss_det, overlap_score = self._compute_detection_loss(
            patched_images,
            labels=labels,
            target_class=target_class,
            clean_images=clean_images_tensor,  # 전달
        )
        loss_tv = self.tv_loss_fn(patch)

        # Overlap loss 추가
        loss_attack = -loss_det
        loss_overlap = -overlap_score  # 최대화
        loss = loss_attack + self.tv_weight * loss_tv + self.overlap_weight * loss_overlap

        loss.backward()
        self.optimizer.step()
```

#### 방안 2: 실제 BBox IoU 기반 (대안)

만약 진짜 BBox IoU를 사용하고 싶다면:

```python
def compute_bbox_iou(boxes1, boxes2):
    """
    Compute IoU between two sets of boxes.

    Args:
        boxes1: [N, 4] (x1, y1, x2, y2)
        boxes2: [M, 4] (x1, y1, x2, y2)

    Returns:
        IoU matrix [N, M]
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Intersection
    x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])

    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    union = area1[:, None] + area2 - intersection

    iou = intersection / (union + 1e-6)
    return iou

# 패치 있을 때와 없을 때의 예측 BBox IoU 계산
# (구현 복잡도 높음, 성능 향상 미미할 가능성)
```

### 1.5 권장 사항

✅ **방안 1 (원본과 동일) 권장**
- 구현 단순
- 원본 논문의 의도 정확히 반영
- Gradient flow 명확

❌ **방안 2 (BBox IoU) 비권장**
- 구현 복잡
- Detector에서 BBox 추출 필요 (non-differentiable)
- 성능 향상 불확실

**파라미터 설정**:
```python
# ensemble.py:82
weight_loss_overlap = 0.0  # 기본값 0 (원본에서도 사용 안 함)
```

**결론**: Overlap loss는 원본에서도 **기본값 0**으로 비활성화되어 있으므로, 마이그레이션에서 제거한 것은 합리적 선택입니다. 필요 시 위 방안 1로 쉽게 추가 가능.

---

## 2. 🎲 Noise 분포 변경 (Uniform → Gaussian)

### 2.1 원본 구현 (adversarialYolo/load_data.py:427, 865)

```python
# Create random noise tensor
noise = torch.cuda.FloatTensor(adv_batch.size()).uniform_(-1, 1) * self.noise_factor
# self.noise_factor = 0.10

# Apply noise
adv_batch = adv_batch + noise
adv_batch = torch.clamp(adv_batch, 0.000001, 0.999999)
```

**분포**:
- Uniform[-1, 1] × 0.1 = **Uniform[-0.1, 0.1]**

### 2.2 마이그레이션 구현 (naturalistic_patch_pytorch.py:396)

```python
# Gaussian noise
noise = torch.randn_like(patch_norm) * self.noise_factor
# self.noise_factor = 0.10

patch_norm = torch.clamp(patch_norm * contrast + brightness + noise, 0.0, 1.0)
```

**분포**:
- Normal(0, 1) × 0.1 ≈ **Normal(0, 0.1)**

### 2.3 통계적 비교

| 속성 | Uniform[-0.1, 0.1] | Normal(0, 0.1) |
|------|-------------------|----------------|
| 평균 (μ) | 0 | 0 |
| 표준편차 (σ) | 0.1/√3 ≈ **0.0577** | **0.1** |
| 분산 (σ²) | 0.00333 | 0.01 |
| 범위 | [-0.1, 0.1] 고정 | 이론상 (-∞, ∞) |
| 99.7% 범위 | [-0.1, 0.1] | [-0.3, 0.3] (3σ) |
| Kurtosis | 1.8 (flat) | 3 (mesokurtic) |

**핵심 차이**:
1. **Gaussian의 표준편차가 더 큼**: 0.1 vs 0.0577 (약 **1.73배**)
2. **Extreme 값 발생**: Gaussian은 작은 확률로 ±0.2 이상의 큰 noise 발생 가능

### 2.4 시각적 비교

```python
import torch
import matplotlib.pyplot as plt

# Uniform
uniform_noise = torch.empty(10000).uniform_(-0.1, 0.1)
print(f"Uniform std: {uniform_noise.std():.4f}")  # 0.0577

# Gaussian
gaussian_noise = torch.randn(10000) * 0.1
print(f"Gaussian std: {gaussian_noise.std():.4f}")  # 0.1000

# Histogram
plt.hist(uniform_noise.numpy(), bins=50, alpha=0.5, label='Uniform')
plt.hist(gaussian_noise.numpy(), bins=50, alpha=0.5, label='Gaussian')
plt.legend()
plt.show()
```

### 2.5 영향 분석

#### 긍정적 영향 (Gaussian)
1. **자연스러운 noise**: 실제 카메라 센서 노이즈는 Gaussian 분포
2. **Regularization 효과**: Extreme 값이 가끔 발생하여 overfitting 방지
3. **Robustness 향상**: 다양한 noise 레벨에 대한 강건성

#### 부정적 영향 (Gaussian)
1. **Clipping 발생**: Extreme 값 (±0.2 이상)이 [0, 1]로 clipping되어 gradient 소실
2. **분산 증가**: Uniform보다 약 3배 큰 분산 (0.01 vs 0.00333)

### 2.6 검증 실험 설계

```python
# test_noise_distribution.py
import torch
import numpy as np
from app.ai.attacks.evasion.naturalistic_patch_pytorch import NaturalisticPatchPyTorch

def compare_noise_distributions(estimator, x, y, iterations=100):
    """
    Compare attack performance with different noise distributions.
    """
    results = {
        'uniform': [],
        'gaussian': [],
    }

    for noise_type in ['uniform', 'gaussian']:
        # Modify _apply_patch_to_images to use specific noise
        attack = NaturalisticPatchPyTorch(
            estimator=estimator,
            noise_type=noise_type,  # 새 파라미터
            max_iter=iterations,
        )

        patch, _ = attack.generate(x, y)

        # Evaluate attack success rate
        asr = evaluate_attack(estimator, x, y, patch)
        results[noise_type].append(asr)

    print(f"Uniform ASR: {np.mean(results['uniform']):.2%}")
    print(f"Gaussian ASR: {np.mean(results['gaussian']):.2%}")

    return results
```

### 2.7 권장 사항

✅ **Gaussian 유지 (현재 마이그레이션) - 권장**
- 이유:
  1. 실제 센서 노이즈와 유사 (physical realism)
  2. Regularization 효과
  3. 표준편차를 조정하여 원본과 동일하게 만들 수 있음

⚠️ **표준편차 조정 고려**:
```python
# 원본과 동일한 표준편차를 원한다면
noise_factor_adjusted = 0.1 / np.sqrt(3)  # ≈ 0.0577

# naturalistic_patch_pytorch.py
noise = torch.randn_like(patch_norm) * noise_factor_adjusted
```

❌ **Uniform으로 되돌리기 - 비권장**
- Gaussian이 더 현실적이고 일반적

**최종 권장**:
```python
# Config option 추가
self.noise_distribution = 'gaussian'  # or 'uniform'
self.noise_factor = 0.1

# In _apply_patch_to_images
if self.noise_distribution == 'gaussian':
    noise = torch.randn_like(patch_norm) * self.noise_factor
elif self.noise_distribution == 'uniform':
    noise = (torch.rand_like(patch_norm) * 2 - 1) * self.noise_factor
```

---

## 3. 🔢 Latent Rounding 상세 분석

### 3.1 원본 구현 (ensemble.py:394)

```python
# 매 배치마다 강제 적용
latent_shift_biggan.data = torch.round(latent_shift_biggan.data * 10000) * (10**-4)
```

**효과**:
- 소수점 **4자리**로 반올림
- 예: `0.123456789` → `0.1235`

### 3.2 마이그레이션 구현 (naturalistic_patch_pytorch.py:228-232)

```python
# _generate_patch_from_latent() 내부
with torch.no_grad():
    self.latent_shift.clamp_(-self.max_latent_value, self.max_latent_value)

    # 선택적 rounding
    if self.enable_latent_rounding:
        factor = 10 ** self.latent_rounding_precision  # default: 4
        self.latent_shift.copy_(torch.round(self.latent_shift * factor) / factor)
```

**차이점**:
1. ✅ **선택적 적용**: `enable_latent_rounding` 파라미터로 제어
2. ✅ **Precision 조정 가능**: `latent_rounding_precision` 파라미터
3. ✅ **Gradient 보존**: `with torch.no_grad()` 사용

### 3.3 Latent Rounding의 목적

#### 원본 논문/코드의 의도 (추정)

1. **Discretization**: Latent space를 이산화하여 탐색 공간 축소
2. **Regularization**: 과도한 최적화 방지
3. **Reproducibility**: 부동소수점 오차 제거
4. **Quantization awareness**: 실제 배포 시 양자화 대비

#### 수학적 분석

```python
# Gradient 흐름
latent_shift = torch.tensor([0.12349], requires_grad=True)
latent_rounded = torch.round(latent_shift * 10000) / 10000  # 0.1235

# Loss
patch = generator(latent_rounded)
loss = criterion(patch)
loss.backward()

# Gradient는 latent_rounded를 통과하지만,
# rounding은 non-differentiable → straight-through estimator처럼 작동
```

**효과**:
- Latent shift의 **업데이트를 0.0001 단위로 제한**
- 너무 미세한 변화 방지 → **학습 안정화**

### 3.4 검증 실험 설계

```python
# test_latent_rounding.py
import torch
import matplotlib.pyplot as plt

def test_latent_rounding_effect(estimator, x, y, max_iter=500):
    """
    Compare convergence with/without latent rounding.
    """
    results = {
        'with_rounding': {'losses': [], 'latent_norms': []},
        'without_rounding': {'losses': [], 'latent_norms': []},
    }

    for enable_rounding in [True, False]:
        attack = NaturalisticPatchPyTorch(
            estimator=estimator,
            enable_latent_rounding=enable_rounding,
            latent_rounding_precision=4,
            max_iter=max_iter,
        )

        # Hook to record intermediate values
        losses = []
        latent_norms = []

        def record_hook(module, iteration, loss, latent):
            losses.append(loss.item())
            latent_norms.append(torch.norm(latent).item())

        attack.register_forward_hook(record_hook)

        patch, _ = attack.generate(x, y)

        key = 'with_rounding' if enable_rounding else 'without_rounding'
        results[key]['losses'] = losses
        results[key]['latent_norms'] = latent_norms

    # Plot convergence
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(results['with_rounding']['losses'], label='With Rounding')
    plt.plot(results['without_rounding']['losses'], label='Without Rounding')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Convergence')

    plt.subplot(1, 2, 2)
    plt.plot(results['with_rounding']['latent_norms'], label='With Rounding')
    plt.plot(results['without_rounding']['latent_norms'], label='Without Rounding')
    plt.xlabel('Iteration')
    plt.ylabel('Latent Norm')
    plt.legend()
    plt.title('Latent Vector Norm')

    plt.tight_layout()
    plt.savefig('latent_rounding_comparison.png')

    return results

# 예상 결과:
# - With rounding: 계단식 수렴 (0.0001 단위)
# - Without rounding: 부드러운 수렴
# - 최종 성능 차이 미미할 것으로 예상
```

### 3.5 Gradient Magnitude 분석

```python
# test_gradient_magnitude.py
def analyze_gradient_magnitude(estimator, x, y, iterations=50):
    """
    Compare gradient magnitudes with/without rounding.
    """
    for enable_rounding in [True, False]:
        attack = NaturalisticPatchPyTorch(
            estimator=estimator,
            enable_latent_rounding=enable_rounding,
        )

        gradients = []

        # Simulate training loop
        for i in range(iterations):
            attack.optimizer.zero_grad()

            patch = attack._generate_patch_from_latent()
            patched_images = attack._apply_patch_to_images(x, patch, ...)
            loss = attack._compute_detection_loss(patched_images, ...)

            loss.backward()

            # Record gradient magnitude
            grad_norm = torch.norm(attack.latent_shift.grad).item()
            gradients.append(grad_norm)

            attack.optimizer.step()

        print(f"{'With' if enable_rounding else 'Without'} Rounding:")
        print(f"  Mean gradient norm: {np.mean(gradients):.6f}")
        print(f"  Std gradient norm: {np.std(gradients):.6f}")
        print(f"  Max gradient norm: {np.max(gradients):.6f}")
```

### 3.6 권장 사항

✅ **기본값: Rounding 활성화 (현재 마이그레이션)**
```python
enable_latent_rounding = True
latent_rounding_precision = 4  # 원본과 동일
```

**이유**:
1. 원본 논문의 설정 준수
2. 학습 안정화 (미세한 진동 방지)
3. Reproducibility 향상

⚠️ **실험적 비활성화 고려**:
```python
# 더 정밀한 최적화가 필요한 경우
enable_latent_rounding = False

# 또는 precision 증가
latent_rounding_precision = 6  # 소수점 6자리
```

**검증 우선순위**:
1. **수렴 속도**: Rounding이 수렴을 늦추는가?
2. **최종 성능**: ASR (Attack Success Rate) 차이
3. **Gradient magnitude**: Rounding이 gradient를 왜곡하는가?

---

## 4. 🎥 3D Projection (Perspective Transformation)

### 4.1 원본 구현 (adversarialYolo/load_data.py:706-800, 1088-1109)

```python
def get_warpR(anglex, angley, anglez, fov, w, h):
    """
    3D rotation matrix for perspective transformation.

    Args:
        anglex, angley, anglez: Rotation angles (degrees)
        fov: Field of view (degrees)
        w, h: Image dimensions

    Returns:
        Perspective transformation matrix
    """
    fov = torch.tensor(fov).float().cuda()
    w = torch.tensor(w).float().cuda()
    h = torch.tensor(h).float().cuda()

    # Camera distance
    z = torch.sqrt(w ** 2 + h ** 2) / 2 / torch.tan(deg_to_rad(fov / 2))

    # Rotation matrices
    rx = torch.tensor([
        [1, 0, 0, 0],
        [0, torch.cos(deg_to_rad(anglex)), -torch.sin(deg_to_rad(anglex)), 0],
        [0, -torch.sin(deg_to_rad(anglex)), torch.cos(deg_to_rad(anglex)), 0],
        [0, 0, 0, 1]
    ]).float().cuda()

    ry = torch.tensor([
        [torch.cos(deg_to_rad(angley)), 0, torch.sin(deg_to_rad(angley)), 0],
        [0, 1, 0, 0],
        [-torch.sin(deg_to_rad(angley)), 0, torch.cos(deg_to_rad(angley)), 0],
        [0, 0, 0, 1]
    ]).float().cuda()

    rz = torch.tensor([
        [torch.cos(deg_to_rad(anglez)), torch.sin(deg_to_rad(anglez)), 0, 0],
        [-torch.sin(deg_to_rad(anglez)), torch.cos(deg_to_rad(anglez)), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]).float().cuda()

    r = torch.matmul(torch.matmul(rx, ry), rz)

    # 4-point perspective transform
    pcenter = torch.tensor([h / 2, w / 2, 0, 0]).float().cuda()

    p1 = torch.tensor([0, 0, 0, 0]).float().cuda() - pcenter
    p2 = torch.tensor([w, 0, 0, 0]).float().cuda() - pcenter
    p3 = torch.tensor([0, h, 0, 0]).float().cuda() - pcenter
    p4 = torch.tensor([w, h, 0, 0]).float().cuda() - pcenter

    # Apply rotation
    dst1 = torch.matmul(r, p1)
    dst2 = torch.matmul(r, p2)
    dst3 = torch.matmul(r, p3)
    dst4 = torch.matmul(r, p4)

    # ... (perspective projection 계산)

    return warp_matrix

# Apply 3D projection
if with_projection:
    # Padding
    padding_border = torch.nn.ZeroPad2d(50)
    input_ = padding_border(adv_batch)

    # Random angle
    angle = np.random.randint(low=-50, high=51)  # degrees

    # Get transformation matrix
    mat = get_warpR(anglex=0, angley=angle, anglez=0, fov=42, w=width, h=height)
    mat = mat.expand(batch, -1, -1, -1)

    # Apply perspective warp (using kornia)
    adv_batch = tgm.warp_perspective(input_, mat, (input_.size()[-2], input_.size()[-1]))
```

**핵심 아이디어**:
1. **3D 회전**: Y축 회전 (좌우 시점 변화) -50° ~ 50°
2. **Perspective projection**: 3D → 2D 투영
3. **Differentiable**: kornia의 `warp_perspective` 사용

### 4.2 마이그레이션 적용 방안

#### 방안 1: kornia 사용 (원본과 유사)

```python
# requirements.txt
kornia>=0.6.0

# naturalistic_patch_pytorch.py
import kornia.geometry.transform as K

class NaturalisticPatchPyTorch(EvasionAttack):
    def __init__(
        self,
        # ... (기존 파라미터)
        enable_3d_projection: bool = False,
        projection_angle_range: float = 50.0,  # degrees
        projection_fov: float = 42.0,
        **kwargs
    ):
        # ...
        self.enable_3d_projection = enable_3d_projection
        self.projection_angle_range = projection_angle_range
        self.projection_fov = projection_fov

    def _get_3d_rotation_matrix(
        self,
        angle_y: float,
        fov: float,
        width: int,
        height: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Compute 3D rotation matrix for perspective transformation.

        Args:
            angle_y: Y-axis rotation angle (degrees)
            fov: Field of view (degrees)
            width, height: Image dimensions
            device: Torch device

        Returns:
            Perspective transformation matrix [1, 3, 3]
        """
        import math

        # Convert to radians
        angle_rad = math.radians(angle_y)
        fov_rad = math.radians(fov)

        # Camera distance
        z = math.sqrt(width**2 + height**2) / 2 / math.tan(fov_rad / 2)

        # Y-axis rotation matrix
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        ry = torch.tensor([
            [cos_a, 0, sin_a, 0],
            [0, 1, 0, 0],
            [-sin_a, 0, cos_a, 0],
            [0, 0, 0, 1]
        ], dtype=torch.float32, device=device)

        # Center point
        pcenter = torch.tensor([height / 2, width / 2, 0, 0], device=device)

        # Corner points
        corners = torch.tensor([
            [0, 0, 0, 0],
            [width, 0, 0, 0],
            [0, height, 0, 0],
            [width, height, 0, 0]
        ], dtype=torch.float32, device=device)

        # Apply rotation
        corners_centered = corners - pcenter
        corners_rotated = torch.matmul(ry, corners_centered.T).T

        # Perspective projection
        corners_projected = corners_rotated[:, :2] / (corners_rotated[:, 2:3] / z + 1)
        corners_projected += pcenter[:2]

        # Compute perspective transform matrix
        src_points = torch.tensor([
            [0, 0],
            [width, 0],
            [0, height],
            [width, height]
        ], dtype=torch.float32, device=device)

        # Use kornia to get perspective matrix
        M = K.get_perspective_transform(
            src_points.unsqueeze(0),
            corners_projected.unsqueeze(0)
        )

        return M

    def _apply_patch_to_images(self, images, patch, labels, target_class):
        # ... (기존 코드)

        # 3D Projection (선택적)
        if self.enable_3d_projection:
            # Random angle
            angle_y = torch.empty(1, device=patch_resized.device).uniform_(
                -self.projection_angle_range,
                self.projection_angle_range
            ).item()

            # Get transformation matrix
            H, W = patch_resized.shape[1:]
            M = self._get_3d_rotation_matrix(
                angle_y=angle_y,
                fov=self.projection_fov,
                width=W,
                height=H,
                device=patch_resized.device,
            )

            # Apply perspective warp
            patch_resized = K.warp_perspective(
                patch_resized.unsqueeze(0),
                M,
                dsize=(H, W),
                mode='bilinear',
                padding_mode='zeros',
            )[0]

        # ... (나머지 코드)
```

#### 방안 2: 순수 PyTorch (kornia 없이)

```python
def _apply_perspective_transform_pytorch(
    self,
    image: torch.Tensor,
    angle_y: float,
    fov: float = 42.0,
) -> torch.Tensor:
    """
    Apply perspective transformation using pure PyTorch.

    Args:
        image: [C, H, W]
        angle_y: Y-axis rotation (degrees)
        fov: Field of view (degrees)

    Returns:
        Transformed image [C, H, W]
    """
    import math

    C, H, W = image.shape

    # ... (3D rotation matrix 계산 - 위와 동일)

    # Grid sampling으로 perspective warp 구현
    # (복잡하지만 kornia 의존성 제거 가능)

    return transformed_image
```

### 4.3 검증 실험 설계

```python
# test_3d_projection.py
def test_3d_projection_robustness(estimator, x, y):
    """
    Test if 3D projection improves physical world robustness.
    """
    results = {
        'without_projection': [],
        'with_projection': [],
    }

    for enable_projection in [False, True]:
        attack = NaturalisticPatchPyTorch(
            estimator=estimator,
            enable_3d_projection=enable_projection,
            max_iter=200,
        )

        patch, _ = attack.generate(x, y)

        # Evaluate on multiple viewing angles
        angles = [-30, -20, -10, 0, 10, 20, 30]
        asrs = []

        for angle in angles:
            # Apply patch with specific angle
            patched = apply_patch_with_angle(x, patch, angle)
            asr = evaluate_attack(estimator, patched, y)
            asrs.append(asr)

        key = 'with_projection' if enable_projection else 'without_projection'
        results[key] = asrs

    # Plot
    plt.plot(angles, results['without_projection'], label='Without Projection')
    plt.plot(angles, results['with_projection'], label='With Projection')
    plt.xlabel('Viewing Angle (degrees)')
    plt.ylabel('Attack Success Rate')
    plt.legend()
    plt.savefig('3d_projection_robustness.png')

    return results
```

### 4.4 권장 사항

✅ **Physical world 테스트 시 추가**:
- 인쇄된 패치를 다양한 각도에서 촬영하는 경우
- 실제 감시 카메라 (다양한 시점)

❌ **디지털 공격에는 불필요**:
- 정면 이미지만 공격하는 경우
- Computational cost 증가

**구현 우선순위**: 낮음 (필요 시에만)

---

## 5. 🌫️ Median Blur

### 5.1 원본 구현 (adversarialYolo/load_data.py:656, 1050-1060)

```python
# __init__
self.medianpooler = MedianPool2d(7, same=True)

# forward
if enable_blurred:
    adv_batch_t = self.medianpooler(adv_batch_t)
```

**목적**:
- 카메라 모션 블러 시뮬레이션
- 초점 흐림 (out-of-focus) 대응

### 5.2 마이그레이션 적용 방안 (간단한 대안)

```python
# MedianPool2d 대신 AveragePool2d 사용 (PyTorch 내장)
class NaturalisticPatchPyTorch(EvasionAttack):
    def __init__(
        self,
        # ...
        enable_blur: bool = False,
        blur_kernel_size: int = 3,
        **kwargs
    ):
        # ...
        self.enable_blur = enable_blur
        self.blur_kernel_size = blur_kernel_size

    def _apply_patch_to_images(self, images, patch, labels, target_class):
        # ...

        # Blur (선택적)
        if self.enable_blur:
            # Simple average blur (median보다 빠름)
            patch_resized = F.avg_pool2d(
                patch_resized,
                kernel_size=self.blur_kernel_size,
                stride=1,
                padding=self.blur_kernel_size // 2,
            )

        # ...
```

### 5.3 권장 사항

⚠️ **구현 필요성 낮음**:
- 현대 카메라는 autofocus로 blur가 적음
- 기본 augmentation (noise, contrast)만으로도 충분한 robustness

✅ **필요 시 간단한 대안**:
- MedianPool2d → AveragePool2d (PyTorch 내장)
- 또는 Gaussian blur (`kornia.filters.gaussian_blur2d`)

---

## 6. 📊 종합 권장 사항 및 우선순위

### 6.1 즉시 적용 가능

1. ✅ **Overlap Loss 추가** (선택적)
   - 난이도: ⭐⭐
   - 효과: ⭐⭐⭐ (다중 객체 공격 시)
   - 구현: 섹션 1.4 방안 1 참고

2. ✅ **Noise 분포 선택 옵션**
   - 난이도: ⭐
   - 효과: ⭐⭐
   - 구현: 섹션 2.7 참고
   ```python
   noise_distribution='gaussian'  # or 'uniform'
   ```

### 6.2 검증 후 결정

3. 🔬 **Latent Rounding 효과 검증**
   - 우선순위: 중
   - 실험: 섹션 3.4, 3.5 참고
   - 현재 기본값 유지 (rounding 활성화)

4. 🔬 **Noise 분포 성능 비교**
   - 우선순위: 중
   - 실험: 섹션 2.6 참고
   - 현재 Gaussian 유지

### 6.3 Physical World 전용

5. ⏸️ **3D Projection** (보류)
   - 난이도: ⭐⭐⭐⭐
   - 효과: ⭐⭐⭐⭐ (physical world)
   - 필요 시: 섹션 4.2 방안 1 참고

6. ⏸️ **Median Blur** (보류)
   - 난이도: ⭐⭐
   - 효과: ⭐⭐
   - 간단한 대안: 섹션 5.2 참고

### 6.4 구현 로드맵

#### Phase 1: 즉시 적용 (1주)
```python
# 추가할 파라미터
class NaturalisticPatchPyTorch:
    def __init__(
        self,
        # ...
        # New parameters
        overlap_weight: float = 0.0,          # Overlap loss weight
        noise_distribution: str = 'gaussian',  # 'gaussian' or 'uniform'
    ):
        pass
```

#### Phase 2: 검증 실험 (2주)
- Latent rounding 효과 측정
- Noise 분포 비교 실험
- Detection loss 계산 방식 비교

#### Phase 3: Physical World (필요 시)
- 3D projection 구현
- 실제 인쇄 테스트
- Blur augmentation 추가

---

## 7. 🔬 실험 프로토콜

### 7.1 기본 설정

```python
# 모든 실험에 공통 설정
COMMON_CONFIG = {
    'gan_class_id': 259,  # Pomeranian
    'tv_weight': 0.1,
    'alpha_latent': 1.0,
    'patch_scale': 0.2,
    'learning_rate': 0.02,
    'max_iter': 500,
    'batch_size': 8,
}

# Dataset
dataset = 'INRIA'  # 또는 'COCO'
num_images = 100   # 테스트 이미지 수
```

### 7.2 평가 메트릭

```python
def evaluate_attack_performance(estimator, x, y, patch):
    """
    Comprehensive evaluation metrics.
    """
    metrics = {}

    # 1. Attack Success Rate (ASR)
    patched = apply_patch(x, patch)
    predictions = estimator.predict(patched)
    asr = calculate_asr(predictions, y, target_class=0)
    metrics['asr'] = asr

    # 2. Detection Confidence Drop
    conf_clean = get_detection_confidence(estimator, x, y)
    conf_patched = get_detection_confidence(estimator, patched, y)
    metrics['conf_drop'] = conf_clean - conf_patched

    # 3. Patch Quality
    metrics['tv_score'] = total_variation(patch)
    metrics['patch_norm'] = torch.norm(patch).item()

    # 4. Convergence Speed
    metrics['iterations_to_90_asr'] = ...

    return metrics
```

### 7.3 통계적 유의성 검증

```python
from scipy import stats

def compare_methods(method_a_results, method_b_results):
    """
    Statistical significance test.
    """
    # T-test
    t_stat, p_value = stats.ttest_ind(method_a_results, method_b_results)

    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.4f}")

    if p_value < 0.05:
        print("Statistically significant difference (p < 0.05)")
    else:
        print("No significant difference")

    # Effect size (Cohen's d)
    cohens_d = (np.mean(method_a_results) - np.mean(method_b_results)) / \
               np.sqrt((np.std(method_a_results)**2 + np.std(method_b_results)**2) / 2)
    print(f"Cohen's d: {cohens_d:.4f}")
```

---

## 8. 📝 결론

### 현재 마이그레이션 상태: ✅ 우수

1. **핵심 기능 완벽 보존**: BigGAN latent optimization
2. **합리적 단순화**: 불필요한 복잡도 제거
3. **개선된 구조**: ART 통합, 명확한 gradient flow

### 검증 필요 항목 우선순위

| 우선순위 | 항목 | 예상 영향 | 구현 난이도 |
|---------|------|----------|-----------|
| **높음** | Overlap loss 추가 | ⭐⭐⭐ | ⭐⭐ |
| **중간** | Noise 분포 검증 | ⭐⭐ | ⭐ |
| **중간** | Latent rounding 효과 | ⭐⭐ | 🔬 실험 필요 |
| **낮음** | 3D Projection | ⭐⭐⭐⭐ (physical) | ⭐⭐⭐⭐ |
| **낮음** | Median Blur | ⭐⭐ | ⭐⭐ |

### 최종 권장사항

✅ **현재 구현 유지 + 선택적 기능 추가**
- Overlap loss: 파라미터로 제공 (기본값 0)
- Noise 분포: 설정 가능하게 (기본값 Gaussian)
- Latent rounding: 현재 설정 유지
- 3D Projection, Blur: 필요 시에만 추가

🔬 **검증 실험 수행**
- 각 항목의 실제 성능 영향 측정
- 통계적 유의성 검증
- Physical world 테스트 (선택적)

이 접근법은 **유연성과 단순성의 균형**을 유지하면서, 필요에 따라 고급 기능을 활성화할 수 있는 최선의 방안입니다.
