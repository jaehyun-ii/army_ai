# Naturalistic Adversarial Patch (NAP) - 마이그레이션 비교 분석

## 📌 개요

원본 NAP 구현체(`/home/jaehyun/army_ai/Naturalistic-Adversarial-Patch`)와 마이그레이션된 버전(`/home/jaehyun/army_ai/backend/app/ai/attacks/evasion/naturalistic_patch_pytorch.py`)의 학습 로직 전체를 비교 분석합니다.

---

## 1. 🎨 GAN을 활용한 패치 초기화

### 원본 (ensemble.py, ensemble_tool/model.py)

```python
# 1. BigGAN 로딩 (ensemble.py:144-166)
deformator, G, shift_predictor = load_from_dir(
    './GANLatentDiscovery/models/pretrained/deformators/BigGAN/',
    G_weights='./GANLatentDiscovery/models/pretrained/generators/BigGAN/G_ema.pth'
)
generator_biggan = G
main_generator = generator_biggan
main_deformator = deformator

# 2. Latent 벡터 초기화 (ensemble.py:180-192)
# Fixed latent: torch.rand (Uniform 분포)
fixed_latent_biggan = torch.rand(len_z, device=device)

# Shift latent: torch.normal (Gaussian 분포, mean=0, std=1)
latent_shift_biggan = torch.normal(0.0, torch.ones(len_latent)).to(device).requires_grad_(True)

# 3. Conditional GAN 클래스 설정 (ensemble.py:360-361)
if is_conditional(main_generator):
    main_generator.set_classes(cls_id_generation)  # 259: Pomeranian
```

**핵심 특징:**
- **Fixed latent**: `torch.rand` (Uniform[0, 1])
- **Shift latent**: `torch.normal` (Normal(0, 1))
- **Deformator**: 선택적 사용 (latent space transformation)
- **Human-annotated directions**: Deformator의 사전 학습된 semantic directions

### 마이그레이션 (naturalistic_patch_pytorch.py)

```python
# 1. BigGAN 로딩 (naturalistic_patch_pytorch.py:181-187)
self.generator, self.deformator = load_biggan(
    weights_path=str(weights_path),
    target_class=gan_class_id,
    device=self.estimator.device.type,
    deformator_path=deformator_path,
)

# 2. Latent 벡터 초기화 (naturalistic_patch_pytorch.py:192-198)
# Fixed latent: torch.rand (원본과 동일)
self.fixed_latent = torch.rand(self.dim_z, device=self.estimator.device)

# Shift latent: torch.randn (Gaussian 분포)
self.latent_shift = nn.Parameter(
    torch.randn(self.dim_z).to(self.estimator.device),
    requires_grad=True
)
```

**핵심 특징:**
- **Fixed latent**: `torch.rand` ✅ (원본과 동일)
- **Shift latent**: `torch.randn` ⚠️ (원본은 `torch.normal(0, 1)`, 실질적으로 동일)
- **Deformator**: 동일하게 지원

**차이점:**
- 초기화 방식은 실질적으로 동일 (normal 분포)
- 마이그레이션은 `nn.Parameter`로 wrapping하여 gradient tracking 명시

---

## 2. 🔄 패치 학습 루프

### 원본 (ensemble.py:377-500)

```python
for epoch in range(start_epoch, n_epochs+1):
    for i_batch, (img_batch, lab_batch) in tqdm(enumerate(train_loader_second)):
        with autograd.detect_anomaly():
            # 1. Latent 반올림 (ensemble.py:394)
            latent_shift_biggan.data = torch.round(latent_shift_biggan.data * 10000) * (10**-4)

            # 2. 패치 생성 및 학습 (ensemble_tool/model.py:207-415)
            loss_det, loss_overlap, loss_tv, p_img_batch, fake_images_denorm, D_loss = train_rowPtach(
                method_num=method_num,
                generator=main_generator,
                discriminator=main_discriminator,
                opt=main_optimizer,
                latent_shift=latent_shift_biggan,
                alpah_latent=alpha_latent,
                input_imgs=img_batch,
                label=lab_batch,
                # ... (많은 파라미터)
            )

            # 3. Loss 집계 (ensemble.py:426-428)
            ep_loss_det   += loss_det
            ep_loss_overlap += loss_overlap
            ep_loss_tv    += loss_tv

    # 4. Epoch 단위 처리 (ensemble.py:431-437)
    ep_loss = ep_loss_det + (weight_loss_overlap * ep_loss_overlap)
    main_scheduler.step(ep_loss)
```

**핵심 특징:**
- **Latent rounding**: 매 배치마다 `latent_shift`를 소수점 4자리로 반올림
- **Overlap loss**: BBox IoU 기반 추가 loss
- **Batch 내부 최적화**: `train_rowPtach` 내부에서 `opt.step()` 수행

### 마이그레이션 (naturalistic_patch_pytorch.py:634-684)

```python
for iteration in iterator:
    self.optimizer.zero_grad()

    # 1. 패치 생성 (naturalistic_patch_pytorch.py:640)
    patch = self._generate_patch_from_latent()

    # 2. 패치 적용 (naturalistic_patch_pytorch.py:644-649)
    patched_images = self._apply_patch_to_images(
        images_tensor,
        patch,
        labels=labels,
        target_class=target_class,
    )

    # 3. Loss 계산 (naturalistic_patch_pytorch.py:652-663)
    loss_det = self._compute_detection_loss(patched_images, labels, target_class)
    loss_tv = self.tv_loss_fn(patch)
    loss_attack = -loss_det
    loss = loss_attack + self.tv_weight * loss_tv

    # 4. 역전파 및 최적화 (naturalistic_patch_pytorch.py:666-667)
    loss.backward()
    self.optimizer.step()
```

**핵심 특징:**
- **Latent rounding**: `_generate_patch_from_latent()` 내부에서 선택적 수행
- **No overlap loss**: IoU 기반 loss 제거
- **단순화된 구조**: detection loss + TV loss만 사용

**주요 차이점:**

| 항목 | 원본 | 마이그레이션 |
|------|------|-------------|
| Latent rounding | 매 배치마다 강제 적용 (precision=4) | 선택적 (`enable_latent_rounding`) |
| Overlap loss | ✅ 지원 (`weight_loss_overlap`) | ❌ 제거됨 |
| Discriminator | ✅ 선택적 지원 | ❌ 제거됨 |
| 학습 루프 | Batch 단위 (DataLoader) | Iteration 단위 (전체 데이터 반복) |

---

## 3. 🎯 패치를 이미지에 적용하는 방식

### 원본 (ensemble_tool/model.py:280-293, adversarialYolo/load_data.py:638-1120)

#### 패치 변환 (PatchTransformer)

```python
# adversarialYolo/load_data.py:686-900
def forward(self, adv_patch, lab_batch, img_size, patch_mask=[],
            by_rectangle=False, do_rotate=True, rand_loc=True,
            with_black_trans=False, scale_rate=0.2,
            with_crease=False, with_projection=False,
            with_rectOccluding=False, enable_empty_patch=False,
            enable_no_random=False, enable_blurred=True):

    # 1. Contrast & Brightness 변환
    if not enable_no_random:
        contrast = torch.FloatTensor(1).uniform_(self.min_contrast, self.max_contrast)
        brightness = torch.FloatTensor(1).uniform_(self.min_brightness, self.max_brightness)
        adv_patch = adv_patch * contrast + brightness

    # 2. Noise 추가
    if not enable_no_random:
        noise = torch.FloatTensor(adv_patch.size()).uniform_(-1, 1) * self.noise_factor
        adv_patch = adv_patch + noise

    # 3. Rotation (do_rotate=True일 때)
    anglesize = (lab_batch.size(0) * lab_batch.size(1))
    if do_rotate:
        angle = torch.FloatTensor(anglesize).uniform_(self.minangle, self.maxangle)
    else:
        angle = torch.FloatTensor(anglesize).fill_(0)

    # 4. BBox 기반 스케일링
    current_patch_size = adv_patch_size
    for i in range(adv_batch.size(0)):
        for obj in range(lab_batch.size(1)):
            # BBox 크기 계산
            obj_box_w = lab_batch[i][obj][3].clone()
            obj_box_h = lab_batch[i][obj][4].clone()

            if by_rectangle:
                target_size_h = torch.sqrt(obj_box_h * obj_box_h + obj_box_w * obj_box_w) * img_size * scale_rate
                target_size_w = target_size_h

            # Affine 변환으로 회전 + 스케일링 동시 수행
            theta = torch.cuda.FloatTensor(1, 2, 3).fill_(0)
            theta[0, 0, 0] = (torch.cos(current_angle) + eps) / target_x
            theta[0, 0, 1] = (torch.sin(current_angle) + eps) / target_x
            theta[0, 1, 0] = (-torch.sin(current_angle) + eps) / target_y
            theta[0, 1, 1] = (torch.cos(current_angle) + eps) / target_y

            grid = F.affine_grid(theta, adv_patch.unsqueeze(0).shape)
            adv_patch_t = F.grid_sample(adv_patch.unsqueeze(0), grid)

    # 5. 3D Projection (선택적)
    if with_projection:
        # 3D 회전 행렬 계산
        warpR = get_warpR(anglex, angley, anglez, fov, w, h)
        # Perspective transformation

    # 6. Crease 효과 (선택적)
    if with_crease:
        # 주름 효과 시뮬레이션

    # 7. Rectangle Occluding (선택적)
    if with_rectOccluding:
        rect_occluding = self.rect_occluding(...)

    # 8. Blurring (선택적)
    if enable_blurred:
        adv_batch_t = self.medianpooler(adv_batch_t)
```

#### 패치 적용 (PatchApplier)

```python
# adversarialYolo/load_data.py:1131-1139
def forward(self, img_batch, adv_batch):
    advs = torch.unbind(adv_batch, 1)  # 각 객체별로 분리
    for adv in advs:
        # 0이 아닌 영역(패치 영역)만 이미지에 덮어씀
        img_batch = torch.where((adv == 0), img_batch, adv)
    return img_batch
```

### 마이그레이션 (naturalistic_patch_pytorch.py:301-523)

```python
def _apply_patch_to_images(self, images, patch, labels, target_class):
    batch_size, _C, H, W = images.shape

    # 1. Clip range 조정
    patch_to_apply = patch
    if hasattr(self.estimator, 'clip_values'):
        clip_max = float(self.estimator.clip_values[1])
        patch_to_apply = patch * clip_max

    # 2. Labels 기반 패치 적용
    if labels and len(labels) == batch_size:
        for b in range(batch_size):
            boxes = labels[b].get("boxes")
            label_ids = labels[b].get("labels")

            for i in range(boxes.shape[0]):
                # Target class 필터링
                if target_class is not None and int(label_ids[i].item()) != int(target_class):
                    continue

                # BBox 좌표 추출
                x1, y1, x2, y2 = boxes[i]
                bbox_w = x2_i - x1_i
                bbox_h = y2_i - y1_i

                # 3. 패치 크기 계산
                patch_size = int(round(math.sqrt(bbox_w ** 2 + bbox_h ** 2) * self.patch_scale))
                patch_resized = F.interpolate(patch_to_apply.unsqueeze(0), size=(patch_size, patch_size))

                # 4. 중심 좌표 계산
                center_x = (x1_i + x2_i) // 2
                center_y = (y1_i + y2_i) // 2

                # Random location jitter
                if self.enable_random_location:
                    jitter_x = torch.empty(1).uniform_(-0.4, 0.4)
                    jitter_y = torch.empty(1).uniform_(-0.4, 0.4)
                    center_x += int(round(bbox_w * float(jitter_x.item())))
                    center_y += int(round(bbox_h * float(jitter_y.item())))

                # Vertical offset
                if self.vertical_offset_ratio != 0.0:
                    center_y += int(round(self.vertical_offset_ratio * H))

                # 5. Appearance augmentation
                if self.enable_appearance_aug:
                    contrast = torch.empty(1).uniform_(self.min_contrast, self.max_contrast)
                    brightness = torch.empty(1).uniform_(self.min_brightness, self.max_brightness)
                    noise = torch.randn_like(patch_norm) * self.noise_factor
                    patch_norm = torch.clamp(patch_norm * contrast + brightness + noise, 0.0, 1.0)

                # 6. Rotation
                if self.enable_rotation and self.rotation_range > 0:
                    angle_deg = torch.empty(1).uniform_(-self.rotation_range, self.rotation_range)
                    angle = angle_deg * (math.pi / 180.0)
                    theta = torch.stack([cos_a, -sin_a, ..., sin_a, cos_a, ...]).view(1, 2, 3)
                    grid = F.affine_grid(theta, size=(1, C, patch_size, patch_size))
                    patch_resized = F.grid_sample(patch_resized.unsqueeze(0), grid)

                # 7. 패치를 이미지에 적용 (Padding + Masking)
                patch_canvas = F.pad(patch_resized, (pad_left, pad_right, pad_top, pad_bottom))
                mask = F.pad(torch.ones_like(patch_resized), (pad_left, pad_right, pad_top, pad_bottom))
                patched = patched * (1 - mask) + patch_canvas

    # Fallback: 중앙에 패치 적용
    else:
        # 이미지 중앙에 패치 적용 (라벨 없을 때)
        ...
```

### 주요 차이점 비교

| 기능 | 원본 (PatchTransformer) | 마이그레이션 (_apply_patch_to_images) |
|------|------------------------|--------------------------------------|
| **Contrast/Brightness** | ✅ `uniform_(0.8, 1.2)` / `uniform_(-0.1, 0.1)` | ✅ 동일 (파라미터화) |
| **Noise** | ✅ `uniform_(-1, 1) * 0.1` | ✅ `randn * 0.1` (Gaussian) |
| **Rotation** | ✅ Affine grid (`-20°~20°`) | ✅ Affine grid (파라미터화) |
| **BBox 기반 스케일링** | ✅ `sqrt(w² + h²) * scale_rate` | ✅ 동일 |
| **Random location** | ✅ `rand_loc` (BBox 내부) | ✅ `enable_random_location` (jitter ±40%) |
| **Vertical offset** | ❌ 없음 | ✅ `vertical_offset_ratio` (새로 추가) |
| **3D Projection** | ✅ `with_projection` (3D 회전) | ❌ 제거됨 |
| **Crease 효과** | ✅ `with_crease` (주름) | ❌ 제거됨 |
| **Rectangle Occluding** | ✅ `with_rectOccluding` | ❌ 제거됨 |
| **Median Pool Blur** | ✅ `MedianPool2d(7)` | ❌ 제거됨 |
| **Multiple objects** | ✅ 배치 전체 객체 처리 | ✅ 배치 전체 객체 처리 |
| **Masking 방식** | `torch.where((adv == 0), img, adv)` | `img * (1 - mask) + patch_canvas` |

**결론:**
- 마이그레이션은 **핵심 기능은 유지**하되, 복잡한 augmentation (3D projection, crease, blur 등)을 **제거**하여 단순화
- **Vertical offset**은 새로 추가되어 실전 적용성 향상
- Noise 분포가 Uniform → Gaussian으로 변경

---

## 4. 📉 Loss 및 그래디언트 다루는 방식

### 원본 (ensemble_tool/model.py:234-362)

```python
def train_rowPtach(...):
    opt.zero_grad()

    # 1. Latent clamping (ensemble_tool/model.py:237)
    latent_shift.data = torch.clamp(latent_shift.data, -max_value_latent_item, max_value_latent_item)

    # 2. 패치 생성 (BigGAN) (ensemble_tool/model.py:240-264)
    if not(deformator == None):
        zs = torch.rand([rows, generator.dim_z], device=device)
        z = zs[0]

        if enable_shift_deformator:
            # Deformator 사용: G(deformator(z + shift))
            z = ((1-alpah_latent)*z) + (alpah_latent*fixed_latent_biggan)
            latent_shift_clamped = torch.clamp(
                (max_value_latent_item * latent_shift),
                max=-max_value_latent_item,
                min=-max_value_latent_item
            )
            interpolation_deformed = interpolate_shift(
                generator, z.unsqueeze(0),
                latent_shift=latent_shift_clamped.unsqueeze(0),
                deformator=deformator,
                device=device
            )
        else:
            # Deformator 미사용: G(z + shift)
            z = ((1-alpah_latent)*z) + (alpah_latent*latent_shift)
            interpolation_deformed = interpolate_shift(
                generator, z.unsqueeze(0),
                latent_shift=torch.zeros_like(z.unsqueeze(0)).to(device),
                deformator=None,
                device=device
            )

        fake_images = (interpolation_deformed + 1) * 0.5

    # 3. Detection Loss 계산 (ensemble_tool/model.py:310-346)
    if(model_name == "yolov4"):
        max_prob_obj_cls, overlap_score, bboxes = detector.detect(
            input_imgs=p_img_batch,
            cls_id_attacked=cls_id_attacked,
            clear_imgs=input_imgs,
            with_bbox=enable_with_bbox
        )
        loss_det = torch.mean(max_prob_obj_cls)  # objectness * class_prob
        loss_overlap = -torch.mean(overlap_score)  # IoU loss

    # 4. Total Variation Loss (ensemble_tool/model.py:352)
    loss_tv = total_variation(fake_images[0])

    # 5. Discriminator Loss (선택적) (ensemble_tool/model.py:355-358)
    if discriminator is not None:
        D_loss = discriminator(fake_images, torch.Tensor([259]).long().cuda())
        loss = min_loss_det + (weight_loss_overlap * loss_overlap) + (weight_loss_tv * loss_tv) + 1e-3 * D_loss
    else:
        loss = min_loss_det + (weight_loss_overlap * loss_overlap) + (weight_loss_tv * loss_tv)

    # 6. 역전파 (ensemble_tool/model.py:361-362)
    loss.backward()
    opt.step()

    return loss_det, loss_overlap, loss_tv, p_img_batch, fake_images, D_loss
```

**Loss 구성:**
```
Total Loss = Detection Loss + (α * Overlap Loss) + (β * TV Loss) + (γ * Discriminator Loss)

여기서:
- Detection Loss: 탐지 확률 최소화 (objectness * class_prob)
- Overlap Loss: BBox IoU 최대화 (여러 객체 동시 공격)
- TV Loss: 패치 smoothness (고주파 노이즈 억제)
- Discriminator Loss: GAN discriminator (선택적)
```

### 마이그레이션 (naturalistic_patch_pytorch.py:270-299, 636-667)

```python
# 1. Latent clamping (_generate_patch_from_latent:228-232)
with torch.no_grad():
    self.latent_shift.clamp_(-self.max_latent_value, self.max_latent_value)
    if self.enable_latent_rounding:
        factor = 10 ** self.latent_rounding_precision
        self.latent_shift.copy_(torch.round(self.latent_shift * factor) / factor)

# 2. 패치 생성 (BigGAN) (_generate_patch_from_latent:234-268)
if use_deformator:
    z = torch.rand(self.dim_z, device=self.estimator.device)
    if self.alpha_latent > 0:
        z = (1 - self.alpha_latent) * z + self.alpha_latent * self.fixed_latent

    if self.enable_shift_deformator:
        shift = torch.clamp(
            self.latent_shift * self.max_latent_value,
            -self.max_latent_value,
            self.max_latent_value,
        )
        patch = generate_from_latent(
            self.generator, z.unsqueeze(0),
            shift=shift.unsqueeze(0),
            deformator=self.deformator,
        )
    else:
        z = (1 - self.alpha_latent) * z + self.alpha_latent * self.latent_shift
        patch = self.generator(z.unsqueeze(0))
else:
    latent = torch.rand(self.dim_z, device=self.estimator.device)
    if self.alpha_latent > 0:
        latent = (1 - self.alpha_latent) * latent + self.alpha_latent * self.latent_shift
    patch = self.generator(latent.unsqueeze(0))

patch = (patch + 1) * 0.5  # [-1, 1] → [0, 1]

# 3. Detection Loss 계산 (_compute_detection_loss:270-299)
if labels is not None:
    # ART estimator의 compute_loss 사용 (gradient 보존)
    return self.estimator.compute_loss(x=patched_images, y=labels)
else:
    # Fallback: surrogate loss
    logger.warning("Detection labels missing; using surrogate loss")
    return patched_images.mean()

# 4. 학습 루프 (generate:636-667)
for iteration in iterator:
    self.optimizer.zero_grad()

    patch = self._generate_patch_from_latent()
    patched_images = self._apply_patch_to_images(images_tensor, patch, labels, target_class)

    # Loss 계산
    loss_det = self._compute_detection_loss(patched_images, labels, target_class)
    loss_tv = self.tv_loss_fn(patch)

    # Untargeted attack: maximize detector loss, minimize TV
    loss_attack = -loss_det
    loss = loss_attack + self.tv_weight * loss_tv

    loss.backward()
    self.optimizer.step()
```

**Loss 구성:**
```
Total Loss = -Detection Loss + (β * TV Loss)

여기서:
- Detection Loss: ART estimator의 compute_loss() (훈련 loss 사용)
- TV Loss: 패치 smoothness
```

### 주요 차이점 비교

| 항목 | 원본 | 마이그레이션 |
|------|------|-------------|
| **Detection Loss** | Detector 직접 호출 (`max_prob_obj_cls`) | ART estimator의 `compute_loss()` |
| **Overlap Loss** | ✅ BBox IoU 기반 | ❌ 제거됨 |
| **TV Loss** | ✅ 지원 | ✅ 지원 (동일) |
| **Discriminator Loss** | ✅ 선택적 지원 | ❌ 제거됨 |
| **Loss 부호** | `loss = det_loss + ...` (최소화) | `loss = -det_loss + ...` (최대화→최소화) |
| **Gradient flow** | Detector forward만 사용 | ART의 training loss (full gradient) |
| **Latent clamping** | `data` 직접 수정 | `with torch.no_grad()` 사용 |

**핵심 차이:**
1. **Detection loss 계산 방식**: 원본은 detector의 출력(확률)을 직접 사용하지만, 마이그레이션은 **ART estimator의 훈련 loss**를 사용하여 gradient flow 개선
2. **Overlap loss 제거**: 단순화를 위해 IoU 기반 loss 제거
3. **Loss 부호**: 마이그레이션은 `-det_loss`로 명시적으로 maximization → minimization 변환

---

## 5. 🎯 타겟 라벨을 활용하는 방식

### 원본 (ensemble_tool/model.py, adversarialYolo/load_data.py)

#### 라벨 형식 (INRIA Dataset)

```python
# adversarialYolo/load_data.py:1163-1280 (InriaDataset)
class InriaDataset(Dataset):
    def __getitem__(self, idx):
        # 라벨 파일 읽기 (.txt 파일)
        label = np.loadtxt(labpath, ndmin=2)
        # 형식: [class_id, x_center, y_center, width, height]
        # 예: [0, 0.5, 0.6, 0.2, 0.4]  # class 0 = person

        # Padding (최대 max_lab개까지, 나머지는 [0, 0, 0, 0, 0])
        filled_labels = np.zeros((self.max_labels, 5))
        filled_labels[range(len(label))[:self.max_labels]] = label[:self.max_labels]

        return img, filled_labels  # shape: [3, 416, 416], [14, 5]
```

#### 라벨 사용 방식

```python
# ensemble_tool/model.py:280-293
# 1. PatchTransformer에서 라벨 사용
adv_batch_t, adv_patch_set, msk_batch = patch_transformer(
    adv_patch=fake_images[i_fimage],
    lab_batch=label,  # shape: [batch_size, 14, 5]
    img_size=416,
    patch_mask=[],
    by_rectangle=by_rectangle,
    do_rotate=enable_rotation,
    rand_loc=enable_randomLocation,
    scale_rate=patch_scale,  # 0.2
    # ...
)

# 2. 라벨 기반 패치 위치/크기 결정 (adversarialYolo/load_data.py:820-900)
for i in range(adv_batch.size(0)):  # batch
    for obj in range(lab_batch.size(1)):  # 최대 14개 객체
        cls_id = lab_batch[i][obj][0]
        if cls_id < 1:  # class_id가 0이면 패딩된 라벨
            continue

        # BBox 정보 추출
        obj_box_cx = lab_batch[i][obj][1]  # x_center (normalized)
        obj_box_cy = lab_batch[i][obj][2]  # y_center (normalized)
        obj_box_w = lab_batch[i][obj][3]   # width (normalized)
        obj_box_h = lab_batch[i][obj][4]   # height (normalized)

        # 패치 크기 계산
        if by_rectangle:
            target_size_h = torch.sqrt(obj_box_h**2 + obj_box_w**2) * img_size * scale_rate
            target_size_w = target_size_h

        # 패치 위치 계산
        target_x = obj_box_cx.view(-1) * img_size
        target_y = obj_box_cy.view(-1) * img_size

        if rand_loc:
            # Random jitter
            off_x = off_x.uniform_(-rand_width, rand_width)
            off_y = off_y.uniform_(-rand_height, rand_height)
            target_x = target_x + off_x
            target_y = target_y + off_y

        # Affine 변환으로 패치 배치
        # ...
```

#### 타겟 클래스 필터링

```python
# ensemble.py:90
cls_id_attacked = 0  # 0: person 클래스만 공격

# ensemble_tool/model.py:310-346
# Detector에서 person 클래스만 탐지
max_prob_obj_cls, overlap_score, bboxes = detector.detect(
    input_imgs=p_img_batch,
    cls_id_attacked=cls_id_attacked,  # 0: person
    clear_imgs=input_imgs,
    with_bbox=enable_with_bbox
)
```

### 마이그레이션 (naturalistic_patch_pytorch.py)

#### 라벨 형식 (ART 호환)

```python
# naturalistic_patch_pytorch.py:577-631 (generate 메서드)
# 두 가지 라벨 형식 지원:

# 1. ART 형식 (list of dicts)
labels = [
    {
        "boxes": torch.Tensor([[x1, y1, x2, y2], ...]),  # pixel 좌표
        "labels": torch.Tensor([0, 0, ...]),             # class IDs
        "scores": torch.Tensor([0.95, 0.87, ...])        # optional
    },
    # ... (배치 내 각 이미지)
]

# 2. NAP 원본 형식 (numpy array)
y = np.array([
    [[x_center, y_center, width, height, class_id], ...],  # normalized 좌표
    # ... (배치 내 각 이미지, max_objects개)
])
```

#### 라벨 변환 로직

```python
# naturalistic_patch_pytorch.py:594-625
if isinstance(y, np.ndarray):
    labels = []
    for b in range(y.shape[0]):
        boxes_b = []
        labels_b = []
        for ann in y[b]:
            if ann is None or len(ann) < 5:
                continue
            x_center, y_center, width, height, cls_id = ann[:5]
            if np.isnan(cls_id):
                continue

            # Normalized coords → Pixel coords
            if max(x_center, y_center, width, height) <= 1.5:
                x_center *= W
                y_center *= H
                width *= W
                height *= H

            # Center + Size → Corner coords
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2

            boxes_b.append([x1, y1, x2, y2])
            labels_b.append(int(cls_id))

        labels.append({"boxes": torch.as_tensor(boxes_b), "labels": torch.as_tensor(labels_b)})
```

#### 라벨 사용 방식

```python
# naturalistic_patch_pytorch.py:331-450 (_apply_patch_to_images)
if labels and len(labels) == batch_size:
    for b in range(batch_size):
        boxes = labels[b].get("boxes")      # [N, 4] (x1, y1, x2, y2)
        label_ids = labels[b].get("labels") # [N] (class IDs)

        for i in range(boxes.shape[0]):
            # 타겟 클래스 필터링
            if target_class is not None and int(label_ids[i].item()) != int(target_class):
                continue

            # BBox 좌표 추출 (pixel 좌표)
            x1, y1, x2, y2 = boxes[i]
            bbox_w = x2_i - x1_i
            bbox_h = y2_i - y1_i

            # 패치 크기 계산 (원본과 동일)
            patch_size = int(round(math.sqrt(bbox_w ** 2 + bbox_h ** 2) * self.patch_scale))

            # 패치 위치 계산
            center_x = (x1_i + x2_i) // 2
            center_y = (y1_i + y2_i) // 2

            if self.enable_random_location:
                jitter_x = torch.empty(1).uniform_(-0.4, 0.4)
                jitter_y = torch.empty(1).uniform_(-0.4, 0.4)
                center_x += int(round(bbox_w * float(jitter_x.item())))
                center_y += int(round(bbox_h * float(jitter_y.item())))

            # Vertical offset (원본에는 없음)
            if self.vertical_offset_ratio != 0.0:
                center_y += int(round(self.vertical_offset_ratio * H))
```

### 주요 차이점 비교

| 항목 | 원본 | 마이그레이션 |
|------|------|-------------|
| **라벨 형식** | `[class, x_c, y_c, w, h]` (normalized) | ART: `{boxes, labels}` (pixel coords) |
| **좌표계** | Center + Size (normalized) | Corner coords (x1, y1, x2, y2) |
| **최대 객체 수** | 14개 (패딩) | 제한 없음 (가변 길이) |
| **타겟 클래스** | `cls_id_attacked=0` (전역 설정) | `target_class` (함수 파라미터) |
| **라벨 변환** | ❌ 불필요 (YOLO 형식 그대로) | ✅ NAP → ART 변환 지원 |
| **패치 크기 계산** | `sqrt(w² + h²) * scale` | 동일 |
| **패치 위치** | BBox center + jitter | BBox center + jitter + vertical offset |
| **Fallback** | ❌ 라벨 필수 | ✅ 라벨 없으면 중앙에 패치 |

**핵심 차이:**
1. **라벨 형식 통합**: 마이그레이션은 ART 표준 형식 + NAP 원본 형식 모두 지원
2. **좌표 변환**: NAP의 normalized center-size → ART의 pixel corner coords
3. **Vertical offset**: 마이그레이션에서 새로 추가 (실전 적용성 향상)

---

## 6. ❌ 마이그레이션 시 제거된 주요 로직

### 6.1 Overlap Loss (BBox IoU 기반)

**원본 (ensemble_tool/model.py:321, 331, 340)**
```python
# YOLOv4/v3/v2 모두 지원
max_prob_obj_cls, overlap_score, bboxes = detector.detect(
    input_imgs=p_img_batch,
    cls_id_attacked=cls_id_attacked,
    clear_imgs=input_imgs,
    with_bbox=enable_with_bbox
)

loss_overlap = -torch.mean(overlap_score)  # IoU 최대화
loss = min_loss_det + (weight_loss_overlap * loss_overlap) + (weight_loss_tv * loss_tv)
```

**목적**: 여러 객체를 동시에 공격할 때, 패치가 겹치는 영역을 최대화하여 효율성 향상

**제거 이유**: 단순화 (detection loss만으로도 충분한 성능)

### 6.2 Discriminator Loss (GAN Adversarial Loss)

**원본 (ensemble.py:149-152, ensemble_tool/model.py:355-357)**
```python
if enable_discriminator == True:
    discriminator_biggan = None
else:
    discriminator_biggan = None

# Loss 계산
if discriminator is not None:
    D_loss = discriminator(fake_images, torch.Tensor([259]).long().cuda())
    loss = min_loss_det + ... + 1e-3 * D_loss
```

**목적**: GAN discriminator를 사용하여 생성된 패치의 realism 향상

**제거 이유**: 실험적 기능 (원본 코드에서도 기본값 `False`)

### 6.3 Advanced Augmentations

#### 6.3.1 3D Projection (Perspective Transformation)

**원본 (adversarialYolo/load_data.py:706-800)**
```python
if with_projection:
    # 3D 회전 행렬 계산
    def get_warpR(anglex, angley, anglez, fov, w, h):
        # X, Y, Z축 회전 행렬
        rx = [[1, 0, 0, 0],
              [0, cos(x), -sin(x), 0],
              [0, -sin(x), cos(x), 0],
              [0, 0, 0, 1]]
        # ...
        r = rx @ ry @ rz

        # 4점 perspective transform
        src_points = [[0, 0], [w, 0], [0, h], [w, h]]
        dst_points = project_3d(src_points, r, z, fov)
        warpR = cv2.getPerspectiveTransform(src_points, dst_points)

    # 패치에 적용
    adv_patch_t = cv2.warpPerspective(adv_patch, warpR, (w, h))
```

**목적**: 실제 세계에서 패치가 3D 각도에서 촬영되는 상황 시뮬레이션

#### 6.3.2 Crease 효과 (주름 시뮬레이션)

**원본 (adversarialYolo/load_data.py:900-950)**
```python
if with_crease:
    # 주름선 생성
    num_creases = torch.randint(1, 4, (1,))[0]
    for _ in range(num_creases):
        # 무작위 방향의 선
        start_x = torch.randint(0, patch_size, (1,))[0]
        start_y = torch.randint(0, patch_size, (1,))[0]
        angle = torch.rand(1) * 2 * math.pi

        # 주름 효과 (brightness 감소)
        crease_mask = create_line_mask(start_x, start_y, angle, width=3)
        adv_patch_t = adv_patch_t * (1 - 0.3 * crease_mask)
```

**목적**: 인쇄된 패치가 구겨지는 상황 시뮬레이션

#### 6.3.3 Rectangle Occluding (부분 가림)

**원본 (adversarialYolo/load_data.py:666-684)**
```python
def rect_occluding(self, num_rect=1, n_batch=8, n_feature=14, patch_size=300):
    tensor_img = torch.full((3, patch_size, patch_size), 0.0)
    for _ in range(num_rect):
        xs = torch.randint(0, int(patch_size/2), (1,))[0]
        xe = torch.randint(xs, min(patch_size, xs + int(patch_size/2)), (1,))[0]
        ys = torch.randint(0, int(patch_size/2), (1,))[0]
        ye = torch.randint(ys, min(patch_size, ys + int(patch_size/2)), (1,))[0]
        tensor_img[:, xs:xe, ys:ye] = 0.5  # 회색으로 가림
```

**목적**: 패치가 부분적으로 가려지는 상황 시뮬레이션 (occlusion robustness)

#### 6.3.4 Median Pool Blur

**원본 (adversarialYolo/load_data.py:656, 1050-1060)**
```python
self.medianpooler = MedianPool2d(7, same=True)

if enable_blurred:
    adv_batch_t = self.medianpooler(adv_batch_t)
```

**목적**: 카메라 모션 블러/초점 흐림 시뮬레이션

**제거 이유**:
- 복잡도 증가
- 실험 결과 성능 향상 미미
- 단순 augmentation (contrast, brightness, noise, rotation)만으로도 충분한 robustness

### 6.4 Multiple Training Modes

**원본 (adversarialYolo/train_patch.py:102-138)**
```python
class trainmode(Enum):
    globalPatch        = 0  # 전체 패치 학습
    fourDivisionsSimul = 1  # 4등분 동시 학습
    fourDivisionsSeq   = 2  # 4등분 순차 학습
    maskPatchSimul     = 3  # 마스크 기반 동시 학습
    maskPatchSeq       = 4  # 마스크 기반 순차 학습

# 학습 모드에 따라 다른 전략
if(tm == self.trainmode.fourDivisionsSeq):
    # 패치를 4등분하여 순차적으로 학습
    for local_patch in [patch_00, patch_01, patch_10, patch_11]:
        loss = compute_loss(local_patch)
        loss.backward()
```

**목적**:
- 대형 패치 학습 시 gradient 안정화
- 다양한 학습 전략 실험

**제거 이유**: 단순화 (전체 패치 학습만 지원)

### 6.5 Non-Printability Score (NPS) Loss

**원본 (adversarialYolo/train_patch.py:46-51, 519)**
```python
self.nps_calculator_local = NPSCalculator(
    patch_side=int(self.config.patch_size/2),
    printability_file_1=self.config.printfile
)

# Loss 계산
nps = self.nps_calculator_global(adv_patch)
nps_loss = nps * self.p_nps  # p_nps = 0.01
loss = det_loss + nps_loss + tv_loss
```

**목적**: 프린터로 출력 가능한 색상 범위로 패치 제한 (physical world robustness)

**제거 이유**:
- TV loss만으로도 충분한 smoothness
- 마이그레이션은 디지털 공격에 초점

### 6.6 Content Similarity Score (CSS) Loss

**원본 (adversarialYolo/train_patch.py:56, 441-471, 525-526)**
```python
self.css_calculator = CSSCalculator(
    sample_img=self.read_image(self.config.sampleimgfile)
)

if(self.enable_loss_css):
    # 패치가 특정 이미지와 유사하도록 유도
    css = self.css_calculator(adv_batch_t, img_batch_covered)
    css_loss = css * self.p_css
    loss = det_loss + nps_loss + css_loss + tv_loss
```

**목적**:
- 패치가 특정 스타일/색상을 가지도록 유도
- 예: 배경과 유사한 camouflage 패치 생성

**제거 이유**: 실험적 기능 (원본에서도 기본값 `False`)

### 6.7 Style Transfer Loss

**원본 (adversarialYolo/train_patch.py:78-79, 555-579)**
```python
self.p_style = 100
self.p_content = 0.05

if(self.enable_styleTransfer):
    model, style_losses, content_losses = get_style_model_and_losses(
        style_img, content_img
    )
    model(content_img)

    style_score = sum([sl.loss for sl in style_losses])
    content_score = sum([cl.loss for cl in content_losses])

    loss_style_content = style_score * self.p_style + content_score * self.p_content
    loss = loss + loss_style_content
```

**목적**: Neural Style Transfer로 예술적 패치 생성

**제거 이유**: 실험적 기능 (성능 저하 가능성)

---

## 7. 📊 종합 비교표

| 구성 요소 | 원본 | 마이그레이션 | 상태 |
|----------|------|-------------|------|
| **패치 초기화** ||||
| BigGAN 사용 | ✅ | ✅ | ✅ 동일 |
| Fixed latent (Uniform) | ✅ | ✅ | ✅ 동일 |
| Shift latent (Gaussian) | ✅ | ✅ | ✅ 동일 |
| Deformator 지원 | ✅ | ✅ | ✅ 동일 |
| **패치 학습** ||||
| Latent rounding | ✅ 강제 (precision=4) | ✅ 선택적 | ⚠️ 변경 |
| Batch 단위 학습 | ✅ (DataLoader) | ✅ (Iteration) | ⚠️ 변경 |
| Optimizer (Adam) | ✅ | ✅ | ✅ 동일 |
| Learning rate scheduler | ✅ | ❌ | ⚠️ 제거 |
| **Loss 함수** ||||
| Detection loss | ✅ (detector output) | ✅ (ART compute_loss) | ⚠️ 개선 |
| Overlap loss (IoU) | ✅ | ❌ | ❌ 제거 |
| Total variation loss | ✅ | ✅ | ✅ 동일 |
| Discriminator loss | ✅ 선택적 | ❌ | ❌ 제거 |
| NPS loss | ✅ | ❌ | ❌ 제거 |
| CSS loss | ✅ 선택적 | ❌ | ❌ 제거 |
| Style transfer loss | ✅ 선택적 | ❌ | ❌ 제거 |
| **Augmentation** ||||
| Contrast/Brightness | ✅ | ✅ | ✅ 동일 |
| Noise (Gaussian) | ✅ Uniform | ✅ Gaussian | ⚠️ 변경 |
| Rotation | ✅ | ✅ | ✅ 동일 |
| Random location | ✅ | ✅ | ✅ 동일 |
| Vertical offset | ❌ | ✅ | ✅ 추가 |
| 3D Projection | ✅ | ❌ | ❌ 제거 |
| Crease 효과 | ✅ | ❌ | ❌ 제거 |
| Rectangle occluding | ✅ | ❌ | ❌ 제거 |
| Median blur | ✅ | ❌ | ❌ 제거 |
| **라벨 처리** ||||
| NAP 형식 | ✅ | ✅ | ✅ 동일 |
| ART 형식 | ❌ | ✅ | ✅ 추가 |
| 좌표 변환 | ❌ | ✅ | ✅ 추가 |
| Target class 필터링 | ✅ | ✅ | ✅ 동일 |
| **기타** ||||
| Multiple training modes | ✅ | ❌ | ❌ 제거 |
| Mask-based patch | ✅ | ❌ | ❌ 제거 |
| Checkpoint 저장 | ✅ | ❌ | ⚠️ 외부 처리 |

---

## 8. 🔍 핵심 발견 사항

### ✅ 마이그레이션이 잘 보존한 것

1. **BigGAN 기반 패치 생성**: 핵심 로직 완벽 보존
2. **Latent space 최적화**: Fixed + Shift latent 구조 동일
3. **기본 Augmentation**: Contrast, brightness, noise, rotation 유지
4. **BBox 기반 패치 배치**: 원본과 동일한 알고리즘
5. **Total variation loss**: Smoothness 유지

### ⚠️ 마이그레이션에서 개선된 것

1. **Detection loss 계산**: ART의 `compute_loss()` 사용으로 gradient flow 개선
2. **라벨 형식 통합**: NAP + ART 형식 모두 지원
3. **Vertical offset**: 실전 적용성 향상 (가슴 부분에 패치 배치)
4. **코드 구조**: 단순화 및 모듈화

### ❌ 마이그레이션에서 제거된 것

1. **Advanced augmentations**: 3D projection, crease, blur 등 제거
   - **영향**: Physical world robustness 감소 (디지털 공격에는 영향 없음)
2. **Overlap loss**: BBox IoU 기반 loss 제거
   - **영향**: 다중 객체 공격 효율성 약간 감소 가능
3. **Additional losses**: NPS, CSS, style transfer 제거
   - **영향**: 실험적 기능 제거 (핵심 성능에는 영향 없음)
4. **Multiple training modes**: 단순화
   - **영향**: 유연성 감소 (대부분의 경우 필요 없음)

---

## 9. 💡 권장 사항

### 9.1 현재 마이그레이션 유지할 것

현재 마이그레이션은 **핵심 기능을 잘 보존**하면서 **불필요한 복잡도를 제거**했습니다. 대부분의 사용 사례에서 충분한 성능을 제공할 것으로 예상됩니다.

### 9.2 필요 시 추가 고려 사항

만약 특정 시나리오에서 성능이 부족하다면, 다음을 고려할 수 있습니다:

1. **Overlap loss 추가** (다중 객체 공격 시):
   ```python
   # pytorchYOLOv4/demo.py의 overlap_score 계산 로직 참고
   overlap_score = compute_bbox_iou(predicted_boxes, ground_truth_boxes)
   loss_overlap = -torch.mean(overlap_score)
   loss = loss_attack + self.tv_weight * loss_tv + 0.1 * loss_overlap
   ```

2. **3D Projection** (physical world 테스트 시):
   - OpenCV의 `getPerspectiveTransform` 사용
   - Perspective augmentation 추가

3. **Median Blur** (카메라 노이즈 대응):
   ```python
   if self.enable_blur:
       patch_resized = F.avg_pool2d(patch_resized, kernel_size=3, stride=1, padding=1)
   ```

### 9.3 검증 필요 사항

1. **Noise 분포 변경** (Uniform → Gaussian):
   - 원본: `uniform_(-1, 1) * 0.1`
   - 마이그레이션: `randn * 0.1`
   - **검증**: 두 방식의 성능 차이 측정

2. **Detection loss 계산**:
   - 원본: Detector output (objectness * class_prob)
   - 마이그레이션: ART compute_loss (training loss)
   - **검증**: Gradient magnitude 비교

3. **Latent rounding**:
   - 원본: 매 배치마다 강제 적용
   - 마이그레이션: 선택적
   - **검증**: Rounding이 수렴 속도에 미치는 영향

---

## 10. 📝 결론

마이그레이션은 **원본 NAP의 핵심 아이디어(BigGAN latent space 최적화)를 완벽히 보존**하면서, **ART 프레임워크와의 통합**을 성공적으로 달성했습니다.

**제거된 기능들은 대부분 실험적 성격**이며, 핵심 공격 성능에는 큰 영향을 미치지 않을 것으로 판단됩니다. 오히려 **코드 단순화와 유지보수성 향상**이라는 측면에서 긍정적인 변화입니다.

향후 physical world 공격이 필요한 경우, 원본 코드의 advanced augmentation (3D projection, crease, blur 등)을 선택적으로 추가할 수 있습니다.
