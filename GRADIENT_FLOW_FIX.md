# Gradient Flow Fix for Naturalistic Adversarial Patch

## 수정 일자
2024-12-24

## 문제 요약

`backend/app/ai/attacks/evasion/naturalistic_patch_pytorch.py`의 `_generate_patch_from_latent()` 메서드에서 그래디언트 플로우가 단절되어 latent vector 최적화가 제대로 작동하지 않는 문제가 발견되었습니다.

## 발견된 버그

### 1. `.data` 직접 수정으로 인한 그래디언트 단절 (Line 177-181)

**기존 코드:**
```python
# ❌ 잘못된 구현
self.latent_shift.data = torch.clamp(
    self.latent_shift.data,
    -self.max_latent_value,
    self.max_latent_value
)
```

**문제점:**
- `tensor.data`를 직접 수정하면 autograd 그래프가 끊어짐
- `latent_shift`로 그래디언트가 역전파되지 않음

**수정 코드:**
```python
# ✅ 올바른 구현
with torch.no_grad():
    self.latent_shift.clamp_(-self.max_latent_value, self.max_latent_value)
```

**수정 이유:**
- `clamp_()`: in-place 연산이지만 그래디언트 그래프 유지
- `with torch.no_grad()`: clamping 자체는 그래디언트 추적 불필요 (메모리 절약)

### 2. **치명적: BigGAN 생성 시 그래디언트 차단 (Line 194)**

**기존 코드:**
```python
# ❌ 치명적인 버그
with torch.no_grad():
    # For deformator path (NAP's approach)
    if self.enable_deformator and self.deformator is not None:
        shift = self.latent_shift.unsqueeze(0)
        patch = generate_from_latent(...)
    else:
        z_shifted = z + self.latent_shift
        patch = self.generator(z_shifted.unsqueeze(0))
```

**문제점:**
- **패치 생성 전체가 `torch.no_grad()` 블록 안에 있음**
- `self.generator(z_shifted)` 연산이 그래디언트 추적 안됨
- `latent_shift`로 역전파 불가능
- **학습이 전혀 이루어지지 않음**

**수정 코드:**
```python
# ✅ 올바른 구현 (torch.no_grad() 제거)
# For deformator path (NAP's approach)
if self.enable_deformator and self.deformator is not None:
    shift = self.latent_shift.unsqueeze(0)
    patch = generate_from_latent(
        self.generator,
        z.unsqueeze(0),
        shift=shift,
        deformator=self.deformator
    )
else:
    z_shifted = z + self.latent_shift
    patch = self.generator(z_shifted.unsqueeze(0))
```

**수정 이유:**
- Generator를 통한 forward pass에서 그래디언트 추적 필요
- `latent_shift` → `z_shifted` → `generator` → `patch` → `loss` 경로의 그래디언트가 흘러야 함
- Generator는 이미 `.eval()` 모드이므로 가중치는 업데이트되지 않지만, 그래디언트는 통과함

## 그래디언트 플로우 경로

### 수정 전 (학습 불가능)
```
latent_shift (requires_grad=True)
    ↓
    X (data 직접 수정으로 끊김)
    ↓
z_shifted = z + latent_shift
    ↓
    X (torch.no_grad()로 완전 차단)
    ↓
generator(z_shifted)
    ↓
patch → loss → backward()
    ↓
    X (latent_shift로 그래디언트가 전혀 전달되지 않음)
```

### 수정 후 (정상 학습)
```
latent_shift (requires_grad=True)
    ↓
    ✓ (in-place clamp은 no_grad 블록 안에서만)
    ↓
z_shifted = z + latent_shift
    ↓
    ✓ (그래디언트 추적 활성화)
    ↓
generator(z_shifted)  [.eval() 모드, 가중치는 고정]
    ↓
patch → loss → backward()
    ↓
    ✓ (latent_shift로 그래디언트가 정상 전달됨!)
```

## 영향 분석

### 수정 전
- ❌ `latent_shift` 최적화 불가능
- ❌ 패치가 학습되지 않음 (랜덤 초기값 유지)
- ❌ Detection loss가 감소하지 않음
- ❌ 공격 효과 없음

### 수정 후
- ✅ `latent_shift` 정상 최적화
- ✅ BigGAN latent 공간에서 adversarial 방향 학습
- ✅ Detection loss 정상 감소
- ✅ 효과적인 adversarial patch 생성

## 추가 확인 사항

### Generator와 Deformator 설정 확인
- `load_biggan()` (loader.py:146)에서 `.eval()` 모드로 설정됨
- Generator 가중치는 고정되어 업데이트되지 않음
- 그래디언트는 통과하지만 가중치 업데이트는 안됨 (정상)

### Optimizer 설정 확인
```python
# Line 156
self.optimizer = torch.optim.Adam([self.latent_shift], lr=self.learning_rate)
```
- `latent_shift`만 optimizer에 등록됨 (정상)
- Generator 파라미터는 등록되지 않음 (정상)

## 테스트 방법

### 1. 그래디언트 플로우 확인
```python
# 패치 생성 후
patch = attack._generate_patch_from_latent()
loss = patch.sum()  # 간단한 더미 loss
loss.backward()

# latent_shift에 그래디언트가 있는지 확인
assert attack.latent_shift.grad is not None, "Gradient flow broken!"
print(f"Gradient norm: {attack.latent_shift.grad.norm().item()}")
```

### 2. 학습 과정 모니터링
```python
# training loop에서
initial_latent = attack.latent_shift.clone()
# ... 몇 iteration 후 ...
latent_change = (attack.latent_shift - initial_latent).norm()
print(f"Latent change: {latent_change.item()}")  # 0이 아니어야 함
```

### 3. Loss 감소 확인
```bash
# 로그에서 loss가 감소하는지 확인
# 수정 전: loss가 거의 변하지 않음
# 수정 후: loss가 점진적으로 감소
```

## 원본 NAP 구현과의 비교

### 원본 (ensemble_tool/model.py)
- **동일한 버그 존재**: `latent_shift.data = torch.clamp(...)`
- 하지만 `with torch.no_grad()` 블록은 없음
- → 원본도 부분적으로 문제가 있지만, army_ai보다는 나음

### army_ai 구현
- **더 심각한 버그**: `with torch.no_grad()` 추가로 완전히 학습 불가능
- 수정 후: 원본보다 더 올바른 구현

## 참고 문헌

- PyTorch Autograd 문서: https://pytorch.org/docs/stable/notes/autograd.html
- Naturalistic Physical Adversarial Patch (ICCV 2021)
- 원본 구현: https://github.com/aiiu-lab/Naturalistic-Adversarial-Patch

## 관련 파일

- `backend/app/ai/attacks/evasion/naturalistic_patch_pytorch.py` (수정됨)
- `backend/app/ai/gan/biggan/loader.py` (확인 완료, 문제없음)
- `backend/app/services/patch_service.py` (호출 코드, 문제없음)

## 커밋 정보

```bash
git add backend/app/ai/attacks/evasion/naturalistic_patch_pytorch.py
git commit -m "Fix gradient flow in NaturalisticPatchPyTorch

- Remove torch.no_grad() from patch generation
- Fix latent_shift clamping to use in-place operation
- Ensure gradients flow through generator to latent_shift
- Critical fix: enables actual training of adversarial patches"
```
