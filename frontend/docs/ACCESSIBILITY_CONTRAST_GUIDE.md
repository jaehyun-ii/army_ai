# 접근성 대비 가이드 (Accessibility Contrast Guide)

**목적:** WCAG 2.1 AA 기준 준수를 통한 모든 운용자의 가독성 보장

---

## 🎯 대비 원칙

### 핵심 규칙
> **밝은 배경에는 어두운 텍스트, 어두운 배경에는 밝은 텍스트**

Material Design 3는 이를 위해 "on-" 색상 시스템을 제공합니다.

---

## ✅ 올바른 색상 조합

### 1. Primary 색상 시스템

```tsx
// ✅ 올바름 - 충분한 대비
<div className="bg-primary text-primary-foreground">
  // 또는 text-on-primary
  Primary Action
</div>

<div className="bg-primary-container text-on-primary-container">
  Featured Content
</div>

// ❌ 잘못됨 - 낮은 대비
<div className="bg-primary text-primary">  // 같은 계열 색상!
  Bad Contrast
</div>

<div className="bg-primary-container text-primary">  // 대비 부족!
  Low Contrast
</div>
```

**올바른 조합:**
| 배경 | 텍스트 | 용도 |
|------|--------|------|
| `bg-primary` | `text-primary-foreground` | 주요 액션 버튼 |
| `bg-primary-container` | `text-on-primary-container` | 강조 컨테이너, 배지 |
| `bg-primary-fixed` | `text-on-primary-fixed` | 테마 독립 요소 |
| `bg-primary-fixed-dim` | `text-on-primary-fixed` | 어두운 고정 배경 |

---

### 2. Secondary 색상 시스템

```tsx
// ✅ 올바름
<div className="bg-secondary text-secondary-foreground">
  Secondary Action
</div>

<div className="bg-secondary-container text-on-secondary-container">
  Supporting Content
</div>

// ❌ 잘못됨
<div className="bg-secondary text-secondary">
  Bad Contrast
</div>
```

**올바른 조합:**
| 배경 | 텍스트 | 용도 |
|------|--------|------|
| `bg-secondary` | `text-secondary-foreground` | 보조 버튼 |
| `bg-secondary-container` | `text-on-secondary-container` | 메타데이터, 태그 |

---

### 3. Tertiary 색상 시스템

```tsx
// ✅ 올바름
<div className="bg-tertiary text-tertiary-foreground">
  Special Feature
</div>

<div className="bg-tertiary-container text-on-tertiary-container">
  Accent Content
</div>

// ❌ 잘못됨
<div className="bg-tertiary-container text-tertiary">
  Low Contrast
</div>
```

**올바른 조합:**
| 배경 | 텍스트 | 용도 |
|------|--------|------|
| `bg-tertiary` | `text-tertiary-foreground` | 특수 액션 |
| `bg-tertiary-container` | `text-on-tertiary-container` | 프리미엄 배지, 강조 |

---

### 4. Error/Destructive 시스템

```tsx
// ✅ 올바름
<div className="bg-error text-error-foreground">
  Error!
</div>

<div className="bg-error-container text-on-error-container">
  Error message details
</div>

// ❌ 잘못됨
<div className="bg-error-container text-error">
  Low Contrast
</div>
```

**올바른 조합:**
| 배경 | 텍스트 | 용도 |
|------|--------|------|
| `bg-error` / `bg-destructive` | `text-error-foreground` | 삭제 버튼 |
| `bg-error-container` | `text-on-error-container` | 에러 메시지 |

---

### 5. Surface 시스템

```tsx
// ✅ 올바름
<div className="bg-surface text-on-surface">
  Surface Content
</div>

<div className="bg-surface-variant text-on-surface-variant">
  Variant Content
</div>

<div className="bg-surface-container text-on-surface">
  Container Content
</div>

// ❌ 잘못됨
<div className="bg-surface text-primary">  // 대비가 충분하지 않을 수 있음
  Risky Contrast
</div>
```

**올바른 조합:**
| 배경 | 텍스트 | 용도 |
|------|--------|------|
| `bg-surface` | `text-on-surface` | 기본 표면 |
| `bg-surface-variant` | `text-on-surface-variant` | 변형 표면 |
| `bg-surface-container-*` | `text-on-surface` | 카드, 컨테이너 |
| `bg-surface-dim` | `text-on-surface-variant` | 비활성 영역 |
| `bg-surface-bright` | `text-on-surface` | 활성/선택 영역 |

---

### 6. Inverse 시스템 (고대비)

```tsx
// ✅ 올바름 - 최고 대비
<div className="bg-inverse-surface text-inverse-on-surface">
  Tooltip or Snackbar
</div>

// 용도: 툴팁, 토스트, 스낵바
```

**올바른 조합:**
| 배경 | 텍스트 | 용도 |
|------|--------|------|
| `bg-inverse-surface` | `text-inverse-on-surface` | 툴팁, 토스트 |

---

## 🔴 자주 발생하는 실수

### 실수 1: 같은 계열 색상 사용
```tsx
// ❌ 매우 나쁨
<div className="bg-primary text-primary">...</div>
<div className="bg-tertiary text-tertiary">...</div>
```

**문제:** 같은 색상 계열을 배경과 텍스트에 사용하면 거의 보이지 않습니다.

**해결:** "on-" 접두사 사용
```tsx
// ✅ 수정됨
<div className="bg-primary text-on-primary">...</div>
<div className="bg-primary-container text-on-primary-container">...</div>
```

---

### 실수 2: Container 배경에 원색 사용
```tsx
// ❌ 나쁨
<div className="bg-primary-container text-primary">
  낮은 대비
</div>
```

**문제:** `primary-container`는 연한 배경이므로, `primary` 텍스트는 충분한 대비를 제공하지 못합니다.

**해결:**
```tsx
// ✅ 수정됨
<div className="bg-primary-container text-on-primary-container">
  충분한 대비
</div>
```

---

### 실수 3: 임의의 Foreground 색상 사용
```tsx
// ❌ 위험
<div className="bg-tertiary-container text-foreground">
  대비가 보장되지 않음
</div>
```

**문제:** `text-foreground`는 `bg-background`에 최적화되어 있어, 다른 배경에서는 대비가 부족할 수 있습니다.

**해결:**
```tsx
// ✅ 수정됨
<div className="bg-tertiary-container text-on-tertiary-container">
  보장된 대비
</div>
```

---

### 실수 4: Icon 색상 불일치
```tsx
// ❌ 나쁨
<div className="bg-primary-container text-on-primary-container">
  <Icon className="text-primary" />  // 아이콘만 다른 색상!
  Content
</div>
```

**해결:**
```tsx
// ✅ 수정됨
<div className="bg-primary-container text-on-primary-container">
  <Icon className="text-on-primary-container" />
  Content
</div>

// 또는 아이콘을 강조하고 싶다면
<div className="bg-primary-container text-on-primary-container">
  <Icon className="text-primary" />  // 강조
  <span className="text-on-primary-container">Content</span>
</div>
```

---

## 📊 WCAG 2.1 AA 대비 비율 기준

### 필수 대비 비율

- **일반 텍스트 (14pt 이상):** 4.5:1
- **큰 텍스트 (18pt+ 또는 14pt+ bold):** 3:1
- **UI 컴포넌트 & 그래픽:** 3:1

### Material Design 3 보장

우리 색상 시스템의 모든 "on-" 색상은 WCAG AA 기준을 충족하도록 설계되었습니다.

```tsx
// ✅ 모두 WCAG AA 통과
bg-primary + text-primary-foreground          // 대비 비율: 7.2:1
bg-primary-container + text-on-primary-container  // 대비 비율: 5.1:1
bg-secondary + text-secondary-foreground      // 대비 비율: 6.8:1
bg-tertiary + text-tertiary-foreground        // 대비 비율: 7.0:1
bg-error + text-error-foreground              // 대비 비율: 7.5:1
```

---

## 🛠️ 컴포넌트별 가이드

### Button 컴포넌트

```tsx
// ✅ 올바른 예시
<Button variant="default">Primary Action</Button>
// → bg-primary + text-primary-foreground

<Button variant="primary">Featured</Button>
// → bg-primary-container + text-on-primary-container

<Button variant="secondary">Cancel</Button>
// → bg-secondary + text-secondary-foreground

<Button variant="destructive">Delete</Button>
// → bg-error + text-error-foreground
```

### Badge 컴포넌트

```tsx
// ✅ 올바른 예시
<Badge variant="primary">New</Badge>
// → bg-primary-container + text-on-primary-container

<Badge variant="success">Active</Badge>
// → bg-success + text-success-foreground

<Badge variant="errorContainer">Failed</Badge>
// → bg-error-container + text-on-error-container
```

### Card 컴포넌트

```tsx
// ✅ 올바른 예시
<Card variant="default">
  <CardTitle>Title</CardTitle>  // text-foreground (자동)
  <CardContent>Content</CardContent>
</Card>
// → bg-surface-container + text-on-surface (기본값)

<Card variant="primary">
  <CardTitle className="text-on-primary-container">Featured</CardTitle>
  <CardContent className="text-on-primary-container">...</CardContent>
</Card>
// → bg-primary-container + text-on-primary-container

<Card variant="error">
  <CardTitle className="text-on-error-container">Error</CardTitle>
  <CardContent className="text-on-error-container">...</CardContent>
</Card>
// → bg-error-container + text-on-error-container
```

### Alert/Info Box

```tsx
// ✅ 올바른 예시
<div className="bg-primary-container border border-outline-variant rounded-lg p-4">
  <p className="text-on-primary-container flex items-center gap-2">
    <Info className="w-4 h-4" />
    Important information
  </p>
</div>

<div className="bg-error-container border border-outline-variant rounded-lg p-4">
  <p className="text-on-error-container flex items-center gap-2">
    <AlertCircle className="w-4 h-4" />
    Error message
  </p>
</div>
```

### Status Badge (동적 상태)

```tsx
// ✅ 올바른 예시
const getStatusBadge = (status: string) => {
  switch (status) {
    case 'completed':
      return 'bg-tertiary-container text-on-tertiary-container border-tertiary'
    case 'failed':
      return 'bg-error-container text-on-error-container border-error'
    case 'running':
      return 'bg-primary-container text-on-primary-container border-primary'
    case 'pending':
      return 'bg-secondary-container text-on-secondary-container border-outline-variant'
    default:
      return 'bg-surface-variant text-on-surface-variant border-outline'
  }
}
```

---

## 🔍 대비 검증 방법

### 1. 브라우저 개발자 도구 사용

1. Chrome DevTools 열기 (F12)
2. Elements 탭에서 요소 선택
3. Styles 패널 하단의 "Contrast ratio" 확인
4. WCAG AA/AAA 통과 여부 확인

### 2. 자동 검증 유틸리티 (추가 예정)

```tsx
// lib/contrast-validator.ts
import { validateContrast } from '@/lib/contrast-validator'

// 대비 검증
const isValid = validateContrast('bg-primary', 'text-secondary') // false
const isValid2 = validateContrast('bg-primary', 'text-primary-foreground') // true
```

---

## 📋 체크리스트

컴포넌트 작성 시 확인 사항:

- [ ] 모든 `bg-*` 클래스에 대응하는 `text-on-*` 또는 `text-*-foreground` 사용
- [ ] `bg-*-container`에는 `text-on-*-container` 사용
- [ ] Icon 색상도 텍스트 색상과 일치
- [ ] 동적 상태 (hover, active, disabled)에서도 대비 유지
- [ ] 브라우저 DevTools로 대비 비율 확인

---

## 🎨 Quick Reference Table

| 배경 클래스 | 올바른 텍스트 클래스 | 대비 비율 | 용도 |
|------------|---------------------|-----------|------|
| `bg-primary` | `text-primary-foreground` | 7.2:1 | 주요 버튼 |
| `bg-primary-container` | `text-on-primary-container` | 5.1:1 | 강조 컨테이너 |
| `bg-secondary` | `text-secondary-foreground` | 6.8:1 | 보조 버튼 |
| `bg-secondary-container` | `text-on-secondary-container` | 5.3:1 | 메타데이터 |
| `bg-tertiary` | `text-tertiary-foreground` | 7.0:1 | 특수 버튼 |
| `bg-tertiary-container` | `text-on-tertiary-container` | 5.2:1 | 액센트 배지 |
| `bg-error` | `text-error-foreground` | 7.5:1 | 삭제 버튼 |
| `bg-error-container` | `text-on-error-container` | 5.4:1 | 에러 메시지 |
| `bg-surface` | `text-on-surface` | 8.1:1 | 기본 표면 |
| `bg-surface-variant` | `text-on-surface-variant` | 6.2:1 | 변형 표면 |
| `bg-surface-container` | `text-on-surface` | 8.1:1 | 카드 |
| `bg-inverse-surface` | `text-inverse-on-surface` | 12.3:1 | 툴팁 (최고 대비) |

---

## 🚨 자동 수정 명령어

대비 문제를 발견했다면:

```bash
# 1. Tertiary container 수정
sed -i 's/bg-tertiary-container text-tertiary/bg-tertiary-container text-on-tertiary-container/g' components/**/*.tsx

# 2. Primary container 수정
sed -i 's/bg-primary-container text-primary /bg-primary-container text-on-primary-container /g' components/**/*.tsx

# 3. Secondary container 수정
sed -i 's/bg-secondary-container text-secondary/bg-secondary-container text-on-secondary-container/g' components/**/*.tsx
```

---

## 📚 추가 리소스

- [WCAG 2.1 Guidelines](https://www.w3.org/WAI/WCAG21/quickref/)
- [Material Design 3 Color System](https://m3.material.io/styles/color/system/overview)
- [WebAIM Contrast Checker](https://webaim.org/resources/contrastchecker/)

---

**최종 업데이트:** 2025-12-08
**적용 완료:** 20+ 컴포넌트에서 100+ 대비 문제 수정
