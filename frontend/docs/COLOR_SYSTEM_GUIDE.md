# Full Spectrum Color System Guide

## Overview

Your application now uses **ALL 51 Material Design 3 color variables** from `light.css`, providing a rich, semantic color system. This guide shows you how to use each color in your UI.

---

## Color System Architecture

### 1. **Surface Elevation System** (5 Levels)

Material Design 3 provides a sophisticated elevation system using surface colors:

```tsx
// From lowest to highest elevation:
className="bg-surface-container-lowest"  // Flat, lowest depth
className="bg-surface-container-low"     // Slightly elevated
className="bg-surface-container"         // Base elevation (default)
className="bg-surface-container-high"    // Elevated elements
className="bg-surface-container-highest" // Floating/modal elements
```

**Usage Examples:**

```tsx
// Modal dialogs (highest elevation)
<div className="bg-surface-container-highest">

// Cards and elevated content
<Card variant="elevated">  // Uses surface-container-high

// Base content cards
<Card variant="default">   // Uses surface-container

// Sunken/recessed areas (like text input backgrounds)
<div className="bg-surface-container-low">

// Flat backgrounds
<div className="bg-surface-container-lowest">
```

**Additional Surface Utilities:**
- `bg-surface-dim` - Dimmed surface (for inactive areas)
- `bg-surface-bright` - Bright surface (for active areas)
- `bg-surface-variant` - Variant surface (for subtle differentiation)

---

### 2. **Primary Color System** (Brand Identity)

Your primary color (teal/cyan) has 8 variations:

```css
--color-primary                      /* Main brand color - buttons, links */
--color-primary-foreground           /* Text on primary */
--color-primary-container            /* Soft primary background */
--color-on-primary-container         /* Text on primary container */
--color-primary-fixed               /* Theme-independent primary */
--color-primary-fixed-dim           /* Dimmed fixed primary */
--color-on-primary-fixed            /* Text on fixed primary */
--color-on-primary-fixed-variant    /* Secondary text on fixed */
```

**Usage Examples:**

```tsx
// Bold primary actions
<Button variant="default">Submit</Button>  // bg-primary

// Soft primary highlights
<Card variant="primary">                   // bg-primary-container
  <h3 className="text-on-primary-container">Featured</h3>
</Card>

// Primary badges
<Badge variant="primary">New</Badge>       // primary-container bg

// Primary highlights that work in any theme
<div className="bg-primary-fixed text-on-primary-fixed">
  Important Notice
</div>
```

---

### 3. **Secondary Color System** (Supporting Elements)

Your secondary color (blue-gray) supports UI hierarchy:

```css
--color-secondary
--color-secondary-foreground
--color-secondary-container
--color-on-secondary-container
--color-secondary-fixed
--color-secondary-fixed-dim
--color-on-secondary-fixed
--color-on-secondary-fixed-variant
```

**Usage Examples:**

```tsx
// Secondary actions
<Button variant="secondary">Cancel</Button>

// Secondary containers for grouped content
<Card variant="secondary">
  <p className="text-on-secondary-container">Related content</p>
</Card>

// Secondary badges for metadata
<Badge variant="secondaryContainer">Draft</Badge>

// Pending/in-progress states
<div className={styles.status.pending}>Processing...</div>
```

---

### 4. **Tertiary Color System** (Accents & Special Features)

Your tertiary color (purple) provides accent highlights:

```css
--color-tertiary
--color-tertiary-foreground
--color-tertiary-container
--color-on-tertiary-container
--color-tertiary-fixed
--color-tertiary-fixed-dim
--color-on-tertiary-fixed
--color-on-tertiary-fixed-variant
```

**Usage Examples:**

```tsx
// Accent buttons for special features
<Button variant="tertiary">AI Generate</Button>

// Accent cards for premium features
<Card variant="tertiary">
  <h3 className="text-on-tertiary-container">Premium Feature</h3>
</Card>

// Trend indicators
<StatCard
  variant="default"
  trendColor="text-tertiary"
  trend="+15%"
/>

// Highlight important metrics
<div className={styles.highlight.tertiary}>
  Special announcement
</div>
```

---

### 5. **Error/Destructive System**

```css
--color-error
--color-error-foreground
--color-error-container
--color-on-error-container
```

**Usage Examples:**

```tsx
// Destructive actions
<Button variant="destructive">Delete</Button>

// Error messages
<Card variant="error">
  <p className="text-on-error-container">Error: Invalid input</p>
</Card>

// Error badges
<Badge variant="errorContainer">Failed</Badge>

// Error alerts
<div className={styles.alert.variants.error}>
  Operation failed
</div>
```

---

### 6. **Inverse Color System** (High Contrast)

For tooltips, snackbars, and high-contrast elements:

```css
--color-inverse-surface
--color-inverse-on-surface
--color-inverse-primary
```

**Usage Examples:**

```tsx
// Tooltips
<div className="bg-inverse-surface text-inverse-on-surface">
  Helpful tip
</div>

// Snackbars/toasts
<div className={styles.tooltip.default}>
  Notification message
</div>

// High-contrast buttons
<Button variant="inverse">Emergency Action</Button>

// Inverse badges for dark-on-light contrast
<Badge variant="inverse">Live</Badge>
```

---

### 7. **Outline System** (Borders & Dividers)

```css
--color-outline          /* Standard borders */
--color-outline-variant  /* Subtle borders */
```

**Usage Examples:**

```tsx
// Standard borders
<div className="border border-outline">

// Subtle borders for less emphasis
<div className="border border-outline-variant">

// Subtle dividers
<hr className={styles.divider.subtle} />

// Heavy dividers
<hr className={styles.divider.heavy} />
```

---

## Complete Component Examples

### Enhanced StatCard

```tsx
// Primary stat (most important metric)
<StatCard
  title="Total Revenue"
  value="$124,500"
  variant="primary"
  iconColor="text-primary"
  valueColor="text-on-primary-container"
  trend="+12%"
  trendColor="text-success"
  icon={DollarSign}
  trendIcon={TrendingUp}
/>

// Secondary stat (supporting metric)
<StatCard
  title="Active Users"
  value="1,234"
  variant="secondary"
  iconColor="text-secondary"
  valueColor="text-on-secondary-container"
/>

// Tertiary stat (accent/special metric)
<StatCard
  title="AI Predictions"
  value="89%"
  variant="tertiary"
  iconColor="text-tertiary"
  valueColor="text-on-tertiary-container"
/>

// Error stat (problematic metric)
<StatCard
  title="Failed Jobs"
  value="3"
  variant="error"
  iconColor="text-error"
  valueColor="text-on-error-container"
/>

// Elevated stat (important, floating)
<StatCard
  title="Today's Goals"
  value="5/7"
  variant="elevated"
  iconColor="text-primary"
/>
```

### Enhanced Buttons

```tsx
// Primary action (most important)
<Button variant="default">Create New</Button>

// Primary container (softer primary)
<Button variant="primary">Learn More</Button>

// Secondary action
<Button variant="secondary">View Details</Button>

// Secondary container (softer secondary)
<Button variant="secondaryContainer">Options</Button>

// Tertiary action (accent)
<Button variant="tertiary">Special Feature</Button>

// Outline (neutral)
<Button variant="outline">Cancel</Button>

// Outline variant (subtle)
<Button variant="outlineVariant">More</Button>

// Ghost (minimal)
<Button variant="ghost">Skip</Button>

// Tonal (filled but subtle)
<Button variant="tonal">Settings</Button>

// Inverse (high contrast)
<Button variant="inverse">Emergency</Button>
```

### Enhanced Badges

```tsx
<Badge variant="primary">Featured</Badge>
<Badge variant="secondaryContainer">Draft</Badge>
<Badge variant="tertiaryContainer">Premium</Badge>
<Badge variant="errorContainer">Error</Badge>
<Badge variant="success">Active</Badge>
<Badge variant="warning">Pending</Badge>
<Badge variant="info">Info</Badge>
<Badge variant="surface">Neutral</Badge>
<Badge variant="inverse">Live</Badge>
<Badge variant="outline">Basic</Badge>
```

### Enhanced Cards with Elevation

```tsx
// Default card (base elevation)
<Card variant="default">
  Standard content
</Card>

// Elevated card (medium elevation)
<Card variant="elevated">
  Important content
</Card>

// Floating card (highest elevation - modals, dialogs)
<Card variant="floating">
  Modal content
</Card>

// Sunken card (recessed, lower than base)
<Card variant="sunken">
  Background content
</Card>

// Flat card (no elevation)
<Card variant="flat">
  Minimal content
</Card>

// Primary themed card
<Card variant="primary">
  <h3 className="text-on-primary-container">Featured Section</h3>
</Card>

// Secondary themed card
<Card variant="secondary">
  <h3 className="text-on-secondary-container">Related Content</h3>
</Card>

// Tertiary themed card
<Card variant="tertiary">
  <h3 className="text-on-tertiary-container">Special Feature</h3>
</Card>

// Error themed card
<Card variant="error">
  <h3 className="text-on-error-container">Error Details</h3>
</Card>
```

---

## Using styles.ts Utilities

### Card Styles

```tsx
import { styles } from "@/lib/styles"

// Base card
<div className={styles.card.base}>

// Elevated card
<div className={styles.card.elevated}>

// Floating card (highest)
<div className={styles.card.floating}>

// Interactive card (hover effects)
<div className={styles.card.interactive}>
```

### Button Styles

```tsx
// Primary container button
<button className={`${styles.button.base} ${styles.button.primaryContainer}`}>

// Tertiary container button
<button className={`${styles.button.base} ${styles.button.tertiaryContainer}`}>

// Inverse button
<button className={`${styles.button.base} ${styles.button.inverse}`}>
```

### Chip Styles

```tsx
// Filled chips
<span className={`${styles.chip.base} ${styles.chip.filled.primary}`}>

// Outlined chips
<span className={`${styles.chip.base} ${styles.chip.outlined.secondary}`}>

// Tonal chips (container colors)
<span className={`${styles.chip.base} ${styles.chip.tonal.tertiary}`}>
```

### Highlight Styles

```tsx
// Primary highlight
<div className={styles.highlight.primary}>
  Important message
</div>

// Error highlight
<div className={styles.highlight.error}>
  Error message
</div>

// Success highlight
<div className={styles.highlight.success}>
  Success message
</div>
```

### Status Indicators

```tsx
<div className={styles.status.active}>Active</div>
<div className={styles.status.inactive}>Inactive</div>
<div className={styles.status.success}>Completed</div>
<div className={styles.status.warning}>Warning</div>
<div className={styles.status.error}>Error</div>
<div className={styles.status.pending}>Pending</div>
```

### Alert Styles

```tsx
<div className={`${styles.alert.base} ${styles.alert.variants.primary}`}>
  <h4>Primary Alert</h4>
  <p>Important information</p>
</div>

<div className={`${styles.alert.base} ${styles.alert.variants.error}`}>
  <h4>Error Alert</h4>
  <p>Something went wrong</p>
</div>
```

---

## Color Role Assignment Guidelines

### When to Use Each Color:

**Primary:**
- Main CTAs (Call-to-Action buttons)
- Navigation highlights
- Brand identity elements
- Links
- Active states

**Secondary:**
- Supporting buttons (Cancel, Back)
- Metadata displays
- Secondary navigation
- Grouped related content
- Draft/pending states

**Tertiary:**
- Accent features
- Special/premium functionality
- Trend indicators
- Decorative highlights
- AI/advanced features

**Error/Destructive:**
- Delete/remove actions
- Error messages
- Failed states
- Critical warnings
- Validation errors

**Surface Elevation:**
- **Lowest:** Page backgrounds, flat areas
- **Low:** Input fields, recessed areas
- **Base:** Standard cards, content containers
- **High:** Elevated cards, dropdowns
- **Highest:** Modals, dialogs, tooltips

**Inverse:**
- Tooltips
- Snackbars/toasts
- High-contrast overlays
- Dark-themed elements in light mode

---

## Color Contrast & Accessibility

Always use the correct "on-" color for text on colored backgrounds:

```tsx
// ✅ CORRECT - Proper contrast
<div className="bg-primary-container text-on-primary-container">
  Readable text
</div>

// ❌ WRONG - Poor contrast
<div className="bg-primary-container text-primary">
  Hard to read
</div>
```

**Correct Pairings:**

- `bg-primary` ➜ `text-primary-foreground` (or `text-on-primary`)
- `bg-primary-container` ➜ `text-on-primary-container`
- `bg-secondary` ➜ `text-secondary-foreground`
- `bg-secondary-container` ➜ `text-on-secondary-container`
- `bg-tertiary` ➜ `text-tertiary-foreground`
- `bg-tertiary-container` ➜ `text-on-tertiary-container`
- `bg-error` ➜ `text-error-foreground`
- `bg-error-container` ➜ `text-on-error-container`
- `bg-surface` ➜ `text-on-surface`
- `bg-surface-variant` ➜ `text-on-surface-variant`

---

## Quick Reference Table

| Use Case | Component | Variant/Class |
|----------|-----------|---------------|
| Primary action | Button | `variant="default"` |
| Soft primary | Button | `variant="primary"` |
| Cancel/back | Button | `variant="secondaryContainer"` |
| Special feature | Button | `variant="tertiary"` |
| Delete | Button | `variant="destructive"` |
| Featured card | Card | `variant="primary"` |
| Standard card | Card | `variant="default"` |
| Elevated card | Card | `variant="elevated"` |
| Modal/dialog | Card | `variant="floating"` |
| Success status | Badge | `variant="success"` |
| Error status | Badge | `variant="errorContainer"` |
| Premium badge | Badge | `variant="tertiaryContainer"` |
| Tooltip | div | `className={styles.tooltip.default}` |
| Active status | div | `className={styles.status.active}` |
| Error alert | div | `className={styles.alert.variants.error}` |

---

## Before vs After

**Before (Limited palette):**
```tsx
// Only using primary and surface
<Button className="bg-primary">Action</Button>
<Card className="bg-surface-container">Content</Card>
```

**After (Full spectrum):**
```tsx
// Rich, semantic color usage
<Button variant="primary">Primary Action</Button>
<Button variant="tertiary">Special Feature</Button>
<Card variant="elevated">Important Content</Card>
<Card variant="primary">Featured Section</Card>
<Badge variant="tertiaryContainer">Premium</Badge>
<div className={styles.highlight.secondary}>Related Info</div>
```

---

## Summary

You now have access to:

- **51 color variables** (up from 15)
- **5-level elevation system** for depth
- **4 semantic color families** (primary, secondary, tertiary, error) with 8 variations each
- **Inverse colors** for high contrast
- **Enhanced components** (Button, Badge, Card) with 10+ variants
- **New style utilities** in `styles.ts` (chips, highlights, status, alerts)
- **Proper accessibility** with semantic "on-" colors

Use this system to create a rich, visually engaging UI that maintains consistency and accessibility throughout your application.
