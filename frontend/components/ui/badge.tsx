import * as React from 'react'
import { Slot } from '@radix-ui/react-slot'
import { cva, type VariantProps } from 'class-variance-authority'

import { cn } from '@/lib/utils'

const badgeVariants = cva(
  'inline-flex items-center justify-center rounded-md border px-2 py-0.5 text-xs font-medium w-fit whitespace-nowrap shrink-0 [&>svg]:size-3 gap-1 [&>svg]:pointer-events-none focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px] aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 aria-invalid:border-destructive transition-[color,box-shadow] overflow-hidden',
  {
    variants: {
      variant: {
        default:
          'border-transparent bg-primary text-primary-foreground [a&]:hover:bg-primary/90',
        primary:
          'border-outline-variant bg-primary-container text-on-primary-container [a&]:hover:bg-primary-container/80',
        secondary:
          'border-transparent bg-secondary text-foreground [a&]:hover:bg-secondary/90',
        secondaryContainer:
          'border-outline-variant bg-secondary-container text-on-secondary-container [a&]:hover:bg-secondary-container/80',
        tertiary:
          'border-transparent bg-tertiary text-tertiary-foreground [a&]:hover:bg-tertiary/90',
        tertiaryContainer:
          'border-outline-variant bg-tertiary-container text-on-tertiary-container [a&]:hover:bg-tertiary-container/80',
        destructive:
          'border-transparent bg-destructive text-foreground [a&]:hover:bg-destructive/90 focus-visible:ring-destructive/20 dark:focus-visible:ring-destructive/40 dark:bg-destructive/60',
        errorContainer:
          'border-outline-variant bg-error-container text-on-error-container [a&]:hover:bg-error-container/80',
        success:
          'border-transparent bg-success text-success-foreground [a&]:hover:bg-success/90',
        warning:
          'border-transparent bg-warning text-warning-foreground [a&]:hover:bg-warning/90',
        info:
          'border-transparent bg-info text-info-foreground [a&]:hover:bg-info/90',
        outline:
          'border-outline text-foreground [a&]:hover:bg-surface-container-high',
        surface:
          'border-outline-variant bg-surface-container-high text-on-surface [a&]:hover:bg-surface-container-highest',
        inverse:
          'border-transparent bg-inverse-surface text-inverse-on-surface [a&]:hover:bg-inverse-surface/90',
        fixedPrimary:
          'border-transparent bg-primary-fixed text-on-primary-fixed [a&]:hover:bg-primary-fixed/90',
        fixedPrimaryDim:
          'border-transparent bg-primary-fixed-dim text-on-primary-fixed [a&]:hover:bg-primary-fixed-dim/90',
        fixedSecondary:
          'border-transparent bg-secondary-fixed text-on-secondary-fixed [a&]:hover:bg-secondary-fixed/90',
        fixedSecondaryDim:
          'border-transparent bg-secondary-fixed-dim text-on-secondary-fixed [a&]:hover:bg-secondary-fixed-dim/90',
        fixedTertiary:
          'border-transparent bg-tertiary-fixed text-on-tertiary-fixed [a&]:hover:bg-tertiary-fixed/90',
        fixedTertiaryDim:
          'border-transparent bg-tertiary-fixed-dim text-on-tertiary-fixed [a&]:hover:bg-tertiary-fixed-dim/90',
      },
    },
    defaultVariants: {
      variant: 'default',
    },
  },
)

function Badge({
  className,
  variant,
  asChild = false,
  ...props
}: React.ComponentProps<'span'> &
  VariantProps<typeof badgeVariants> & { asChild?: boolean }) {
  const Comp = asChild ? Slot : 'span'

  return (
    <Comp
      data-slot="badge"
      className={cn(badgeVariants({ variant }), className)}
      {...props}
    />
  )
}

export { Badge, badgeVariants }
