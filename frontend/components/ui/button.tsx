import * as React from 'react'
import { Slot } from '@radix-ui/react-slot'
import { cva, type VariantProps } from 'class-variance-authority'

import { cn } from '@/lib/utils'

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium transition-all disabled:pointer-events-none disabled:opacity-50 disabled:bg-surface-dim disabled:text-on-surface-variant [&_svg]:pointer-events-none [&_svg:not([class*='size-'])]:size-4 shrink-0 [&_svg]:shrink-0 outline-none focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px] aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 aria-invalid:border-destructive",
  {
    variants: {
      variant: {
        default:
          'bg-primary text-primary-foreground shadow-xs hover:bg-primary/90',
        primary:
          'bg-primary-container text-on-primary-container shadow-xs hover:bg-primary-container/80 border border-outline-variant',
        secondary:
          'bg-secondary text-secondary-foreground shadow-xs hover:bg-secondary/90',
        secondaryContainer:
          'bg-secondary-container text-on-secondary-container shadow-xs hover:bg-secondary-container/80 border border-outline-variant',
        tertiary:
          'bg-tertiary text-tertiary-foreground shadow-xs hover:bg-tertiary/90',
        tertiaryContainer:
          'bg-tertiary-container text-on-tertiary-container shadow-xs hover:bg-tertiary-container/80 border border-outline-variant',
        destructive:
          'bg-destructive text-foreground shadow-xs hover:bg-destructive/90 focus-visible:ring-destructive/20 dark:focus-visible:ring-destructive/40 dark:bg-destructive/60',
        errorContainer:
          'bg-error-container text-on-error-container shadow-xs hover:bg-error-container/80 border border-outline-variant',
        outline:
          'border border-outline bg-transparent text-foreground shadow-xs hover:bg-surface-variant hover:text-foreground',
        outlineVariant:
          'border border-outline-variant bg-transparent text-on-surface-variant hover:bg-surface-container-high',
        tonal:
          'bg-surface-container-high text-foreground shadow-xs hover:bg-surface-container-highest',
        ghost:
          'text-foreground hover:bg-surface-container-high hover:text-foreground',
        inverse:
          'bg-inverse-surface text-inverse-on-surface shadow-xs hover:bg-inverse-surface/90',
        link: 'text-primary underline-offset-4 hover:underline',
      },
      size: {
        default: 'h-10 px-4 py-2 has-[>svg]:px-3',
        sm: 'h-10 rounded-md gap-1.5 px-3 has-[>svg]:px-2.5',
        lg: 'h-10 rounded-md px-6 has-[>svg]:px-4',
        icon: 'size-10',
      },
    },
    defaultVariants: {
      variant: 'default',
      size: 'default',
    },
  },
)

const Button = React.forwardRef<
  HTMLButtonElement,
  React.ComponentProps<'button'> &
    VariantProps<typeof buttonVariants> & {
      asChild?: boolean
    }
>(({ className, variant, size, asChild = false, ...props }, ref) => {
  const Comp = asChild ? Slot : 'button'

  return (
    <Comp
      ref={ref}
      data-slot="button"
      className={cn(buttonVariants({ variant, size, className }))}
      {...props}
    />
  )
})
Button.displayName = 'Button'

export { Button, buttonVariants }
