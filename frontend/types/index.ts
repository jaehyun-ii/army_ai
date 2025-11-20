/**
 * 타입 정의 모음 - 중앙 export
 */

export * from './common'

// 컴포넌트별 타입도 여기서 export
export type { Column, DataTableProps } from '@/components/common/data-table'

// 유틸리티 타입
export type ValueOf<T> = T[keyof T]
export type Nullable<T> = T | null
export type Optional<T> = T | undefined

// 이벤트 핸들러 타입
export type ClickHandler = (event: React.MouseEvent) => void
export type ChangeHandler<T = HTMLInputElement> = (event: React.ChangeEvent<T>) => void
export type SubmitHandler = (event: React.FormEvent) => void

// 컴포넌트 Props 기본 타입
export interface BaseComponentProps {
  className?: string
  children?: React.ReactNode
}

// 폼 관련 타입
export interface FormField {
  name: string
  label: string
  type: 'text' | 'email' | 'password' | 'number' | 'select' | 'textarea' | 'checkbox'
  value?: any
  required?: boolean
  disabled?: boolean
  placeholder?: string
  options?: Array<{ value: string; label: string }>
  validation?: FormValidation
}

export interface FormValidation {
  required?: boolean
  minLength?: number
  maxLength?: number
  pattern?: RegExp
  custom?: (value: any) => boolean | string
}

// 상태 관련 타입
export type LoadingState = 'idle' | 'loading' | 'success' | 'error'
export type AsyncState<T> = {
  data: T | null
  loading: boolean
  error: Error | null
}

// 테마 관련 타입
export type ColorVariant = 'primary' | 'secondary' | 'success' | 'warning' | 'error' | 'info'
export type SizeVariant = 'xs' | 'sm' | 'md' | 'lg' | 'xl'

// 권한 관련 타입
export interface Permission {
  resource: string
  actions: string[]
}

export interface RolePermissions {
  role: string
  permissions: Permission[]
}