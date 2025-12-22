"use client"

import type React from "react"

import { useState } from "react"
import { useRouter } from "next/navigation"
import NextImage from "next/image"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Shield, Lock, User, Eye, EyeOff, Cpu, Zap, AlertCircle, ShieldAlert, Clock } from "lucide-react"
import { useAuth } from '@/contexts/AuthContext'

export function LoginFormDB() {
  const [showPassword, setShowPassword] = useState(false)
  const [formData, setFormData] = useState({
    username: "",
    password: "",
  })
  const [error, setError] = useState("")
  const [errorType, setErrorType] = useState<'error' | 'warning' | 'locked'>('error')
  const [loading, setLoading] = useState(false)
  const router = useRouter()
  const { login } = useAuth()

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    e.stopPropagation()

    console.log('🎯 LoginFormDB: handleSubmit called')
    console.log('📝 Form data:', { username: formData.username, password: '***' })

    setError("")
    setLoading(true)

    try {
      console.log('🔄 LoginFormDB: Calling login function...')
      await login(formData.username, formData.password)
      console.log('✅ LoginFormDB: Login function completed successfully')
    } catch (error) {
      console.error('❌ LoginFormDB: Login error:', error)
      const message = error instanceof Error ? error.message : '로그인 중 오류가 발생했습니다.'

      // Categorize error type based on message content
      if (message.includes('잠겨있습니다') || message.includes('잠겼습니다')) {
        setErrorType('locked')
      } else if (message.includes('남은 시도') || message.includes('세션이 종료')) {
        setErrorType('warning')
      } else {
        setErrorType('error')
      }

      setError(message)
    } finally {
      setLoading(false)
      console.log('🏁 LoginFormDB: handleSubmit finished')
    }
  }

  return (
    <Card className="w-full shadow-2xl border border-border bg-background/90 backdrop-blur-xl">
      <CardHeader className="text-center space-y-6">
        <div className="flex justify-center mb-6">
          <div className="w-32 h-32 relative flex items-center justify-center">
            <NextImage
              src="/army_logos.png"
              alt="육군 로고"
              width={128}
              height={128}
              className="object-contain"
            />
          </div>
        </div>
        <div className="text-center space-y-4 max-w-md bg-surface-container-high rounded-xl backdrop-blur-sm">
            <h1 className="text-primary drop-shadow-2xl">객체식별 AI 모델 신뢰성 검증 실증 체계</h1>
        </div>
      </CardHeader>
        <div className="flex justify-center space-x-3">
          <div className="w-2 h-2 bg-primary rounded-full animate-pulse"></div>
          <div className="w-2 h-2 bg-tertiary rounded-full animate-pulse delay-100"></div>
          <div className="w-2 h-2 bg-primary rounded-full animate-pulse delay-200"></div>
        </div>
      <CardContent className="space-y-6">
        <form onSubmit={handleSubmit} className="space-y-5">
          <div className="space-y-3">
            <Label htmlFor="username" className="text-sm font-medium flex items-center gap-2 text-foreground">
              <User className="w-4 h-4 text-primary" />
              운용자 ID
            </Label>
            <Input
              id="username"
              type="text"
              placeholder="아이디를 입력하세요"
              value={formData.username}
              onChange={(e) => setFormData({ ...formData, username: e.target.value })}
              className="h-12 border-2 border-border bg-surface-container/70 backdrop-blur-sm text-foreground placeholder:text-muted focus:border-primary focus:bg-surface-container/90 transition-all duration-300"
              required
              disabled={loading}
            />
          </div>

          <div className="space-y-3">
            <Label htmlFor="password" className="text-sm font-medium flex items-center gap-2 text-foreground">
              <Lock className="w-4 h-4 text-primary" />
              비밀번호
            </Label>
            <div className="relative">
              <Input
                id="password"
                type={showPassword ? "text" : "password"}
                placeholder="비밀번호를 입력하세요"
                value={formData.password}
                onChange={(e) => setFormData({ ...formData, password: e.target.value })}
                className="h-12 border-2 border-border bg-surface-container/70 backdrop-blur-sm text-foreground placeholder:text-muted focus:border-primary focus:bg-surface-container/90 transition-all duration-300 pr-12"
                required
                disabled={loading}
              />
              <Button
                type="button"
                variant="ghost"
                size="sm"
                className="absolute right-2 top-1/2 -translate-y-1/2 h-8 w-8 p-0 text-foreground hover:text-foreground hover:bg-surface-variant"
                onClick={() => setShowPassword(!showPassword)}
              >
                {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </Button>
            </div>
          </div>

          {error && (
            <Alert className={
              errorType === 'locked'
                ? 'bg-error-container border-error'
                : errorType === 'warning'
                ? 'bg-warning-container border-warning'
                : 'bg-error-container border-error'
            }>
              {errorType === 'locked' ? (
                <ShieldAlert className="h-5 w-5 text-error" />
              ) : errorType === 'warning' ? (
                <Clock className="h-5 w-5 text-tertiary" />
              ) : (
                <AlertCircle className="h-4 w-4 text-error" />
              )}
              <AlertTitle className={
                errorType === 'locked'
                  ? 'text-error font-semibold'
                  : errorType === 'warning'
                  ? 'text-tertiary font-semibold'
                  : 'text-error'
              }>
                {errorType === 'locked'
                  ? '계정 잠금'
                  : errorType === 'warning'
                  ? '로그인 실패'
                  : '오류'}
              </AlertTitle>
              <AlertDescription className={
                errorType === 'locked'
                  ? 'text-error-foreground'
                  : errorType === 'warning'
                  ? 'text-warning'
                  : 'text-error'
              }>
                {error}
              </AlertDescription>
              {errorType === 'locked' && (
                <AlertDescription className="text-error-foreground mt-2 text-sm">
                  보안을 위해 로그인 시도가 5회 초과되어 계정이 잠겼습니다.
                  잠금 시간이 만료된 후 다시 시도해주세요.
                </AlertDescription>
              )}
            </Alert>
          )}

          <div className="space-y-4 pt-2">
            <Button
              type="submit"
              className="w-full h-12 bg-gradient-to-r from-primary to-tertiary hover:from-primary/90 hover:to-tertiary/90 text-foreground font-semibold text-lg shadow-lg transform transition-all duration-300 hover:scale-[1.02] active:scale-[0.98] disabled:opacity-50 disabled:hover:scale-100"
              disabled={loading}
            >
              {loading ? "로그인 중..." : "로그인"}
            </Button>
          </div>
        </form>
      </CardContent>
    </Card>
  )
}
