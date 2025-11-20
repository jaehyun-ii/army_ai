"use client"

import type React from "react"
import { useState } from "react"
import { useRouter } from "next/navigation"
import NextImage from "next/image"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Shield, Lock, User, Eye, EyeOff, Cpu, Zap, AlertCircle } from "lucide-react"
import { styles, combineStyles } from "@/lib/styles"
import { useLoading } from "@/hooks/use-loading"

interface UnifiedLoginFormProps {
  useDatabase?: boolean // true: DB 연동, false: 목업 데이터
  onSuccess?: () => void
  className?: string
}

/**
 * 통합 로그인 폼 컴포넌트
 * DB 연동과 목업 데이터 모두 지원
 */
export function UnifiedLoginForm({
  useDatabase = false,
  onSuccess,
  className
}: UnifiedLoginFormProps) {
  const [showPassword, setShowPassword] = useState(false)
  const [formData, setFormData] = useState({
    username: "",
    password: "",
  })
  const [error, setError] = useState("")
  const { isLoading, startLoading, stopLoading } = useLoading()
  const router = useRouter()

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError("")
    startLoading()

    try {
      if (useDatabase) {
        // DB 연동 로그인
        const response = await fetch('/api/auth/login', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            username: formData.username,
            password: formData.password
          }),
        })

        const data = await response.json()

        if (response.ok && data.success) {
          console.log('로그인 성공')
          if (onSuccess) {
            onSuccess()
          } else {
            setTimeout(() => {
              window.location.replace('/dashboard')
            }, 100)
          }
        } else {
          setError(data.error || "로그인에 실패했습니다.")
        }
      } else {
        // 목업 데이터 로그인
        if (formData.username === "admin" && formData.password === "adminpw") {
          console.log("로그인 성공")
          if (onSuccess) {
            onSuccess()
          } else {
            router.push("/dashboard")
          }
        } else {
          setError("잘못된 사용자 정보입니다.")
        }
      }
    } catch (err) {
      console.error('Login error:', err)
      setError("로그인 중 오류가 발생했습니다.")
    } finally {
      stopLoading()
    }
  }

  return (
    <Card className={combineStyles("w-full shadow-2xl", styles.card.base, className)}>
      <CardHeader className="text-center space-y-6 pb-8">
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

        <div className="space-y-3">
          <CardTitle className="text-3xl font-bold text-white">대한민국 육군</CardTitle>
          <CardTitle className="text-xl font-semibold text-slate-50 flex items-center justify-center gap-2">
            <Zap className="w-5 h-5 text-accent" />
            AI 통합 관제 시스템
          </CardTitle>
          <CardDescription className="text-slate-100 text-base">
            차세대 보안 인증으로 시스템에 접속하십시오
          </CardDescription>
        </div>

        <div className="flex justify-center space-x-3">
          <div className="w-2 h-2 bg-primary rounded-full animate-pulse"></div>
          <div className="w-2 h-2 bg-accent rounded-full animate-pulse delay-100"></div>
          <div className="w-2 h-2 bg-primary rounded-full animate-pulse delay-200"></div>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        {error && (
          <Alert className="bg-red-900/20 border-red-500/30">
            <AlertCircle className="h-4 w-4 text-red-400" />
            <AlertDescription className="text-red-400">
              {error}
            </AlertDescription>
          </Alert>
        )}

        <form onSubmit={handleSubmit} className="space-y-5">
          <div className="space-y-3">
            <Label htmlFor="username" className="text-sm font-medium flex items-center gap-2 text-slate-50">
              <User className="w-4 h-4 text-primary" />
              사용자 ID
            </Label>
            <Input
              id="username"
              type="text"
              placeholder="군번 또는 사용자 ID를 입력하세요"
              value={formData.username}
              onChange={(e) => setFormData({ ...formData, username: e.target.value })}
              className={combineStyles("h-12 border-2", styles.input.base, styles.input.focus)}
              required
              disabled={isLoading}
            />
          </div>

          <div className="space-y-3">
            <Label htmlFor="password" className="text-sm font-medium flex items-center gap-2 text-slate-50">
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
                className={combineStyles("h-12 border-2 pr-12", styles.input.base, styles.input.focus)}
                required
                disabled={isLoading}
              />
              <Button
                type="button"
                variant="ghost"
                size="sm"
                className="absolute right-2 top-1/2 -translate-y-1/2 h-8 w-8 p-0 text-slate-200 hover:text-white hover:bg-white/10"
                onClick={() => setShowPassword(!showPassword)}
                disabled={isLoading}
              >
                {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </Button>
            </div>
          </div>

          <Button
            type="submit"
            disabled={isLoading}
            className={combineStyles(
              "w-full h-12 text-lg font-semibold shadow-lg hover:shadow-xl transform hover:scale-[1.02]",
              styles.button.primary
            )}
          >
            {isLoading ? (
              <>
                <span className="animate-spin mr-2">⏳</span>
                인증 중...
              </>
            ) : (
              <>
                <Shield className="w-5 h-5 mr-2" />
                보안 로그인
              </>
            )}
          </Button>
        </form>

        <div className="bg-slate-800/80 border border-white/10 rounded-xl p-4 backdrop-blur-sm">
          <div className="flex items-start gap-3">
            <Shield className="w-5 h-5 text-accent mt-0.5 flex-shrink-0" />
            <div className="text-sm space-y-1">
              <p className="font-medium text-accent">보안 알림</p>
              <p className="text-slate-100 leading-relaxed">
                본 시스템은 국가기밀보호법에 의해 보호됩니다. 무단 접근 시 법적 처벌을 받을 수 있습니다.
              </p>
            </div>
          </div>
        </div>

        <div className="text-center space-y-3 pt-4 border-t border-white/10">
          <div className="flex justify-center space-x-6 text-sm">
            <button className="text-blue-300 hover:text-blue-200 font-medium transition-colors duration-200 hover:underline">
              비밀번호 찾기
            </button>
            <span className="text-slate-400">|</span>
            <button className="text-blue-300 hover:text-blue-200 font-medium transition-colors duration-200 hover:underline">
              계정 문의
            </button>
          </div>
          <p className="text-xs text-slate-200">시스템 관리: 육군본부 정보통신단 © 2024</p>
        </div>
      </CardContent>
    </Card>
  )
}