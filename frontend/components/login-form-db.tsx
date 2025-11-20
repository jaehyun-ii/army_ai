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
import { useAuth } from '@/contexts/AuthContext'

export function LoginFormDB() {
  const [showPassword, setShowPassword] = useState(false)
  const [formData, setFormData] = useState({
    username: "",
    password: "",
  })
  const [error, setError] = useState("")
  const [loading, setLoading] = useState(false)
  const router = useRouter()
  const { login } = useAuth()

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError("")
    setLoading(true)

    try {
      await login(formData.username, formData.password)
    } catch (error) {
      console.error('Login error:', error)
      const message = error instanceof Error ? error.message : '로그인 중 오류가 발생했습니다.'
      setError(message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <Card className="w-full shadow-2xl border border-white/20 bg-slate-900/90 backdrop-blur-xl">
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
        <div className="text-center space-y-4 max-w-md bg-slate-900/70 rounded-xl backdrop-blur-sm">
            <h1 className="text-xl font-bold text-white drop-shadow-2xl">객체식별 모델 신뢰성 검증 실증 체계</h1>
        </div>
      </CardHeader>
        <div className="flex justify-center space-x-3">
          <div className="w-2 h-2 bg-primary rounded-full animate-pulse"></div>
          <div className="w-2 h-2 bg-accent rounded-full animate-pulse delay-100"></div>
          <div className="w-2 h-2 bg-primary rounded-full animate-pulse delay-200"></div>
        </div>
      <CardContent className="space-y-6">
        <form onSubmit={handleSubmit} className="space-y-5">
          <div className="space-y-3">
            <Label htmlFor="username" className="text-sm font-medium flex items-center gap-2 text-slate-50">
              <User className="w-4 h-4 text-primary" />
              사용자 ID
            </Label>
            <Input
              id="username"
              type="text"
              placeholder="아이디를 입력하세요"
              value={formData.username}
              onChange={(e) => setFormData({ ...formData, username: e.target.value })}
              className="h-12 border-2 border-white/20 bg-slate-800/70 backdrop-blur-sm text-white placeholder:text-slate-300 focus:border-primary focus:bg-slate-800/90 transition-all duration-300"
              required
              disabled={loading}
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
                className="h-12 border-2 border-white/20 bg-slate-800/70 backdrop-blur-sm text-white placeholder:text-slate-300 focus:border-primary focus:bg-slate-800/90 transition-all duration-300 pr-12"
                required
                disabled={loading}
              />
              <Button
                type="button"
                variant="ghost"
                size="sm"
                className="absolute right-2 top-1/2 -translate-y-1/2 h-8 w-8 p-0 text-slate-200 hover:text-white hover:bg-white/10"
                onClick={() => setShowPassword(!showPassword)}
              >
                {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </Button>
            </div>
          </div>

          {error && (
            <Alert className="bg-red-900/20 border-red-500/30">
              <AlertCircle className="h-4 w-4 text-red-400" />
              <AlertDescription className="text-red-300">
                {error}
              </AlertDescription>
            </Alert>
          )}

          <div className="space-y-4 pt-2">
            <Button
              type="submit"
              className="w-full h-12 bg-gradient-to-r from-primary to-accent hover:from-primary/90 hover:to-accent/90 text-white font-semibold text-lg shadow-lg transform transition-all duration-300 hover:scale-[1.02] active:scale-[0.98] disabled:opacity-50 disabled:hover:scale-100"
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
