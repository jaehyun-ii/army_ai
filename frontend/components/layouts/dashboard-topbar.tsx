"use client"

import { Button } from '@/components/ui/button'
import { User, LogOut } from 'lucide-react'
import { useAuth } from '@/contexts/AuthContext'

export function DashboardTopbar() {
  const { user, logout } = useAuth()

  return (
    <header className="flex-shrink-0 bg-slate-900/95 backdrop-blur-sm border-b border-white/10 z-40">
      <div className="px-4 py-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="flex flex-col justify-center">
              <h1 className="text-lg font-bold text-white">객체식별 AI 모델 신뢰성 검증 체계</h1>
              <p className="text-xs text-slate-400">AI Model Reliability Verification Framework</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-gradient-to-br from-slate-700 to-slate-600 rounded-full flex items-center justify-center ring-2 ring-white/20 shadow-md">
                <User className="w-4 h-4 text-white" />
              </div>
              <div>
                <p className="text-white text-sm font-semibold drop-shadow-sm">{user?.name || '사용자'}</p>
              </div>
            </div>
            <Button
              onClick={logout}
              size="sm"
              className="bg-red-600 hover:bg-red-700 text-white border-0"
            >
              <LogOut className="w-4 h-4 mr-2" />
              로그아웃
            </Button>
          </div>
        </div>
      </div>
    </header>
  )
}
