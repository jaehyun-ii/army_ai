"use client"

import { Button } from '@/components/ui/button'
import { User, LogOut, Shield, UserCircle } from 'lucide-react'
import { useAuth } from '@/contexts/AuthContext'
import { Badge } from '@/components/ui/badge'

export function DashboardTopbar() {
  const { user, logout } = useAuth()

  const getRoleDisplay = (role: string | undefined) => {
    if (role === 'admin') {
      return { label: '관리자', color: 'bg-amber-500/20 text-amber-400 border-amber-500/30' }
    }
    return { label: '사용자', color: 'bg-blue-500/20 text-blue-400 border-blue-500/30' }
  }

  const roleInfo = getRoleDisplay(user?.role)
  const isAdmin = user?.role === 'admin'

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
              <div className={`w-8 h-8 ${isAdmin ? 'bg-gradient-to-br from-amber-600 to-amber-500' : 'bg-gradient-to-br from-slate-700 to-slate-600'} rounded-full flex items-center justify-center ring-2 ring-white/20 shadow-md`}>
                {isAdmin ? (
                  <Shield className="w-4 h-4 text-white" />
                ) : (
                  <UserCircle className="w-4 h-4 text-white" />
                )}
              </div>
              <div className="flex flex-col gap-0.5">
                <p className="text-white text-sm font-semibold drop-shadow-sm leading-tight">{user?.username || 'Guest'}</p>
                <Badge variant="outline" className={`text-[10px] ${roleInfo.color} border px-1.5 py-0 w-fit leading-tight`}>
                  {roleInfo.label}
                </Badge>
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
