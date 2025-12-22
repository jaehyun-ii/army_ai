"use client"

import { Button } from '@/components/ui/button'
import { User, LogOut, Shield, UserCircle } from 'lucide-react'
import { useAuth } from '@/contexts/AuthContext'
import { Badge } from '@/components/ui/badge'

export function DashboardTopbar() {
  const { user, logout, loading } = useAuth()

  const getRoleDisplay = (role: string | undefined) => {
    if (role === 'admin') {
      return { label: '관리자', color: 'bg-tertiary-container text-on-tertiary-container border-tertiary' }
    }
    return { label: '운용자', color: 'bg-secondary-container text-on-secondary-container border-secondary' }
  }

  const roleInfo = getRoleDisplay(user?.role)
  const isAdmin = user?.role === 'admin'

  return (
    <header className="flex-shrink-0 z-40 bg-[var(--surface)]/85 backdrop-blur-xl border-b border-border shadow-[0_8px_30px_-12px_rgba(0,0,0,0.25)]">
      <div className="px-6 py-3">
        <div className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-xl bg-primary/10 text-primary flex items-center justify-center ring-1 ring-primary/20 shadow-sm">
              <Shield className="w-5 h-5" />
            </div>
            <div className="flex flex-col">
              <h1 className="text-lg font-semibold text-foreground leading-tight">객체식별 AI 모델 신뢰성 검증 체계</h1>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {loading ? (
              <div className="flex items-center gap-3">
                <div className="w-9 h-9 bg-[var(--surface-strong)] rounded-full animate-pulse" />
                <div className="flex flex-col gap-1">
                  <div className="h-3 w-20 bg-[var(--surface-strong)] rounded animate-pulse" />
                  <div className="h-2 w-14 bg-[var(--surface-strong)] rounded animate-pulse" />
                </div>
              </div>
            ) : user ? (
              <>
                <div className="flex items-center gap-3 rounded-full border border-border bg-[var(--surface)]/85 px-3 py-2 shadow-[0_12px_32px_-18px_rgba(0,0,0,0.32)]">
                  <div className={`w-8 h-8 ${isAdmin ? 'bg-tertiary-container text-on-tertiary-container' : 'bg-primary-container text-on-primary-container'} rounded-full flex items-center justify-center`}>
                    {isAdmin ? (
                      <Shield className="w-4 h-4" />
                    ) : (
                      <UserCircle className="w-4 h-4" />
                    )}
                  </div>
                  <div className="flex items-center gap-2">
                    <p className="text-on-surface text-base font-semibold leading-tight">{user.username}</p>
                    <Badge className={`text-xs ${roleInfo.color} px-2 py-0.5 w-fit leading-tight rounded-full`}>
                      {roleInfo.label}
                    </Badge>
                  </div>
                </div>
                <Button
                  onClick={logout}
                  size="sm"
                  variant="ghost"
                  className="text-destructive hover:bg-destructive hover:text-destructive-foreground"
                >
                  <LogOut className="w-4 h-4 mr-2" />
                  로그아웃
                </Button>
              </>
            ) : (
              <Button
                onClick={logout}
                size="sm"
                variant="ghost"
                className="text-destructive hover:bg-destructive hover:text-destructive-foreground"
              >
                <LogOut className="w-4 h-4 mr-2" />
                로그아웃
              </Button>
            )}
          </div>
        </div>
      </div>
    </header>
  )
}
