"use client"

import { useEffect } from "react"
import { useRouter } from "next/navigation"
import { useAuth } from "@/contexts/AuthContext"
import { RefreshCw } from "lucide-react"

export default function DashboardPageProtected({ children }: { children: React.ReactNode }) {
  const { user, loading } = useAuth()
  const router = useRouter()

  useEffect(() => {
    console.log('Dashboard - Auth state:', { user, loading })

    // After loading is complete, check if user exists
    if (!loading && !user) {
      console.log('No user found, redirecting to login')
      router.push('/login')
    }
  }, [user, loading, router])

  // Show loading state
  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen bg-background">
        <div className="text-center">
          <RefreshCw className="w-8 h-8 animate-spin text-foreground mx-auto mb-4" />
          <p className="text-foreground">인증 확인 중...</p>
        </div>
      </div>
    )
  }

  // No user after loading
  if (!user) {
    return (
      <div className="flex items-center justify-center h-screen bg-background">
        <div className="text-center">
          <p className="text-foreground">로그인이 필요합니다...</p>
        </div>
      </div>
    )
  }

  // User is authenticated, render children
  return <>{children}</>
}