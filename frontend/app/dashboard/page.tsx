"use client"

import { useState, useEffect } from "react"
import { Card, CardContent } from "@/components/ui/card"
import {
  Shield,
  Image,
  Box,
  Scan,
  Users,
  Brain,
  BarChart3,
  Activity,
  Settings,
  FileText,
  Database,
  Book
} from "lucide-react"
import { DashboardLayout, DashboardMenuItem } from "@/components/layouts/dashboard-layout"
import { useAuth } from "@/contexts/AuthContext"

// Common Components
import { PageLayout, PageContent } from "@/components/common/page-layout"
// 2D Image Verification Components
import { DataManagementDB } from "@/components/2d-image-verification/2d-data-management-db"
import { AdversarialPatchGeneratorUpdated } from "@/components/2d-image-verification/2d-adversarial-patch-generator"
import { AdversarialDataGeneratorUpdated } from "@/components/2d-image-verification/2d-adversarial-data-generator"
import { AdversarialAssetManagement } from "@/components/2d-image-verification/2d-adversarial-asset-management"

// 3D Image Verification Components
import { DataGeneration3DUpdated } from "@/components/3d-image-verification/3d-data-generation"
import { CarlaAdversarialPatchGenerator } from "@/components/3d-image-verification/3d-carla-adversarial-patch"
import { AdversarialDataGenerator3D } from "@/components/3d-image-verification/3d-adversarial-data-generator"
import { AdversarialAssetManagement3D } from "@/components/3d-image-verification/3d-adversarial-asset-management"

// Evaluation Components
import { UnifiedEvaluationDashboard } from "@/components/evaluation/unified-evaluation-dashboard"
import { EvaluationRecordsDashboard } from "@/components/evaluation/evaluation-records-dashboard"

// Admin Components
import { AccountManagementDB } from "@/components/admin/account-management-db"
import { AIModelManagement } from "@/components/admin/ai-model-management"
import { Object3DManagement } from "@/components/admin/3d-object-management"
import { DBBackupRestore } from "@/components/admin/db-backup-restore"
import { LogManagement } from "@/components/admin/log-management"

// Real-time Verification Components
import { RealTimeCamera } from "@/components/real-time-verification/real-time-camera"
import { CaptureHistory } from "@/components/real-time-verification/capture-history"

export default function Dashboard() {
  const { user, loading, logout } = useAuth()
  const isAdmin = user?.role === 'admin'
  const [activeSection, setActiveSection] = useState("메인 홈")
  const [expandedMenus, setExpandedMenus] = useState<string[]>([isAdmin ? "시스템 관리" : "2D 이미지 기반 신뢰성 검증"])

  // Check authentication status
  useEffect(() => {
    if (!loading && !user) {
      logout().catch(error => {
        console.error('Failed to log out after auth check:', error)
        window.location.href = '/login'
      })
    }
  }, [user, loading, logout])

  // Handle manual download without changing section
  const handleManualDownload = () => {
    // Use the same pattern as other downloads (patches, models, etc.)
    const link = document.createElement('a')
    link.href = '/api/manual/download'
    link.download = 'AI_모델_신뢰성_검증_시스템_운용자_메뉴얼.pdf'
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  // Check localStorage for section navigation
  useEffect(() => {
    if (!user) return

    const savedSection = localStorage.getItem('dashboardSection')
    if (savedSection) {
      setActiveSection(savedSection)
      localStorage.removeItem('dashboardSection')
      // Expand parent menu if needed
      if (savedSection.includes('2d-')) {
        setExpandedMenus(prev => [...prev, "2D 이미지 기반 신뢰성 검증"])
      } else if (savedSection.includes('3d-')) {
        setExpandedMenus(prev => [...prev, "3D 이미지 기반 신뢰성 검증"])
      }
    }
  }, [user])

  // Show loading state while checking auth
  if (loading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <div className="w-12 h-12 border-4 border-primary border-t-transparent rounded-full animate-spin" />
          <p className="text-muted text-sm">로딩 중...</p>
        </div>
      </div>
    )
  }

  // Don't render if no user (redirect in progress)
  if (!user) {
    return null
  }

  const userMenuItems: DashboardMenuItem[] = [
    {
      name: "메인 홈",
      icon: BarChart3,
      content: "dashboard"
    },
    {
      name: "객체식별 AI 모델 관리",
      icon: Brain,
      content: "ai-model-management"
    },
    {
      name: "2D 이미지 기반 신뢰성 검증",
      icon: Image,
      children: [
        { name: "2D 데이터셋 관리", content: "2d-data-management-db" },
        { name: "2D 적대적 패치 생성", content: "2d-adversarial-patch" },
        { name: "2D 적대적 공격 데이터 생성", content: "2d-adversarial-data" },
        { name: "2D 적대적 자산 관리", content: "2d-adversarial-asset-management" }
      ]
    },
    {
      name: "3D 이미지 기반 신뢰성 검증",
      icon: Box,
      children: [
        { name: "3D 데이터셋 생성", content: "3d-data-generation" },
        { name: "3D 적대적 패치 생성", content: "3d-carla-adversarial-patch" },
        { name: "3D 적대적 공격 데이터 생성", content: "3d-adversarial-data" },
        { name: "3D 적대적 자산 관리", content: "3d-adversarial-asset-management" }
      ]
    },
    {
      name: "실물 객체 신뢰성 검증",
      icon: Scan,
      children: [
        { name: "실시간 카메라", content: "real-time-camera" },
        { name: "캡처 기록", content: "capture-history" }
      ]
    },
    {
      name: "성능 및 신뢰성 평가",
      icon: Shield,
      children: [
        { name: "평가 수행", content: "unified-evaluation" },
        { name: "평가 기록 관리", content: "evaluation-records" }
      ]
    },
    {
      name: "운용자 메뉴얼",
      icon: Book,
      onClick: handleManualDownload
    }
  ]

  const adminMenuItems: DashboardMenuItem[] = [
    {
      name: "메인 홈",
      icon: BarChart3,
      content: "dashboard"
    },
    {
      name: "객체식별 AI 모델 관리",
      icon: Brain,
      content: "ai-model-management"
    },
    // {
    //   name: "3D 객체 관리",
    //   icon: Box,
    //   content: "3d-object-management"
    // },
    {
      name: "평가 기록 관리",
      icon: FileText,
      content: "evaluation-records"
    },
    {
      name: "시스템 관리",
      icon: Settings,
      children: [
        { name: "계정 관리", content: "account-management" },
        { name: "DB 백업/복구", content: "db-backup" },
        { name: "로그 관리", content: "log-management" }
      ]
    }
  ]

  const menuItems = isAdmin ? adminMenuItems : userMenuItems

  const toggleMenu = (menuName: string) => {
    setExpandedMenus(prev =>
      prev.includes(menuName)
        ? prev.filter(name => name !== menuName)
        : [...prev, menuName]
    )
  }

  const renderMainContent = () => {
    switch (activeSection) {
      case "메인 홈":
        return (
          <PageLayout>
            <PageContent className="flex-1 flex items-center justify-center">
              <div className="max-w-4xl w-full space-y-8">
                <div className="text-center space-y-4">
                  <div className="inline-flex items-center justify-center w-20 h-20 rounded-full bg-primary-container mb-4">
                    <Shield className="w-10 h-10 text-primary" />
                  </div>
                  <h1 className="text-4xl font-bold text-foreground">
                    객체식별 AI 모델 신뢰성 검증 체계
                  </h1>
                  <p className="text-xl text-muted">
                    {isAdmin ? '관리자' : user?.username} 님, 환영합니다
                  </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mt-12">
                  <Card
                    className="bg-gradient-to-br from-primary-container/40 to-primary-container/20 border-primary hover:border-primary/80 transition-all cursor-pointer group"
                    onClick={() => setActiveSection(isAdmin ? "객체식별 AI 모델 관리" : "데이터 관리")}
                  >
                    <CardContent className="p-6">
                      <div className="flex items-start justify-between mb-4">
                        <div className="p-3 rounded-lg bg-primary-container group-hover:bg-primary-container transition-colors">
                          {isAdmin ? <Brain className="w-6 h-6 text-primary" /> : <Database className="w-6 h-6 text-primary" />}
                        </div>
                      </div>
                      <h3 className="text-lg font-semibold text-foreground mb-2">
                        {isAdmin ? "객체식별 AI 모델 관리" : "2D 이미지 기반 신뢰성 검증"}
                      </h3>
                      <p className="text-sm text-muted">
                        {isAdmin ? "객체식별 AI 모델을 관리합니다" : "2D 이미지 데이터셋을 업로드하고 관리합니다"}
                      </p>
                    </CardContent>
                  </Card>

                  <Card
                    className="bg-gradient-to-br from-tertiary-container/40 to-tertiary-container/20 border-secondary hover:border-tertiary/80 transition-all cursor-pointer group"
                    onClick={() => setActiveSection(isAdmin ? "평가 기록 관리" : "실시간 카메라")}
                  >
                    <CardContent className="p-6">
                      <div className="flex items-start justify-between mb-4">
                        <div className="p-3 rounded-lg bg-secondary-container group-hover:bg-secondary-container transition-colors">
                          <BarChart3 className="w-6 h-6 text-secondary" />
                        </div>
                      </div>
                      <h3 className="text-lg font-semibold text-foreground mb-2">
                        {isAdmin ? "평가 기록 관리" : "실물 객체 신뢰성 평가"}
                      </h3>
                      <p className="text-sm text-muted">
                        {isAdmin ? "모델 평가 기록을 조회하고 분석합니다" : "실시간 카메라로 객체를 탐지하고 검증합니다"}
                      </p>
                    </CardContent>
                  </Card>

                  <Card
                    className="bg-gradient-to-br from-secondary-container/40 to-secondary-container/20 border-secondary hover:border-secondary/80 transition-all cursor-pointer group"
                    onClick={() => setActiveSection(isAdmin ? "계정 관리" : "평가 수행")}
                  >
                    <CardContent className="p-6">
                      <div className="flex items-start justify-between mb-4">
                        <div className="p-3 rounded-lg bg-tertiary-container group-hover:bg-tertiary-container transition-colors">
                          {isAdmin ? <Users className="w-6 h-6 text-tertiary" /> : <Activity className="w-6 h-6 text-tertiary" />}
                        </div>
                      </div>
                      <h3 className="text-lg font-semibold text-foreground mb-2">
                        {isAdmin ? "시스템 관리" : "성능 및 신뢰성 평가"}
                      </h3>
                      <p className="text-sm text-muted">
                        {isAdmin ? "시스템 관리하고 모니터링합니다" : "AI 모델의 성능과 신뢰성을 평가합니다"}
                      </p>
                    </CardContent>
                  </Card>
                </div>
              </div>
            </PageContent>
          </PageLayout>
        )

      case "2D 데이터셋 관리":
        return <div className="h-full overflow-hidden"><DataManagementDB /></div>

      case "2D 적대적 패치 생성":
        return <div className="h-full overflow-hidden"><AdversarialPatchGeneratorUpdated /></div>

      case "2D 적대적 공격 데이터 생성":
        return <div className="h-full overflow-hidden"><AdversarialDataGeneratorUpdated /></div>

      case "2D 적대적 자산 관리":
        return <div className="h-full overflow-hidden"><AdversarialAssetManagement /></div>

      case "3D 데이터셋 생성":
        return <div className="h-full overflow-hidden"><DataGeneration3DUpdated /></div>

      case "3D 적대적 패치 생성":
        return <div className="h-full overflow-hidden"><CarlaAdversarialPatchGenerator /></div>

      case "3D 적대적 공격 데이터 생성":
        return <div className="h-full overflow-hidden"><AdversarialDataGenerator3D /></div>

      case "3D 적대적 자산 관리":
        return <div className="h-full overflow-hidden"><AdversarialAssetManagement3D /></div>

      case "평가 수행":
        return <div className="h-full overflow-hidden"><UnifiedEvaluationDashboard /></div>

      case "평가 기록 관리":
        return <div className="h-full overflow-hidden"><EvaluationRecordsDashboard /></div>

      case "계정 관리":
        return <div className="h-full overflow-hidden"><AccountManagementDB /></div>

      case "실시간 카메라":
        return <div className="h-full overflow-hidden"><RealTimeCamera /></div>

      case "캡처 기록":
        return <div className="h-full overflow-hidden"><CaptureHistory /></div>

      // Admin specific components
      case "객체식별 AI 모델 관리":
        return <div className="h-full overflow-hidden"><AIModelManagement /></div>

      case "3D 객체 관리":
        return <div className="h-full overflow-hidden"><Object3DManagement /></div>

      case "DB 백업/복구":
        return <div className="h-full overflow-hidden"><DBBackupRestore /></div>

      case "로그 관리":
        return <div className="h-full overflow-hidden"><LogManagement /></div>

      default:
        return (
          <div className="h-full flex items-center justify-center">
            <Card className="bg-surface-container/50 border-border p-8">
              <CardContent className="text-center">
                <Settings className="w-16 h-16 text-muted mx-auto mb-4" />
                <h3 className="text-xl font-semibold text-foreground mb-2">{activeSection}</h3>
                <p className="text-muted">
                  선택된 기능의 상세 구현이 준비 중입니다.
                </p>
              </CardContent>
            </Card>
          </div>
        )
    }
  }

  return (
    <DashboardLayout
      menuItems={menuItems}
      activeSection={activeSection}
      onSelectSection={(section) => setActiveSection(section)}
      expandedMenus={expandedMenus}
      onToggleMenu={toggleMenu}
    >
      {renderMainContent()}
    </DashboardLayout>
  )
}
