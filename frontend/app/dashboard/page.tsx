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
  FileArchive,
  ScrollText,
  Book
} from "lucide-react"
import { DashboardLayout, DashboardMenuItem } from "@/components/layouts/dashboard-layout"
import { useAuth } from "@/contexts/AuthContext"

// Common Components
import { StatCard } from "@/components/common/stat-card"
import { PageLayout, PageContent } from "@/components/common/page-layout"
import { DataTable, StatusBadge } from "@/components/common/data-table"
// 2D Image Verification Components
import { DataManagementDB } from "@/components/2d-image-verification/2d-data-management-db"
import { AdversarialPatchGeneratorUpdated } from "@/components/2d-image-verification/2d-adversarial-patch-generator"
import { AdversarialDataGeneratorUpdated } from "@/components/2d-image-verification/2d-adversarial-data-generator"
import { AdversarialAssetManagement } from "@/components/2d-image-verification/2d-adversarial-asset-management"

// 3D Image Verification Components
import { DataGeneration3DUpdated } from "@/components/3d-image-verification/3d-data-generation"
import { AdversarialPatch3D } from "@/components/3d-image-verification/3d-adversarial-patch"
import { AdversarialDataGenerator3D } from "@/components/3d-image-verification/3d-adversarial-data-generator"

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
  const { user } = useAuth()
  const isAdmin = user?.role === 'admin'
  const [activeSection, setActiveSection] = useState("대시보드")
  const [expandedMenus, setExpandedMenus] = useState<string[]>([isAdmin ? "시스템 관리" : "2D 이미지 기반 신뢰성 검증"])

  // Check localStorage for section navigation
  useEffect(() => {
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
  }, [])

  const userMenuItems: DashboardMenuItem[] = [
    {
      name: "대시보드",
      icon: BarChart3,
      content: "dashboard"
    },
    {
      name: "2D 이미지 기반 신뢰성 검증",
      icon: Image,
      children: [
        { name: "데이터 관리", content: "2d-data-management" },
        { name: "2D 적대적 패치 생성", content: "2d-adversarial-patch" },
        { name: "2D 적대적 공격 데이터 생성", content: "2d-adversarial-data" },
        { name: "적대적 자산 관리", content: "2d-adversarial-asset-management" }
      ]
    },
    // {
    //   name: "3D 이미지 기반 신뢰성 검증",
    //   icon: Box,
    //   children: [
    //     { name: "3D 데이터 생성", content: "3d-data-generation" },
    //     { name: "3D 적대적 패치 생성", content: "3d-adversarial-patch" },
    //     { name: "3D 적대적 공격 데이터 생성", content: "3d-adversarial-data" }
    //   ]
    // },
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
      name: "사용자 메뉴얼",
      icon: Book,
      content: "user-manual"
    }
  ]

  const adminMenuItems: DashboardMenuItem[] = [
    {
      name: "대시보드",
      icon: BarChart3,
      content: "dashboard"
    },
    {
      name: "객체식별 AI 모델 관리",
      icon: Brain,
      content: "ai-model-management"
    },
    {
      name: "3D 객체 관리",
      icon: Box,
      content: "3d-object-management"
    },
    {
      name: "평가 기록 관리",
      icon: FileText,
      content: "evaluation-records"
    },
    {
      name: "시스템 관리",
      icon: Settings,
      children: [
        { name: "계정관리", content: "account-management" },
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
      case "대시보드":
        const recentTasks = [
          { task: "YOLOv8 객체 탐지 모델 평가", status: "완료", time: "2024-09-28 14:30", accuracy: "95.2%", model: "YOLOv8" },
          { task: "적대적 패치 공격 내성 테스트", status: "진행중", time: "2024-09-28 15:45", accuracy: "89.7%", model: "ResNet-50" },
          { task: "3D 환경 객체 인식 검증", status: "완료", time: "2024-09-28 11:20", accuracy: "89.7%", model: "EfficientNet" },
          { task: "실시간 카메라 성능 평가", status: "대기", time: "2024-09-29 09:00", accuracy: "92.4%", model: "YOLOv5" },
          { task: "야간 환경 신뢰성 측정", status: "완료", time: "2024-09-28 08:15", accuracy: "87.4%", model: "Faster R-CNN" }
        ]

        const taskColumns = [
          { key: "task", header: "평가 수행 이름", render: (value: string) => (
            <span className="text-white font-medium text-sm">{value}</span>
          )},
          { key: "model", header: "모델", width: "w-32", render: (value: string) => (
            <span className="text-blue-300 text-xs font-mono bg-blue-900/30 px-2 py-1 rounded">{value}</span>
          )},
          { key: "time", header: "시간", width: "w-28", render: (value: string) => (
            <span className="text-slate-300 text-sm">{value}</span>
          )},
          { key: "accuracy", header: "정확도", width: "w-24", align: "right" as const, render: (value: string) => (
            <span className={`font-semibold ${value === '측정중' || value === '-' ? 'text-slate-400' : 'text-green-400'}`}>{value}</span>
          )}
        ]

        return (
          <PageLayout>

            <PageContent className="flex-1">
              <DataTable
                title="최근 검증 활동"
                description="최근 수행된 AI 모델 신뢰성 검증 작업 내역"
                data={recentTasks}
                columns={taskColumns}
                className="h-full flex flex-col"
              />
            </PageContent>
          </PageLayout>
        )

      case "데이터 관리":
        return <div className="h-full overflow-hidden"><DataManagementDB /></div>

      case "2D 적대적 패치 생성":
        return <div className="h-full overflow-hidden"><AdversarialPatchGeneratorUpdated /></div>

      case "2D 적대적 공격 데이터 생성":
        return <div className="h-full overflow-hidden"><AdversarialDataGeneratorUpdated /></div>

      case "적대적 자산 관리":
        return <div className="h-full overflow-hidden"><AdversarialAssetManagement /></div>

      case "3D 데이터 생성":
        return <div className="h-full overflow-hidden"><DataGeneration3DUpdated /></div>

      case "3D 적대적 패치 생성":
        return <div className="h-full overflow-hidden"><AdversarialPatch3D /></div>

      case "3D 적대적 공격 데이터 생성":
        return <div className="h-full overflow-hidden"><AdversarialDataGenerator3D /></div>

      case "평가 수행":
        return <div className="h-full overflow-hidden"><UnifiedEvaluationDashboard /></div>

      case "평가 기록 관리":
        return <div className="h-full overflow-hidden"><EvaluationRecordsDashboard /></div>

      case "계정관리":
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
            <Card className="bg-slate-800/50 border-white/10 p-8">
              <CardContent className="text-center">
                <Settings className="w-16 h-16 text-slate-400 mx-auto mb-4" />
                <h3 className="text-xl font-semibold text-white mb-2">{activeSection}</h3>
                <p className="text-slate-400">
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
