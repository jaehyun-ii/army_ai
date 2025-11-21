"use client"

import { useState, useEffect, useMemo } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { ScrollArea } from "@/components/ui/scroll-area"
import { EvaluationDetailView } from "./EvaluationDetailView"
import { useModels, useEvaluations } from "@/hooks/api"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import {
  FileText,
  Search,
  Filter,
  Eye,
  Download,
  BarChart3,
  TrendingUp,
  TrendingDown,
  Shield,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Calendar,
  Brain,
  Target,
  Award,
  Image as ImageIcon,
  Box,
  ExternalLink,
  Activity,
  Loader2,
  RefreshCw
} from "lucide-react"
import { apiClient } from "@/lib/api-client"
import { toast } from "sonner"

interface EvaluationRun {
  id: string
  name: string
  description?: string
  phase: string
  status: string
  model_id: string
  base_dataset_id?: string
  attack_dataset_id?: string
  metrics_summary?: {
    map: number
    map50: number
    map75: number
    precision: number
    recall: number
    ar_100: number
  }
  created_at: string
  started_at?: string
  ended_at?: string
}

interface Model {
  id: string
  name: string
  model_type: string
}

interface Dataset {
  id: string
  name: string
  description?: string
}

export function EvaluationRecordsDashboard() {
  // Use custom hooks
  const { data: modelsData } = useModels(0, 100)
  const { data: evaluationsData, refetch: refetchEvaluations } = useEvaluations()

  const [activeTab, setActiveTab] = useState("all")
  const [searchQuery, setSearchQuery] = useState("")
  const [filterStatus, setFilterStatus] = useState("all")
  const [filterPhase, setFilterPhase] = useState("all")
  const [sortBy, setSortBy] = useState("timestamp")

  // Detail view state
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null)

  // Data states
  const evaluationRuns = useMemo(() => evaluationsData || [], [evaluationsData])
  const models = useMemo(() => {
    if (!modelsData) return {}
    return modelsData.reduce((acc: Record<string, Model>, model: any) => {
      acc[model.id] = model
      return acc
    }, {})
  }, [modelsData])
  const [datasets, setDatasets] = useState<Record<string, Dataset>>({})
  const [attackDatasets, setAttackDatasets] = useState<Record<string, Dataset>>({})
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const totalEvaluations = evaluationRuns.length
  const [currentPage, setCurrentPage] = useState(1)

  // Load datasets on mount
  useEffect(() => {
    loadDatasetsData()
  }, [])

  const loadDatasetsData = async () => {
    setLoading(true)
    try {
      await Promise.all([
        loadDatasets(),
        loadAttackDatasets()
      ])
    } catch (error) {
      console.error("Failed to load data:", error)
      toast.error("데이터 로딩 실패")
    } finally {
      setLoading(false)
    }
  }

  const loadDatasets = async () => {
    try {
      const response: any = await apiClient.getDatasets()
      console.log("Datasets response:", response)
      const datasetArray = response.items || response || []
      const datasetMap = datasetArray.reduce((acc: Record<string, Dataset>, dataset: Dataset) => {
        acc[dataset.id] = dataset
        return acc
      }, {})
      setDatasets(datasetMap)
    } catch (error) {
      console.error("Failed to load datasets:", error)
      toast.error("데이터셋 목록을 불러오는데 실패했습니다")
    }
  }

  const loadAttackDatasets = async () => {
    try {
      const response: any = await apiClient.listAttackDatasets()
      console.log("Attack datasets response:", response)
      const datasetArray = response.items || response || []
      const attackDatasetMap = datasetArray.reduce((acc: Record<string, Dataset>, dataset: Dataset) => {
        acc[dataset.id] = dataset
        return acc
      }, {})
      setAttackDatasets(attackDatasetMap)
    } catch (error) {
      console.error("Failed to load attack datasets:", error)
      toast.error("공격 데이터셋 목록을 불러오는데 실패했습니다")
    }
  }

  const handleRefresh = async () => {
    setRefreshing(true)
    await Promise.all([
      refetchEvaluations(),
      loadDatasetsData()
    ])
    setRefreshing(false)
    toast.success("데이터 새로고침 완료")
  }

  const filteredRecords = evaluationRuns.filter(record => {
    const modelName = models[record.model_id]?.name || ""
    const matchesSearch = record.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         modelName.toLowerCase().includes(searchQuery.toLowerCase())
    const matchesStatus = filterStatus === "all" || record.status === filterStatus
    const matchesPhase = filterPhase === "all" || record.phase === filterPhase
    const matchesTab = activeTab === "all" ||
                      (activeTab === "completed" && record.status === "completed") ||
                      (activeTab === "running" && record.status === "running") ||
                      (activeTab === "failed" && record.status === "failed")

    return matchesSearch && matchesStatus && matchesPhase && matchesTab
  })

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'completed': return 'bg-green-900/40 text-green-300 border-green-500/40'
      case 'failed': return 'bg-red-900/40 text-red-300 border-red-500/40'
      case 'running': return 'bg-blue-900/40 text-blue-300 border-blue-500/40'
      case 'pending': return 'bg-yellow-900/40 text-yellow-300 border-yellow-500/40'
      default: return 'bg-slate-900/40 text-slate-300 border-slate-500/40'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return <CheckCircle className="w-4 h-4 text-green-400" />
      case 'failed': return <XCircle className="w-4 h-4 text-red-400" />
      case 'running': return <div className="w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
      case 'pending': return <AlertTriangle className="w-4 h-4 text-yellow-400" />
      default: return <AlertTriangle className="w-4 h-4 text-slate-400" />
    }
  }

  const getPhaseBadge = (phase: string) => {
    switch (phase) {
      case 'pre_attack': return 'bg-blue-900/40 text-blue-300 border-blue-500/40'
      case 'post_attack': return 'bg-red-900/40 text-red-300 border-red-500/40'
      default: return 'bg-slate-900/40 text-slate-300 border-slate-500/40'
    }
  }

  const getPhaseLabel = (phase: string) => {
    switch (phase) {
      case 'pre_attack': return '기준 데이터'
      case 'post_attack': return '공격 데이터'
      default: return phase
    }
  }

  const totalRecords = evaluationRuns.length
  const completedRecords = evaluationRuns.filter(r => r.status === 'completed').length
  const failedRecords = evaluationRuns.filter(r => r.status === 'failed').length
  const runningRecords = evaluationRuns.filter(r => r.status === 'running').length
  const runsWithMetrics = evaluationRuns.filter(r => r.metrics_summary?.map50)
  const averageMap50 = runsWithMetrics.length > 0
    ? (runsWithMetrics.reduce((acc, r) => acc + (r.metrics_summary?.map50 || 0), 0) / runsWithMetrics.length * 100).toFixed(1)
    : '0'

  const formatDate = (dateString: string) => {
    if (!dateString) return '-'
    const date = new Date(dateString)
    return date.toLocaleString('ko-KR', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  const handleViewDetail = (record: EvaluationRun) => {
    setSelectedRunId(record.id)
  }

  const handleBackToList = () => {
    setSelectedRunId(null)
  }

  // Show detail view if a run is selected
  if (selectedRunId) {
    return <EvaluationDetailView runId={selectedRunId} onBack={handleBackToList} />
  }

  if (loading) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center space-y-4">
          <Loader2 className="w-12 h-12 animate-spin text-blue-400 mx-auto" />
          <p className="text-slate-400">평가 기록을 불러오는 중...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="h-full flex flex-col gap-2">
      {/* Header */}
      <div className="bg-gradient-to-r from-slate-800/80 to-slate-900/80 rounded-xl p-3 border border-white/10 shadow-xl flex-shrink-0">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-lg lg:text-xl font-bold text-white flex items-center gap-2">
              <FileText className="w-6 h-6 text-cyan-400" />
              평가 기록 관리
            </h1>
            <p className="text-xs text-slate-400">AI 모델 평가 결과 및 기록 통합 관리</p>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={handleRefresh}
            disabled={refreshing}
            className="flex items-center gap-2 bg-slate-700/50 border-white/10 text-white hover:bg-slate-600/50"
          >
            <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} />
            새로고침
          </Button>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 space-y-6 overflow-auto">
        {/* Statistics Cards */}
        <Card className="bg-slate-800/50 border-white/10">
          <CardContent className="pt-6">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
              <div className="bg-slate-700/30 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-2">
                  <BarChart3 className="w-4 h-4 text-blue-400" />
                  <span className="text-slate-400 text-sm">총 평가</span>
                </div>
                <div className="text-2xl font-bold text-white">{totalEvaluations}</div>
                {totalEvaluations > 100 && (
                  <p className="text-xs text-slate-500 mt-1">
                    페이지 {currentPage} / {Math.ceil(totalEvaluations / 100)}
                  </p>
                )}
              </div>
              <div className="bg-slate-700/30 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-2">
                  <CheckCircle className="w-4 h-4 text-green-400" />
                  <span className="text-slate-400 text-sm">완료</span>
                </div>
                <div className="text-2xl font-bold text-green-400">{completedRecords}</div>
              </div>
              <div className="bg-slate-700/30 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-2">
                  <Activity className="w-4 h-4 text-yellow-400" />
                  <span className="text-slate-400 text-sm">실행 중</span>
                </div>
                <div className="text-2xl font-bold text-yellow-400">{runningRecords}</div>
              </div>
              <div className="bg-slate-700/30 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-2">
                  <TrendingUp className="w-4 h-4 text-purple-400" />
                  <span className="text-slate-400 text-sm">평균 mAP@50</span>
                </div>
                <div className="text-2xl font-bold text-purple-400">{averageMap50}%</div>
              </div>
            </div>

            {/* Filters */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400 w-4 h-4" />
                <Input
                  placeholder="평가 이름 또는 모델 검색..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="bg-slate-700/50 border-white/10 text-white pl-10"
                />
              </div>
              <Select value={filterStatus} onValueChange={setFilterStatus}>
                <SelectTrigger className="bg-slate-700/50 border-white/10 text-white">
                  <SelectValue placeholder="상태 필터" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">모든 상태</SelectItem>
                  <SelectItem value="completed">완료</SelectItem>
                  <SelectItem value="running">실행 중</SelectItem>
                  <SelectItem value="failed">실패</SelectItem>
                  <SelectItem value="pending">대기 중</SelectItem>
                </SelectContent>
              </Select>
              <Select value={filterPhase} onValueChange={setFilterPhase}>
                <SelectTrigger className="bg-slate-700/50 border-white/10 text-white">
                  <SelectValue placeholder="평가 유형" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">모든 유형</SelectItem>
                  <SelectItem value="pre_attack">기준 데이터</SelectItem>
                  <SelectItem value="post_attack">공격 데이터</SelectItem>
                </SelectContent>
              </Select>
              <Button variant="outline" className="flex items-center gap-2 bg-slate-700/50 border-white/10 text-white hover:bg-slate-600/50">
                <Download className="w-4 h-4" />
                내보내기
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Tabs and Records List */}
        <Card className="bg-slate-800/50 border-white/10">
          <CardHeader className="pb-3">
            <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
              <TabsList className="grid w-full grid-cols-4 bg-slate-800/50">
                <TabsTrigger value="all" className="text-white">전체</TabsTrigger>
                <TabsTrigger value="completed" className="text-white flex items-center gap-2">
                  <CheckCircle className="w-4 h-4" />
                  완료
                </TabsTrigger>
                <TabsTrigger value="running" className="text-white flex items-center gap-2">
                  <Activity className="w-4 h-4" />
                  실행 중
                </TabsTrigger>
                <TabsTrigger value="failed" className="text-white flex items-center gap-2">
                  <XCircle className="w-4 h-4" />
                  실패
                </TabsTrigger>
              </TabsList>
            </Tabs>
          </CardHeader>

          <CardContent>
            {filteredRecords.length === 0 ? (
              <div className="text-center py-12">
                <FileText className="w-16 h-16 text-slate-600 mx-auto mb-4" />
                <p className="text-slate-400">평가 기록이 없습니다</p>
                <p className="text-slate-500 text-sm mt-2">새로운 평가를 생성해보세요</p>
              </div>
            ) : (
              <ScrollArea className="h-[600px]">
                <Table>
                  <TableHeader>
                    <TableRow className="border-white/10 hover:bg-transparent">
                      <TableHead className="text-slate-300">평가 이름</TableHead>
                      <TableHead className="text-slate-300">모델</TableHead>
                      <TableHead className="text-slate-300">유형</TableHead>
                      <TableHead className="text-slate-300">상태</TableHead>
                      <TableHead className="text-slate-300">mAP@50</TableHead>
                      <TableHead className="text-slate-300">생성 시간</TableHead>
                      <TableHead className="text-slate-300">작업</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {filteredRecords.map((record) => (
                      <TableRow
                        key={record.id}
                        className="border-white/5 hover:bg-slate-700/30 cursor-pointer"
                        onClick={() => handleViewDetail(record)}
                      >
                        <TableCell className="font-medium text-white">
                          <div>
                            <div className="font-semibold">{record.name}</div>
                            {record.description && (
                              <div className="text-xs text-slate-400 line-clamp-1">{record.description}</div>
                            )}
                          </div>
                        </TableCell>
                        <TableCell>
                          <div className="flex items-center gap-2">
                            <Brain className="w-4 h-4 text-blue-400" />
                            <span className="text-slate-300">{models[record.model_id]?.name || 'Unknown'}</span>
                          </div>
                        </TableCell>
                        <TableCell>
                          <Badge className={getPhaseBadge(record.phase)}>
                            {getPhaseLabel(record.phase)}
                          </Badge>
                        </TableCell>
                        <TableCell>
                          <Badge className={getStatusBadge(record.status)}>
                            <div className="flex items-center gap-2">
                              {getStatusIcon(record.status)}
                              {record.status}
                            </div>
                          </Badge>
                        </TableCell>
                        <TableCell>
                          {record.metrics_summary?.map50 ? (
                            <span className="text-green-400 font-semibold">
                              {(record.metrics_summary.map50 * 100).toFixed(1)}%
                            </span>
                          ) : (
                            <span className="text-slate-500">-</span>
                          )}
                        </TableCell>
                        <TableCell className="text-slate-400 text-sm">
                          {formatDate(record.created_at)}
                        </TableCell>
                        <TableCell>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={(e) => {
                              e.stopPropagation()
                              handleViewDetail(record)
                            }}
                            className="text-blue-400 hover:text-blue-300 hover:bg-blue-900/20"
                          >
                            <Eye className="w-4 h-4" />
                          </Button>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </ScrollArea>
            )}

            {/* Pagination */}
            {totalEvaluations > 100 && (
              <div className="flex items-center justify-between mt-4 pt-4 border-t border-white/10">
                <div className="text-sm text-slate-400">
                  표시 중: {(currentPage - 1) * 100 + 1} - {Math.min(currentPage * 100, totalEvaluations)} / 총 {totalEvaluations}
                </div>
                <div className="flex items-center gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setCurrentPage(currentPage - 1)}
                    disabled={currentPage === 1}
                    className="bg-slate-700/50 border-white/10 text-white hover:bg-slate-600/50"
                  >
                    이전
                  </Button>
                  <span className="text-sm text-slate-400">
                    페이지 {currentPage} / {Math.ceil(totalEvaluations / 100)}
                  </span>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setCurrentPage(currentPage + 1)}
                    disabled={currentPage >= Math.ceil(totalEvaluations / 100)}
                    className="bg-slate-700/50 border-white/10 text-white hover:bg-slate-600/50"
                  >
                    다음
                  </Button>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

    </div>
  )
}
