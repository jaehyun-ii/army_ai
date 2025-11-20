"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import {
  Shield,
  Brain,
  Calendar,
  TrendingDown,
  Eye,
  FileText,
  Search,
  Filter,
  ChevronRight,
  Image as ImageIcon,
  AlertTriangle,
  CheckCircle,
  XCircle,
  BarChart3,
  Download,
  ExternalLink,
  Zap,
  Target,
  Award,
  ShieldAlert,
  Info
} from "lucide-react"

interface EvaluationRecord {
  id: string
  name: string
  model: string
  modelType: string
  timestamp: string
  normalDataset: {
    name: string
    imageCount: number
    ap50: number
    f1Score: number
  }
  adversarialDatasets: Array<{
    name: string
    attackMethod: string
    imageCount: number
    ap50: number
    f1Score: number
    performanceDrop: number
    successfulAttacks: number
  }>
  reliabilityLevel: 'high' | 'medium' | 'low'
  reliabilityScore: number
  totalImages: number
  successfulAttacks: number
}

interface AttackImage {
  id: string
  originalImage: string
  attackedImage: string
  originalPrediction: string
  attackPrediction: string
  confidence: {
    original: number
    attacked: number
  }
  attackMethod: string
}

export function ReliabilityEvaluationList() {
  const [selectedRecord, setSelectedRecord] = useState<EvaluationRecord | null>(null)
  const [showDetailModal, setShowDetailModal] = useState(false)
  const [selectedAttackImages, setSelectedAttackImages] = useState<AttackImage[]>([])
  const [selectedImage, setSelectedImage] = useState<AttackImage | null>(null)
  const [showImageModal, setShowImageModal] = useState(false)
  const [searchQuery, setSearchQuery] = useState("")
  const [filterModel, setFilterModel] = useState("all")
  const [filterReliability, setFilterReliability] = useState("all")

  // 샘플 평가 기록 데이터
  const [evaluationRecords] = useState<EvaluationRecord[]>([
    {
      id: "eval_001",
      name: "YOLO v8 종합 신뢰성 평가",
      model: "YOLOv8",
      modelType: "Object Detection",
      timestamp: "2024-01-20 14:30",
      normalDataset: {
        name: "자동차 분류 데이터셋",
        imageCount: 15847,
        ap50: 0.924,
        f1Score: 0.891
      },
      adversarialDatasets: [
        {
          name: "FGSM 공격 데이터셋",
          attackMethod: "FGSM",
          imageCount: 15847,
          ap50: 0.612,
          f1Score: 0.578,
          performanceDrop: 33.8,
          successfulAttacks: 5230
        },
        {
          name: "PGD 공격 데이터셋",
          attackMethod: "PGD",
          imageCount: 15847,
          ap50: 0.485,
          f1Score: 0.442,
          performanceDrop: 47.5,
          successfulAttacks: 7520
        }
      ],
      reliabilityLevel: 'low',
      reliabilityScore: 42.3,
      totalImages: 47541,
      successfulAttacks: 12750
    },
    {
      id: "eval_002",
      name: "ResNet-50 군사 장비 평가",
      model: "ResNet-50",
      modelType: "Classification",
      timestamp: "2024-01-19 09:15",
      normalDataset: {
        name: "군사 장비 데이터셋",
        imageCount: 8932,
        ap50: 0.887,
        f1Score: 0.856
      },
      adversarialDatasets: [
        {
          name: "Patch 공격 데이터셋",
          attackMethod: "Adversarial Patch",
          imageCount: 8932,
          ap50: 0.712,
          f1Score: 0.685,
          performanceDrop: 19.7,
          successfulAttacks: 1760
        }
      ],
      reliabilityLevel: 'medium',
      reliabilityScore: 68.5,
      totalImages: 17864,
      successfulAttacks: 1760
    },
    {
      id: "eval_003",
      name: "EfficientNet 드론 탐지 평가",
      model: "EfficientNet",
      modelType: "Classification",
      timestamp: "2024-01-18 16:45",
      normalDataset: {
        name: "드론 탐지 데이터셋",
        imageCount: 5621,
        ap50: 0.903,
        f1Score: 0.875
      },
      adversarialDatasets: [
        {
          name: "Noise 공격 데이터셋",
          attackMethod: "Gaussian Noise",
          imageCount: 5621,
          ap50: 0.812,
          f1Score: 0.788,
          performanceDrop: 10.1,
          successfulAttacks: 568
        }
      ],
      reliabilityLevel: 'high',
      reliabilityScore: 85.2,
      totalImages: 11242,
      successfulAttacks: 568
    }
  ])

  // 샘플 공격 성공 이미지 데이터
  const sampleAttackImages: AttackImage[] = [
    {
      id: "img_001",
      originalImage: "/api/placeholder/300/300",
      attackedImage: "/api/placeholder/300/300",
      originalPrediction: "car",
      attackPrediction: "truck",
      confidence: { original: 0.95, attacked: 0.87 },
      attackMethod: "FGSM"
    },
    {
      id: "img_002",
      originalImage: "/api/placeholder/300/300",
      attackedImage: "/api/placeholder/300/300",
      originalPrediction: "person",
      attackPrediction: "background",
      confidence: { original: 0.92, attacked: 0.76 },
      attackMethod: "PGD"
    },
    {
      id: "img_003",
      originalImage: "/api/placeholder/300/300",
      attackedImage: "/api/placeholder/300/300",
      originalPrediction: "tank",
      attackPrediction: "truck",
      confidence: { original: 0.88, attacked: 0.65 },
      attackMethod: "Adversarial Patch"
    }
  ]

  const getReliabilityBadge = (level: string) => {
    switch(level) {
      case 'high':
        return <Badge className="bg-green-500/20 text-green-400 border-green-500/30">높음</Badge>
      case 'medium':
        return <Badge className="bg-amber-500/20 text-white border-amber-500/30">보통</Badge>
      case 'low':
        return <Badge className="bg-red-500/20 text-red-400 border-red-500/30">낮음</Badge>
      default:
        return null
    }
  }

  const handleViewDetail = (record: EvaluationRecord) => {
    setSelectedRecord(record)
    setSelectedAttackImages(sampleAttackImages)
    setShowDetailModal(true)
  }

  const handleViewImage = (image: AttackImage) => {
    setSelectedImage(image)
    setShowImageModal(true)
  }

  // 필터링된 레코드
  const filteredRecords = evaluationRecords.filter(record => {
    const matchesSearch = record.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         record.model.toLowerCase().includes(searchQuery.toLowerCase())
    const matchesModel = filterModel === "all" || record.model.toLowerCase() === filterModel.toLowerCase()
    const matchesReliability = filterReliability === "all" || record.reliabilityLevel === filterReliability

    return matchesSearch && matchesModel && matchesReliability
  })

  return (
    <div className="space-y-6">
      {/* 상단 요약 카드 */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="bg-slate-800/50 border-white/10">
          <CardHeader className="pb-3">
            <CardTitle className="text-lg flex items-center gap-2">
              <FileText className="w-5 h-5 text-blue-400" />
              총 평가 기록
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-white">{evaluationRecords.length}</p>
            <p className="text-sm text-slate-400 mt-1">저장된 평가 결과</p>
          </CardContent>
        </Card>

        <Card className="bg-slate-800/50 border-white/10">
          <CardHeader className="pb-3">
            <CardTitle className="text-lg flex items-center gap-2">
              <ShieldAlert className="w-5 h-5 text-red-400" />
              낮은 신뢰성
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-red-400">
              {evaluationRecords.filter(r => r.reliabilityLevel === 'low').length}
            </p>
            <p className="text-sm text-slate-400 mt-1">취약한 모델</p>
          </CardContent>
        </Card>

        <Card className="bg-slate-800/50 border-white/10">
          <CardHeader className="pb-3">
            <CardTitle className="text-lg flex items-center gap-2">
              <Target className="w-5 h-5 text-white" />
              평균 공격 성공률
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-white">
              {(evaluationRecords.reduce((acc, r) => acc + (r.successfulAttacks / r.totalImages), 0) / evaluationRecords.length * 100).toFixed(1)}%
            </p>
            <p className="text-sm text-slate-400 mt-1">전체 평균</p>
          </CardContent>
        </Card>

        <Card className="bg-slate-800/50 border-white/10">
          <CardHeader className="pb-3">
            <CardTitle className="text-lg flex items-center gap-2">
              <CheckCircle className="w-5 h-5 text-green-400" />
              높은 신뢰성
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-green-400">
              {evaluationRecords.filter(r => r.reliabilityLevel === 'high').length}
            </p>
            <p className="text-sm text-slate-400 mt-1">강건한 모델</p>
          </CardContent>
        </Card>
      </div>

      {/* 필터 및 검색 */}
      <Card className="bg-slate-800/50 border-white/10">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Filter className="w-5 h-5" />
            검색 및 필터
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex gap-4">
            <div className="flex-1">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400 w-4 h-4" />
                <Input
                  placeholder="평가 이름 또는 모델 검색..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-10 bg-slate-900/50"
                />
              </div>
            </div>
            <Select value={filterModel} onValueChange={setFilterModel}>
              <SelectTrigger className="w-[180px]">
                <SelectValue placeholder="모델 필터" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">모든 모델</SelectItem>
                <SelectItem value="yolov8">YOLO v8</SelectItem>
                <SelectItem value="resnet-50">ResNet-50</SelectItem>
                <SelectItem value="efficientnet">EfficientNet</SelectItem>
              </SelectContent>
            </Select>
            <Select value={filterReliability} onValueChange={setFilterReliability}>
              <SelectTrigger className="w-[150px]">
                <SelectValue placeholder="신뢰성 필터" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">모든 수준</SelectItem>
                <SelectItem value="high">높음</SelectItem>
                <SelectItem value="medium">보통</SelectItem>
                <SelectItem value="low">낮음</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* 평가 목록 테이블 */}
      <Card className="bg-slate-800/50 border-white/10">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="w-5 h-5" />
            성능 평가 목록
          </CardTitle>
          <CardDescription>클릭하여 상세 결과 및 공격 성공 이미지를 확인하세요</CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="text-slate-400">평가 이름</TableHead>
                <TableHead className="text-slate-400">모델</TableHead>
                <TableHead className="text-slate-400">평가 일시</TableHead>
                <TableHead className="text-slate-400">기본 AP50</TableHead>
                <TableHead className="text-slate-400">평균 성능 하락</TableHead>
                <TableHead className="text-slate-400">신뢰성</TableHead>
                <TableHead className="text-slate-400">공격 성공률</TableHead>
                <TableHead className="text-slate-400"></TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredRecords.map(record => {
                const avgPerformanceDrop = record.adversarialDatasets.reduce(
                  (acc, d) => acc + d.performanceDrop, 0
                ) / record.adversarialDatasets.length

                return (
                  <TableRow
                    key={record.id}
                    className="cursor-pointer hover:bg-slate-800/30"
                    onClick={() => handleViewDetail(record)}
                  >
                    <TableCell className="font-medium text-white">
                      {record.name}
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center gap-2">
                        <Brain className="w-4 h-4 text-blue-400" />
                        <span className="text-white">{record.model}</span>
                        <Badge variant="outline" className="text-xs">
                          {record.modelType}
                        </Badge>
                      </div>
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center gap-1 text-slate-400">
                        <Calendar className="w-3 h-3" />
                        <span className="text-sm">{record.timestamp}</span>
                      </div>
                    </TableCell>
                    <TableCell>
                      <span className="text-green-400 font-medium">
                        {(record.normalDataset.ap50 * 100).toFixed(1)}%
                      </span>
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center gap-2">
                        <TrendingDown className="w-4 h-4 text-red-400" />
                        <span className="text-red-400 font-medium">
                          -{avgPerformanceDrop.toFixed(1)}%
                        </span>
                      </div>
                    </TableCell>
                    <TableCell>
                      {getReliabilityBadge(record.reliabilityLevel)}
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center gap-2">
                        <Progress
                          value={(record.successfulAttacks / record.totalImages) * 100}
                          className="w-20 h-2"
                        />
                        <span className="text-sm text-slate-400">
                          {((record.successfulAttacks / record.totalImages) * 100).toFixed(1)}%
                        </span>
                      </div>
                    </TableCell>
                    <TableCell>
                      <ChevronRight className="w-4 h-4 text-slate-400" />
                    </TableCell>
                  </TableRow>
                )
              })}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      {/* 상세 결과 모달 */}
      <Dialog open={showDetailModal} onOpenChange={setShowDetailModal}>
        <DialogContent className="sm:max-w-[900px] bg-slate-900 border-slate-700 max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 text-white">
              <Shield className="w-5 h-5" />
              {selectedRecord?.name}
            </DialogTitle>
            <DialogDescription className="text-slate-400">
              성능 평가 상세 결과 및 공격 성공 이미지
            </DialogDescription>
          </DialogHeader>

          {selectedRecord && (
            <Tabs defaultValue="summary" className="w-full">
              <TabsList className="grid w-full grid-cols-3">
                <TabsTrigger value="summary">평가 요약</TabsTrigger>
                <TabsTrigger value="performance">성능 지표</TabsTrigger>
                <TabsTrigger value="attacks">공격 성공 이미지</TabsTrigger>
              </TabsList>

              <TabsContent value="summary" className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <Card className="bg-slate-800/50 border-white/10">
                    <CardHeader className="pb-3">
                      <CardTitle className="text-sm">모델 정보</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-slate-400">모델:</span>
                        <span className="text-white">{selectedRecord.model}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-400">유형:</span>
                        <span className="text-white">{selectedRecord.modelType}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-400">평가 시간:</span>
                        <span className="text-white">{selectedRecord.timestamp}</span>
                      </div>
                    </CardContent>
                  </Card>

                  <Card className="bg-slate-800/50 border-white/10">
                    <CardHeader className="pb-3">
                      <CardTitle className="text-sm">신뢰성 평가</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="text-slate-400">신뢰성 수준:</span>
                        {getReliabilityBadge(selectedRecord.reliabilityLevel)}
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-400">신뢰성 점수:</span>
                        <span className="text-white font-bold">
                          {selectedRecord.reliabilityScore.toFixed(1)}/100
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-400">공격 성공률:</span>
                        <span className="text-red-400 font-bold">
                          {((selectedRecord.successfulAttacks / selectedRecord.totalImages) * 100).toFixed(1)}%
                        </span>
                      </div>
                    </CardContent>
                  </Card>
                </div>

                <Card className="bg-slate-800/50 border-white/10">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm">평가 데이터셋</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="p-3 bg-slate-900/50 rounded-lg">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-white font-medium">기본 데이터셋</span>
                        <Badge >
                          {selectedRecord.normalDataset.imageCount.toLocaleString()}개
                        </Badge>
                      </div>
                      <div className="text-sm text-slate-400">
                        {selectedRecord.normalDataset.name}
                      </div>
                    </div>
                    {selectedRecord.adversarialDatasets.map((dataset, index) => (
                      <div key={index} className="p-3 bg-slate-900/50 rounded-lg">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-white font-medium">적대적 데이터셋 {index + 1}</span>
                          <div className="flex gap-2">
                            <Badge variant="outline">{dataset.attackMethod}</Badge>
                            <Badge >
                              {dataset.imageCount.toLocaleString()}개
                            </Badge>
                          </div>
                        </div>
                        <div className="text-sm text-slate-400">
                          {dataset.name}
                        </div>
                      </div>
                    ))}
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="performance" className="space-y-4">
                <Card className="bg-slate-800/50 border-white/10">
                  <CardHeader>
                    <CardTitle className="text-sm">기본 데이터셋 성능</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="p-3 bg-slate-900/50 rounded-lg">
                        <p className="text-xs text-slate-400 mb-1">AP50</p>
                        <p className="text-xl font-bold text-green-400">
                          {(selectedRecord.normalDataset.ap50 * 100).toFixed(1)}%
                        </p>
                      </div>
                      <div className="p-3 bg-slate-900/50 rounded-lg">
                        <p className="text-xs text-slate-400 mb-1">F1-Score</p>
                        <p className="text-xl font-bold text-blue-400">
                          {(selectedRecord.normalDataset.f1Score * 100).toFixed(1)}%
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {selectedRecord.adversarialDatasets.map((dataset, index) => (
                  <Card key={index} className="bg-slate-800/50 border-white/10">
                    <CardHeader>
                      <CardTitle className="text-sm flex items-center justify-between">
                        <span>{dataset.name}</span>
                        <Badge variant="outline">{dataset.attackMethod}</Badge>
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="grid grid-cols-3 gap-3">
                        <div className="p-3 bg-slate-900/50 rounded-lg">
                          <p className="text-xs text-slate-400 mb-1">AP50</p>
                          <p className="text-lg font-bold text-white">
                            {(dataset.ap50 * 100).toFixed(1)}%
                          </p>
                        </div>
                        <div className="p-3 bg-slate-900/50 rounded-lg">
                          <p className="text-xs text-slate-400 mb-1">F1-Score</p>
                          <p className="text-lg font-bold text-white">
                            {(dataset.f1Score * 100).toFixed(1)}%
                          </p>
                        </div>
                        <div className="p-3 bg-slate-900/50 rounded-lg">
                          <p className="text-xs text-slate-400 mb-1">성능 하락</p>
                          <p className="text-lg font-bold text-red-400">
                            -{dataset.performanceDrop.toFixed(1)}%
                          </p>
                        </div>
                      </div>
                      <div className="mt-3 p-2 bg-red-900/20 rounded-lg">
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-slate-400">공격 성공</span>
                          <span className="text-sm font-bold text-red-400">
                            {dataset.successfulAttacks.toLocaleString()} / {dataset.imageCount.toLocaleString()}
                            ({((dataset.successfulAttacks / dataset.imageCount) * 100).toFixed(1)}%)
                          </span>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </TabsContent>

              <TabsContent value="attacks" className="space-y-4">
                <Card className="bg-slate-800/50 border-white/10">
                  <CardHeader>
                    <CardTitle className="text-sm flex items-center gap-2">
                      <AlertTriangle className="w-4 h-4 text-red-400" />
                      공격 성공 이미지 목록
                    </CardTitle>
                    <CardDescription>클릭하여 원본과 공격 이미지를 비교하세요</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-3 gap-4">
                      {selectedAttackImages.map(image => (
                        <div
                          key={image.id}
                          className="cursor-pointer hover:opacity-80 transition-opacity"
                          onClick={() => handleViewImage(image)}
                        >
                          <div className="aspect-square bg-slate-900/50 rounded-lg relative overflow-hidden">
                            <div className="absolute inset-0 bg-gradient-to-br from-slate-700 to-slate-600" />
                            <ImageIcon className="absolute inset-0 m-auto w-12 h-12 text-slate-400" />
                            <div className="absolute top-2 right-2">
                              <Badge variant="destructive" className="text-xs">
                                {image.attackMethod}
                              </Badge>
                            </div>
                          </div>
                          <div className="mt-2 space-y-1">
                            <div className="flex items-center justify-between text-xs">
                              <span className="text-slate-400">원본:</span>
                              <span className="text-green-400">{image.originalPrediction}</span>
                            </div>
                            <div className="flex items-center justify-between text-xs">
                              <span className="text-slate-400">공격후:</span>
                              <span className="text-red-400">{image.attackPrediction}</span>
                            </div>
                            <div className="flex items-center justify-between text-xs">
                              <span className="text-slate-400">신뢰도 변화:</span>
                              <span className="text-white">
                                {(image.confidence.original * 100).toFixed(0)}% → {(image.confidence.attacked * 100).toFixed(0)}%
                              </span>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
          )}

          <div className="flex justify-end gap-2 mt-4">
            <Button variant="outline">
              <Download className="w-4 h-4 mr-2" />
              보고서 다운로드
            </Button>
            <Button variant="outline" onClick={() => setShowDetailModal(false)}>
              닫기
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      {/* 이미지 상세 보기 모달 */}
      <Dialog open={showImageModal} onOpenChange={setShowImageModal}>
        <DialogContent className="sm:max-w-[800px] bg-slate-900 border-slate-700">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 text-white">
              <Eye className="w-5 h-5" />
              공격 성공 이미지 상세
            </DialogTitle>
            <DialogDescription className="text-slate-400">
              원본 이미지와 적대적 공격이 적용된 이미지 비교
            </DialogDescription>
          </DialogHeader>

          {selectedImage && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <h3 className="text-sm font-medium text-white">원본 이미지</h3>
                  <div className="aspect-square bg-slate-900/50 rounded-lg relative overflow-hidden">
                    <div className="absolute inset-0 bg-gradient-to-br from-slate-700 to-slate-600" />
                    <ImageIcon className="absolute inset-0 m-auto w-20 h-20 text-slate-400" />
                  </div>
                  <div className="bg-slate-800/50 p-3 rounded-lg">
                    <div className="flex justify-between text-sm">
                      <span className="text-slate-400">예측:</span>
                      <span className="text-green-400 font-medium">
                        {selectedImage.originalPrediction}
                      </span>
                    </div>
                    <div className="flex justify-between text-sm mt-1">
                      <span className="text-slate-400">신뢰도:</span>
                      <span className="text-white">
                        {(selectedImage.confidence.original * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </div>

                <div className="space-y-2">
                  <h3 className="text-sm font-medium text-white flex items-center gap-2">
                    공격 적용 이미지
                    <Badge variant="destructive" className="text-xs">
                      {selectedImage.attackMethod}
                    </Badge>
                  </h3>
                  <div className="aspect-square bg-slate-900/50 rounded-lg relative overflow-hidden">
                    <div className="absolute inset-0 bg-gradient-to-br from-red-900/20 to-slate-600" />
                    <ImageIcon className="absolute inset-0 m-auto w-20 h-20 text-slate-400" />
                    {selectedImage.attackMethod === 'Adversarial Patch' && (
                      <div className="absolute top-4 right-4 w-12 h-12 bg-red-500/50 rounded" />
                    )}
                  </div>
                  <div className="bg-slate-800/50 p-3 rounded-lg">
                    <div className="flex justify-between text-sm">
                      <span className="text-slate-400">예측:</span>
                      <span className="text-red-400 font-medium">
                        {selectedImage.attackPrediction}
                      </span>
                    </div>
                    <div className="flex justify-between text-sm mt-1">
                      <span className="text-slate-400">신뢰도:</span>
                      <span className="text-white">
                        {(selectedImage.confidence.attacked * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              <Card className="bg-slate-800/50 border-white/10">
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm">공격 분석</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-3 gap-4 text-sm">
                    <div>
                      <p className="text-slate-400 mb-1">패치 생성 방법</p>
                      <p className="text-white font-medium">{selectedImage.attackMethod}</p>
                    </div>
                    <div>
                      <p className="text-slate-400 mb-1">예측 변화</p>
                      <p className="text-white font-medium">
                        {selectedImage.originalPrediction} → {selectedImage.attackPrediction}
                      </p>
                    </div>
                    <div>
                      <p className="text-slate-400 mb-1">신뢰도 감소</p>
                      <p className="text-red-400 font-medium">
                        -{((selectedImage.confidence.original - selectedImage.confidence.attacked) * 100).toFixed(1)}%
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          )}

          <div className="flex justify-end gap-2">
            <Button variant="outline">
              <ExternalLink className="w-4 h-4 mr-2" />
              전체 화면
            </Button>
            <Button variant="outline" onClick={() => setShowImageModal(false)}>
              닫기
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  )
}