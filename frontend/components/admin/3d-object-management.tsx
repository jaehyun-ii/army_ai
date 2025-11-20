"use client"

import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Badge } from '@/components/ui/badge'
import { Box, Upload, Download, Trash2, Edit2, Eye, Layers } from 'lucide-react'
import { Progress } from '@/components/ui/progress'

interface Object3D {
  id: string
  name: string
  category: string
  format: string
  size: string
  polygons: number
  status: 'ready' | 'processing' | 'error'
  lastModified: string
}

export function Object3DManagement() {
  const [objects, setObjects] = useState<Object3D[]>([
    {
      id: '1',
      name: 'K2 전차',
      category: '군용차량',
      format: 'FBX',
      size: '45.2 MB',
      polygons: 125000,
      status: 'ready',
      lastModified: '2024-01-18'
    },
    {
      id: '2',
      name: 'AH-64 아파치',
      category: '항공기',
      format: 'OBJ',
      size: '32.1 MB',
      polygons: 98000,
      status: 'ready',
      lastModified: '2024-01-15'
    },
  ])

  const getStatusBadge = (status: string) => {
    const variants: Record<string, 'default' | 'secondary' | 'destructive'> = {
      ready: 'default',
      processing: 'secondary',
      error: 'destructive'
    }
    const labels: Record<string, string> = {
      ready: '준비됨',
      processing: '처리중',
      error: '오류'
    }
    return <Badge variant={variants[status] || 'default'}>{labels[status] || status}</Badge>
  }

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="bg-gradient-to-br from-purple-900/20 to-purple-800/20 border-purple-700/30">
          <CardHeader className="pb-3">
            <CardTitle className="text-lg flex items-center gap-2">
              <Box className="w-5 h-5" />
              전체 모델
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{objects.length}</div>
            <p className="text-sm text-muted-foreground">3D 객체</p>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-cyan-900/20 to-cyan-800/20 border-cyan-700/30">
          <CardHeader className="pb-3">
            <CardTitle className="text-lg flex items-center gap-2">
              <Layers className="w-5 h-5" />
              총 폴리곤
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">
              {(objects.reduce((acc, obj) => acc + obj.polygons, 0) / 1000).toFixed(0)}K
            </div>
            <p className="text-sm text-muted-foreground">전체 폴리곤 수</p>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-green-900/20 to-green-800/20 border-green-700/30">
          <CardHeader className="pb-3">
            <CardTitle className="text-lg">준비됨</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">
              {objects.filter(o => o.status === 'ready').length}
            </div>
            <p className="text-sm text-muted-foreground">사용 가능</p>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <div className="flex justify-between items-center">
            <div>
              <CardTitle>3D 객체 관리</CardTitle>
              <CardDescription>Carla 시뮬레이션용 3D 모델을 관리합니다</CardDescription>
            </div>
            <div className="flex gap-2">
              <Button size="sm">
                <Box className="w-4 h-4 mr-2" />
                모델 업로드
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="mb-4 flex gap-4">
            <Input placeholder="모델명 검색..." className="max-w-sm" />
            <Select defaultValue="all">
              <SelectTrigger className="w-[180px]">
                <SelectValue placeholder="카테고리" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">전체</SelectItem>
                <SelectItem value="vehicle">군용차량</SelectItem>
                <SelectItem value="aircraft">항공기</SelectItem>
                <SelectItem value="equipment">개인장비</SelectItem>
                <SelectItem value="building">건축물</SelectItem>
              </SelectContent>
            </Select>
            <Select defaultValue="all">
              <SelectTrigger className="w-[180px]">
                <SelectValue placeholder="포맷" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">전체</SelectItem>
                <SelectItem value="fbx">FBX</SelectItem>
                <SelectItem value="obj">OBJ</SelectItem>
                <SelectItem value="gltf">GLTF</SelectItem>
                <SelectItem value="dae">DAE</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>모델명</TableHead>
                <TableHead>카테고리</TableHead>
                <TableHead>포맷</TableHead>
                <TableHead>크기</TableHead>
                <TableHead>폴리곤</TableHead>
                <TableHead>상태</TableHead>
                <TableHead>수정일</TableHead>
                <TableHead className="text-right">작업</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {objects.map((object) => (
                <TableRow key={object.id}>
                  <TableCell className="font-medium">{object.name}</TableCell>
                  <TableCell>{object.category}</TableCell>
                  <TableCell>
                    <Badge variant="outline">{object.format}</Badge>
                  </TableCell>
                  <TableCell>{object.size}</TableCell>
                  <TableCell>{object.polygons.toLocaleString()}</TableCell>
                  <TableCell>{getStatusBadge(object.status)}</TableCell>
                  <TableCell>{object.lastModified}</TableCell>
                  <TableCell className="text-right">
                    <div className="flex justify-end gap-1">
                      <Button variant="ghost" size="sm">
                        <Eye className="w-4 h-4" />
                      </Button>
                      <Button variant="ghost" size="sm">
                        <Edit2 className="w-4 h-4" />
                      </Button>
                      <Button variant="ghost" size="sm">
                        <Download className="w-4 h-4" />
                      </Button>
                      <Button variant="ghost" size="sm" className="text-red-500">
                        <Trash2 className="w-4 h-4" />
                      </Button>
                    </div>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  )
}