"use client"

import React, { useState, useEffect, useCallback } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Checkbox } from "@/components/ui/checkbox"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import {
  UserPlus,
  Search,
  Trash2,
  Shield,
  Users,
  UserCheck,
  UserX,
  Edit,
  Eye,
  EyeOff,
  AlertCircle,
  CheckCircle2,
  Download,
  RefreshCw
} from "lucide-react"
import { useAuth } from '@/contexts/AuthContext'

interface User {
  id: string
  username: string
  name: string
  email: string
  rank?: string
  unit?: string
  role: string
  isActive: boolean
  lastLoginAt?: string
  createdAt: string
  updatedAt: string
}

interface AccountStats {
  totalUsers: number
  activeUsers: number
  inactiveUsers: number
  adminUsers: number
  regularUsers: number
}

export function AccountManagementDB() {
  const { user: currentUser } = useAuth()
  const [users, setUsers] = useState<User[]>([])
  const [loading, setLoading] = useState(true)
  const [searchTerm, setSearchTerm] = useState('')
  const [roleFilter, setRoleFilter] = useState<string>('all')
  const [statusFilter, setStatusFilter] = useState<string>('all')
  const [selectedUsers, setSelectedUsers] = useState<string[]>([])
  const [showCreateDialog, setShowCreateDialog] = useState(false)
  const [showDeleteDialog, setShowDeleteDialog] = useState(false)

  // 새 사용자 생성 폼 상태
  const [newUser, setNewUser] = useState({
    username: '',
    name: '',
    email: '',
    password: '',
    confirmPassword: '',
    rank: '',
    unit: '',
    role: 'USER'
  })

  // 사용자 목록 가져오기
  const fetchUsers = useCallback(async () => {
    try {
      setLoading(true)
      const token = localStorage.getItem('token')
      const response = await fetch('/api/users', {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })
      if (response.ok) {
        const data = await response.json()
        setUsers(data)
      }
    } catch (error) {
      console.error('Error fetching users:', error)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchUsers()
  }, [fetchUsers])

  // 계정 통계 계산
  const getAccountStats = (): AccountStats => {
    return {
      totalUsers: users.length,
      activeUsers: users.filter(u => u.isActive).length,
      inactiveUsers: users.filter(u => !u.isActive).length,
      adminUsers: users.filter(u => u.role === 'admin').length,
      regularUsers: users.filter(u => u.role === 'user').length
    }
  }

  // 사용자 필터링
  const filteredUsers = users.filter(user => {
    const matchesSearch =
      user.username.toLowerCase().includes(searchTerm.toLowerCase()) ||
      user.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      user.email.toLowerCase().includes(searchTerm.toLowerCase())

    const matchesRole = roleFilter === 'all' || user.role === roleFilter
    const matchesStatus = statusFilter === 'all' || (statusFilter === 'active' ? user.isActive : !user.isActive)

    return matchesSearch && matchesRole && matchesStatus
  })

  // 사용자 생성
  const handleCreateUser = async () => {
    if (!newUser.username || !newUser.name || !newUser.email || !newUser.password) {
      alert('모든 필수 필드를 입력해주세요.')
      return
    }

    if (newUser.password !== newUser.confirmPassword) {
      alert('비밀번호가 일치하지 않습니다.')
      return
    }

    try {
      const response = await fetch('/api/auth/register', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          username: newUser.username,
          name: newUser.name,
          email: newUser.email,
          password: newUser.password,
          rank: newUser.rank,
          unit: newUser.unit,
          role: newUser.role
        })
      })

      const result = await response.json()

      if (response.ok) {
        setNewUser({
          username: '',
          name: '',
          email: '',
          password: '',
          confirmPassword: '',
          rank: '',
          unit: '',
          role: 'USER'
        })
        setShowCreateDialog(false)
        await fetchUsers()
      } else {
        alert(result.detail || '계정 생성에 실패했습니다.')
      }
    } catch (error) {
      console.error('Error creating user:', error)
      alert('계정 생성 중 오류가 발생했습니다.')
    }
  }

  // 사용자 삭제
  const handleDeleteUsers = async () => {
    if (selectedUsers.length === 0) {
      alert('삭제할 사용자를 선택해주세요.')
      return
    }

    try {
      const token = localStorage.getItem('token')
      for (const userId of selectedUsers) {
        await fetch(`/api/users/${userId}`, {
          method: 'DELETE',
          headers: {
            'Authorization': `Bearer ${token}`
          }
        })
      }
      setSelectedUsers([])
      setShowDeleteDialog(false)
      await fetchUsers()
    } catch (error) {
      console.error('Error deleting users:', error)
      alert('사용자 삭제 중 오류가 발생했습니다.')
    }
  }

  // 사용자 상태 토글
  const toggleUserStatus = async (userId: string) => {
    const user = users.find(u => u.id === userId)
    if (!user) return

    try {
      const token = localStorage.getItem('token')
      const response = await fetch(`/api/users/${userId}`, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          is_active: !user.isActive
        })
      })

      if (response.ok) {
        await fetchUsers()
      }
    } catch (error) {
      console.error('Error updating user status:', error)
    }
  }

  // 전체 선택/해제
  const handleSelectAll = (checked: boolean) => {
    if (checked) {
      setSelectedUsers(filteredUsers.filter(u => u.id !== currentUser?.id).map(u => u.id))
    } else {
      setSelectedUsers([])
    }
  }

  // 개별 선택
  const handleSelectUser = (userId: string, checked: boolean) => {
    if (checked) {
      setSelectedUsers([...selectedUsers, userId])
    } else {
      setSelectedUsers(selectedUsers.filter(id => id !== userId))
    }
  }

  const stats = getAccountStats()

  // 날짜 포맷
  const formatDate = (dateString?: string) => {
    if (!dateString) return '-'
    return new Date(dateString).toLocaleString('ko-KR')
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <RefreshCw className="w-8 h-8 animate-spin text-white" />
      </div>
    )
  }

  // 권한 체크
  if (currentUser?.role !== 'admin') {
    return (
      <div className="flex items-center justify-center h-full">
        <Alert className="bg-red-900/20 border-red-500/30 max-w-md">
          <AlertCircle className="h-4 w-4 text-red-400" />
          <AlertDescription className="text-red-300">
            관리자 권한이 필요한 페이지입니다.
          </AlertDescription>
        </Alert>
      </div>
    )
  }

  return (
    <div className="h-full flex flex-col gap-2">
      {/* Header */}
      <div className="bg-gradient-to-r from-slate-800/80 to-slate-900/80 rounded-xl p-3 border border-white/10 shadow-xl flex-shrink-0">
        <div className="flex-shrink-0">
          <h1 className="text-lg lg:text-xl font-bold text-white flex items-center gap-2">
            <Users className="w-6 h-6 text-purple-400" />
            계정 관리 (관리자 전용)
          </h1>
          <p className="text-xs text-slate-400">시스템 사용자 계정을 생성, 관리, 삭제할 수 있는 관리자 전용 기능입니다</p>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 space-y-6 overflow-auto">
        {/* 계정 통계 */}
        <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
          <Card className="bg-slate-800/50 border-white/10">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-slate-400">총 계정 수</p>
                  <p className="text-2xl font-bold text-white">{stats.totalUsers}</p>
                </div>
                <Users className="w-8 h-8 text-blue-400" />
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-white/10">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-slate-400">활성 계정</p>
                  <p className="text-2xl font-bold text-green-400">{stats.activeUsers}</p>
                </div>
                <UserCheck className="w-8 h-8 text-green-400" />
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-white/10">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-slate-400">비활성 계정</p>
                  <p className="text-2xl font-bold text-red-400">{stats.inactiveUsers}</p>
                </div>
                <UserX className="w-8 h-8 text-red-400" />
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-white/10">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-slate-400">관리자</p>
                  <p className="text-2xl font-bold text-orange-400">{stats.adminUsers}</p>
                </div>
                <Shield className="w-8 h-8 text-orange-400" />
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-white/10">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-slate-400">일반 사용자</p>
                  <p className="text-2xl font-bold text-purple-400">{stats.regularUsers}</p>
                </div>
                <Users className="w-8 h-8 text-purple-400" />
              </div>
            </CardContent>
          </Card>
        </div>

        {/* 검색 및 필터 */}
        <Card className="bg-slate-800/50 border-white/10">
          <CardContent className="p-4">
            <div className="flex flex-col lg:flex-row gap-4">
              {/* 검색 */}
              <div className="flex-1">
                <div className="relative">
                  <Search className="absolute left-3 top-3 h-4 w-4 text-slate-400" />
                  <Input
                    placeholder="사용자 ID, 이름, 이메일로 검색..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="pl-10 bg-slate-700/50 border-white/10 text-white"
                  />
                </div>
              </div>

              {/* 필터 */}
              <div className="flex gap-2">
                <Select value={roleFilter} onValueChange={setRoleFilter}>
                  <SelectTrigger className="w-32 bg-slate-700/50 border-white/10 text-white">
                    <SelectValue placeholder="권한" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">전체 권한</SelectItem>
                    <SelectItem value="admin">관리자</SelectItem>
                    <SelectItem value="user">일반 사용자</SelectItem>
                  </SelectContent>
                </Select>

                <Select value={statusFilter} onValueChange={setStatusFilter}>
                  <SelectTrigger className="w-32 bg-slate-700/50 border-white/10 text-white">
                    <SelectValue placeholder="상태" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">전체 상태</SelectItem>
                    <SelectItem value="active">활성</SelectItem>
                    <SelectItem value="inactive">비활성</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* 액션 버튼 */}
              <div className="flex gap-2">
                <Dialog open={showCreateDialog} onOpenChange={setShowCreateDialog}>
                  <DialogTrigger asChild>
                    <Button className="bg-green-600 hover:bg-green-700">
                      <UserPlus className="w-4 h-4 mr-2" />
                      계정 생성
                    </Button>
                  </DialogTrigger>
                  <DialogContent className="bg-slate-800 border-white/10">
                    <DialogHeader>
                      <DialogTitle className="text-white">신규 사용자 생성</DialogTitle>
                      <DialogDescription className="text-slate-400">
                        새로운 사용자 계정을 생성합니다
                      </DialogDescription>
                    </DialogHeader>
                    <div className="space-y-4">
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <Label className="text-white">사용자 ID *</Label>
                          <Input
                            value={newUser.username}
                            onChange={(e) => setNewUser({...newUser, username: e.target.value})}
                            className="bg-slate-700/50 border-white/10 text-white"
                            placeholder="영문, 숫자 조합"
                          />
                        </div>
                        <div>
                          <Label className="text-white">이름 *</Label>
                          <Input
                            value={newUser.name}
                            onChange={(e) => setNewUser({...newUser, name: e.target.value})}
                            className="bg-slate-700/50 border-white/10 text-white"
                            placeholder="실명 입력"
                          />
                        </div>
                      </div>
                      <div>
                        <Label className="text-white">이메일 *</Label>
                        <Input
                          type="email"
                          value={newUser.email}
                          onChange={(e) => setNewUser({...newUser, email: e.target.value})}
                          className="bg-slate-700/50 border-white/10 text-white"
                          placeholder="example@army.mil.kr"
                        />
                      </div>
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <Label className="text-white">비밀번호 *</Label>
                          <Input
                            type="password"
                            value={newUser.password}
                            onChange={(e) => setNewUser({...newUser, password: e.target.value})}
                            className="bg-slate-700/50 border-white/10 text-white"
                          />
                        </div>
                        <div>
                          <Label className="text-white">비밀번호 확인 *</Label>
                          <Input
                            type="password"
                            value={newUser.confirmPassword}
                            onChange={(e) => setNewUser({...newUser, confirmPassword: e.target.value})}
                            className="bg-slate-700/50 border-white/10 text-white"
                          />
                        </div>
                      </div>
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <Label className="text-white">계급</Label>
                          <Input
                            value={newUser.rank}
                            onChange={(e) => setNewUser({...newUser, rank: e.target.value})}
                            className="bg-slate-700/50 border-white/10 text-white"
                            placeholder="대위, 중위 등"
                          />
                        </div>
                        <div>
                          <Label className="text-white">소속 부대</Label>
                          <Input
                            value={newUser.unit}
                            onChange={(e) => setNewUser({...newUser, unit: e.target.value})}
                            className="bg-slate-700/50 border-white/10 text-white"
                            placeholder="소속 부대명"
                          />
                        </div>
                      </div>
                      <div>
                        <Label className="text-white">권한</Label>
                        <Select value={newUser.role} onValueChange={(value) => setNewUser({...newUser, role: value})}>
                          <SelectTrigger className="bg-slate-700/50 border-white/10 text-white">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="user">일반 사용자</SelectItem>
                            <SelectItem value="admin">관리자</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                    </div>
                    <DialogFooter>
                      <Button variant="outline" onClick={() => setShowCreateDialog(false)}>
                        취소
                      </Button>
                      <Button onClick={handleCreateUser} className="bg-green-600 hover:bg-green-700">
                        생성
                      </Button>
                    </DialogFooter>
                  </DialogContent>
                </Dialog>

                <Dialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
                  <DialogTrigger asChild>
                    <Button
                      className="bg-red-600 hover:bg-red-700 text-white border-0"
                      disabled={selectedUsers.length === 0}
                    >
                      <Trash2 className="w-4 h-4 mr-2" />
                      선택 삭제 ({selectedUsers.length})
                    </Button>
                  </DialogTrigger>
                  <DialogContent className="bg-slate-800 border-white/10">
                    <DialogHeader>
                      <DialogTitle className="text-white">계정 삭제 확인</DialogTitle>
                      <DialogDescription className="text-slate-400">
                        선택한 {selectedUsers.length}개의 계정을 삭제하시겠습니까?
                        이 작업은 되돌릴 수 없습니다.
                      </DialogDescription>
                    </DialogHeader>
                    <Alert className="bg-yellow-900/20 border-yellow-500/30">
                      <AlertCircle className="h-4 w-4 text-yellow-400" />
                      <AlertDescription className="text-yellow-300">
                        삭제된 계정은 복구할 수 없습니다. 계속하시겠습니까?
                      </AlertDescription>
                    </Alert>
                    <DialogFooter>
                      <Button variant="outline" onClick={() => setShowDeleteDialog(false)}>
                        취소
                      </Button>
                      <Button               
                        className="bg-red-600 hover:bg-red-700 text-white border-0"
                        onClick={handleDeleteUsers}>
                        삭제
                      </Button>
                    </DialogFooter>
                  </DialogContent>
                </Dialog>

                <Button variant="outline" className="border-white/10 text-white" onClick={fetchUsers}>
                  <RefreshCw className="w-4 h-4 mr-2" />
                  새로고침
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* 계정 목록 */}
        <Card className="bg-slate-800/50 border-white/10">
          <CardHeader>
            <CardTitle className="text-white">계정 목록</CardTitle>
            <CardDescription className="text-slate-400">
              총 {filteredUsers.length}개의 계정이 검색되었습니다
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="rounded-md border border-white/10">
              <Table>
                <TableHeader>
                  <TableRow className="border-white/10">
                    <TableHead className="w-12 text-slate-300">
                      <Checkbox
                        checked={selectedUsers.length === filteredUsers.filter(u => u.id !== currentUser?.id).length && filteredUsers.length > 0}
                        onCheckedChange={handleSelectAll}
                      />
                    </TableHead>
                    <TableHead className="text-slate-300">사용자 ID</TableHead>
                    <TableHead className="text-slate-300">이름</TableHead>
                    <TableHead className="text-slate-300">이메일</TableHead>
                    <TableHead className="text-slate-300">계급</TableHead>
                    <TableHead className="text-slate-300">소속</TableHead>
                    <TableHead className="text-slate-300">권한</TableHead>
                    <TableHead className="text-slate-300">상태</TableHead>
                    <TableHead className="text-slate-300">최근 로그인</TableHead>
                    <TableHead className="text-slate-300">생성일</TableHead>
                    <TableHead className="text-slate-300">작업</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filteredUsers.map((user) => (
                    <TableRow key={user.id} className="border-white/10">
                      <TableCell>
                        <Checkbox
                          checked={selectedUsers.includes(user.id)}
                          onCheckedChange={(checked) => handleSelectUser(user.id, checked as boolean)}
                          disabled={user.id === currentUser?.id}
                        />
                      </TableCell>
                      <TableCell className="font-medium text-white">{user.username}</TableCell>
                      <TableCell className="text-slate-300">{user.name}</TableCell>
                      <TableCell className="text-slate-300">{user.email}</TableCell>
                      <TableCell className="text-slate-300">{user.rank || '-'}</TableCell>
                      <TableCell className="text-slate-300">{user.unit || '-'}</TableCell>
                      <TableCell>
                        <Badge variant={user.role === 'admin' ? 'default' : 'secondary'}>
                          {user.role === 'admin' ? (
                            <>
                              <Shield className="w-3 h-3 mr-1" />
                              관리자
                            </>
                          ) : (
                            '일반 사용자'
                          )}
                        </Badge>
                      </TableCell>
                      <TableCell>
                        <Badge variant={user.isActive ? 'default' : 'destructive'}>
                          {user.isActive ? (
                            <>
                              <CheckCircle2 className="w-3 h-3 mr-1" />
                              활성
                            </>
                          ) : (
                            <>
                              <UserX className="w-3 h-3 mr-1" />
                              비활성
                            </>
                          )}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-slate-300">{formatDate(user.lastLoginAt)}</TableCell>
                      <TableCell className="text-slate-300">{formatDate(user.createdAt)}</TableCell>
                      <TableCell>
                        <div className="flex items-center gap-2">
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => toggleUserStatus(user.id)}
                            disabled={user.id === currentUser?.id}
                            className="border-white/10 text-white"
                          >
                            {user.isActive ? (
                              <EyeOff className="w-3 h-3" />
                            ) : (
                              <Eye className="w-3 h-3" />
                            )}
                          </Button>
                        </div>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>

            {filteredUsers.length === 0 && (
              <div className="text-center py-8">
                <Users className="w-12 h-12 mx-auto mb-4 text-slate-500" />
                <p className="text-slate-400">검색 결과가 없습니다</p>
                <p className="text-slate-500 text-sm">검색 조건을 변경해보세요</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}