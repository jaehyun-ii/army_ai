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
import { ScrollArea } from "@/components/ui/scroll-area"
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
  AlertCircle,
  CheckCircle2,
  Download,
  RefreshCw,
  Check,
  X,
  UserCog
} from "lucide-react"
import { useAuth } from '@/contexts/AuthContext'
import { validatePassword } from '@/lib/password-validator'
import { AdversarialToolLayout } from "@/components/layouts/adversarial-tool-layout"

interface User {
  id: string
  username: string
  email: string
  role: string
  isActive: boolean
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
    const [createAccountMessage, setCreateAccountMessage] = useState<{ type: 'success' | 'error' | null, message: string | null }>({ type: null, message: null })
    const [showDeleteDialog, setShowDeleteDialog] = useState(false)
  const [showEditDialog, setShowEditDialog] = useState(false)
  const [selectedUserForEdit, setSelectedUserForEdit] = useState<User | null>(null)
  const [editForm, setEditForm] = useState({
    password: '',
    confirmPassword: '',
    role: 'user',
    isActive: true
  })
  const [editPasswordErrors, setEditPasswordErrors] = useState<string[]>([])
  const [editMessage, setEditMessage] = useState<{ type: 'success' | 'error' | null, message: string | null }>({ type: null, message: null })
  
    // 새 운용자 생성 폼 상태
    const [newUser, setNewUser] = useState({
      username: '',
      email: '',
      password: '',
      confirmPassword: '',
      role: 'user'
    })
    const [passwordErrors, setPasswordErrors] = useState<string[]>([])
    const [passwordValidationStatus, setPasswordValidationStatus] = useState<{ type: 'success' | 'error' | null, message: string | null }>({ type: null, message: null })
    const [showPasswordHelp, setShowPasswordHelp] = useState(false)
  
    // 운용자 목록 가져오기
    const fetchUsers = useCallback(async () => {
      try {
        setLoading(true)
        const token = localStorage.getItem('token')
        const response = await await fetch('/api/users', {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        })
        if (response.ok) {
          const data = await response.json()
          // Transform backend response to match frontend interface (camelCase)
          const transformedUsers = data.map((user: any) => ({
            id: user.id,
            username: user.username,
            email: user.email || '',
            role: user.role,
            isActive: user.is_active,
            createdAt: user.created_at,
            updatedAt: user.updated_at
          }))
          setUsers(transformedUsers)
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
  
    // 운용자 필터링
    const filteredUsers = users.filter(user => {
      const matchesSearch =
        user.username.toLowerCase().includes(searchTerm.toLowerCase()) ||
        (user.email && user.email.toLowerCase().includes(searchTerm.toLowerCase()))
  
      const matchesRole = roleFilter === 'all' || user.role.toLowerCase() === roleFilter.toLowerCase()
      const matchesStatus = statusFilter === 'all' || (statusFilter === 'active' ? user.isActive : !user.isActive)
  
      return matchesSearch && matchesRole && matchesStatus
    })
  
    // 비밀번호 변경 시 유효성 검사
    const handlePasswordChange = (password: string) => {
      setNewUser({...newUser, password})
      const validation = validatePassword(password, newUser.username)
      setPasswordErrors(validation.errors)
  
      if (password.length > 0) { // Only show status if password is being entered
        if (validation.isValid) {
          setPasswordValidationStatus({ type: 'success', message: '비밀번호 정책을 충족합니다.' });
        } else {
          setPasswordValidationStatus({ type: 'error', message: `비밀번호 정책을 충족하지 않습니다.` });
        }
      } else {
        setPasswordValidationStatus({ type: null, message: null });
      }
    }
  
    // 운용자 생성
    const handleCreateUser = async () => {
      if (!newUser.username || !newUser.password) {
        setCreateAccountMessage({ type: 'error', message: '운용자 ID와 비밀번호는 필수 항목입니다.' })
        return
      }
  
      if (newUser.password !== newUser.confirmPassword) {
        setCreateAccountMessage({ type: 'error', message: '비밀번호가 일치하지 않습니다.' })
        return
      }
  
      // Client-side password validation
      const validation = validatePassword(newUser.password, newUser.username)
      if (!validation.isValid) {
        setCreateAccountMessage({ type: 'error', message: `비밀번호 정책 위반: ${validation.errors.join(', ')}` })
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
            email: newUser.email || undefined, // Email is optional
            password: newUser.password,
            role: newUser.role
          })
        })
  
        const result = await response.json()
  
        if (response.ok) {
          setNewUser({
            username: '',
            email: '',
            password: '',
            confirmPassword: '',
            role: 'user'
          })
          setPasswordErrors([])
          setPasswordValidationStatus({ type: null, message: null }) // Clear password validation status
          setShowCreateDialog(false)
          await fetchUsers()
          setCreateAccountMessage({ type: 'success', message: '운용자 계정이 성공적으로 생성되었습니다.' })
        } else {
          setCreateAccountMessage({ type: 'error', message: result.detail || '계정 생성에 실패했습니다.' })
        }
      } catch (error) {
        console.error('Error creating user:', error)
        setCreateAccountMessage({ type: 'error', message: '계정 생성 중 오류가 발생했습니다.' })
      }
    }
  
    // 운용자 삭제
    const handleDeleteUsers = async () => {    if (selectedUsers.length === 0) {
      alert('삭제할 운용자를 선택해주세요.')
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
      alert('운용자 삭제 중 오류가 발생했습니다.')
    }
  }

  // 운용자 수정 다이얼로그 열기
  const openEditDialog = (user: User) => {
    setSelectedUserForEdit(user)
    setEditForm({
      password: '',
      confirmPassword: '',
      role: user.role,
      isActive: user.isActive
    })
    setEditPasswordErrors([])
    setEditMessage({ type: null, message: null })
    setShowEditDialog(true)
  }

  // 운용자 수정 비밀번호 유효성 검사
  const handleEditPasswordChange = (password: string) => {
    setEditForm({...editForm, password})
    if (password.length > 0 && selectedUserForEdit) {
      const validation = validatePassword(password, selectedUserForEdit.username)
      setEditPasswordErrors(validation.errors)
    } else {
      setEditPasswordErrors([])
    }
  }

  // 운용자 수정
  const handleUpdateUser = async () => {
    if (!selectedUserForEdit) return

    // 비밀번호가 입력된 경우에만 검증
    if (editForm.password) {
      if (editForm.password !== editForm.confirmPassword) {
        setEditMessage({ type: 'error', message: '비밀번호가 일치하지 않습니다.' })
        return
      }

      const validation = validatePassword(editForm.password, selectedUserForEdit.username)
      if (!validation.isValid) {
        setEditMessage({ type: 'error', message: `비밀번호 정책 위반: ${validation.errors.join(', ')}` })
        return
      }
    }

    try {
      const token = localStorage.getItem('token')
      const updateData: any = {
        role: editForm.role,
        is_active: editForm.isActive
      }

      // 비밀번호가 입력된 경우에만 포함
      if (editForm.password) {
        updateData.password = editForm.password
      }

      const response = await fetch(`/api/users/${selectedUserForEdit.id}`, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify(updateData)
      })

      if (response.ok) {
        setShowEditDialog(false)
        await fetchUsers()
        setEditMessage({ type: 'success', message: '운용자 정보가 수정되었습니다.' })
        setTimeout(() => setEditMessage({ type: null, message: null }), 3000)
      } else {
        const result = await response.json()
        setEditMessage({ type: 'error', message: result.detail || '수정에 실패했습니다.' })
      }
    } catch (error) {
      console.error('Error updating user:', error)
      setEditMessage({ type: 'error', message: '운용자 수정 중 오류가 발생했습니다.' })
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
        <RefreshCw className="w-8 h-8 animate-spin text-foreground" />
      </div>
    )
  }

  // 권한 체크
  if (currentUser?.role !== 'admin') {
    return (
      <div className="flex items-center justify-center h-full">
        <Alert className="bg-error-container border-error max-w-md">
          <AlertCircle className="h-4 w-4 text-error" />
          <AlertDescription className="text-error">
            관리자 권한이 필요한 페이지입니다.
          </AlertDescription>
        </Alert>
      </div>
    )
  }

  return (
    <AdversarialToolLayout
      title="계정 관리"
      description="시스템 운용자 계정을 생성, 관리, 삭제할 수 있는 관리자 전용 기능입니다"
      icon={Users}
      headerStats={
        <Button
          onClick={fetchUsers}
          className="ds-btn-outline"
        >
          <RefreshCw className="w-4 h-4" />
          새로고침
        </Button>
      }
      rightPanel={{
        title: "계정 목록",
        icon: UserCog,
        description: "시스템 운용자 계정 통합 관리",
        children: (
          <div className="space-y-6">
            {/* 계정 통계 */}
            <Card className="bg-surface-container/50 border-border">
              <CardContent className="pt-6">
                <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
                  <div className="bg-primary-container rounded-xl p-4 border border-primary/20 shadow-sm hover:shadow-md transition-shadow">
                    <div className="flex items-center gap-2 mb-2">
                      <div className="p-2 bg-primary/10 rounded-lg">
                        <Users className="w-4 h-4 text-primary" />
                      </div>
                      <span className="text-on-primary-container/70 text-sm font-medium">총 계정 수</span>
                    </div>
                    <div className="text-3xl font-bold text-on-primary-container">{stats.totalUsers}</div>
                  </div>
                  <div className="bg-tertiary-container rounded-xl p-4 border border-tertiary/20 shadow-sm hover:shadow-md transition-shadow">
                    <div className="flex items-center gap-2 mb-2">
                      <div className="p-2 bg-tertiary/10 rounded-lg">
                        <UserCheck className="w-4 h-4 text-tertiary" />
                      </div>
                      <span className="text-on-tertiary-container/70 text-sm font-medium">활성 계정</span>
                    </div>
                    <div className="text-3xl font-bold text-on-tertiary-container">{stats.activeUsers}</div>
                  </div>
                  <div className="bg-error-container rounded-xl p-4 border border-error/20 shadow-sm hover:shadow-md transition-shadow">
                    <div className="flex items-center gap-2 mb-2">
                      <div className="p-2 bg-error/10 rounded-lg">
                        <UserX className="w-4 h-4 text-error" />
                      </div>
                      <span className="text-on-error-container/70 text-sm font-medium">비활성 계정</span>
                    </div>
                    <div className="text-3xl font-bold text-on-error-container">{stats.inactiveUsers}</div>
                  </div>
                  <div className="bg-secondary-container rounded-xl p-4 border border-secondary/20 shadow-sm hover:shadow-md transition-shadow">
                    <div className="flex items-center gap-2 mb-2">
                      <div className="p-2 bg-secondary/10 rounded-lg">
                        <Shield className="w-4 h-4 text-secondary" />
                      </div>
                      <span className="text-on-secondary-container/70 text-sm font-medium">관리자</span>
                    </div>
                    <div className="text-3xl font-bold text-on-secondary-container">{stats.adminUsers}</div>
                  </div>
                  <div className="bg-tertiary-container rounded-xl p-4 border border-tertiary/20 shadow-sm hover:shadow-md transition-shadow">
                    <div className="flex items-center gap-2 mb-2">
                      <div className="p-2 bg-tertiary/10 rounded-lg">
                        <Users className="w-4 h-4 text-tertiary" />
                      </div>
                      <span className="text-on-tertiary-container/70 text-sm font-medium">일반 운용자</span>
                    </div>
                    <div className="text-3xl font-bold text-on-tertiary-container">{stats.regularUsers}</div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* 검색 및 필터 */}
            <Card className="bg-surface-container/50 border-border">
              <CardContent className="pt-6">
            <div className="flex flex-col lg:flex-row gap-4">
              {/* 검색 */}
              <div className="flex-1">
                <div className="relative">
                  <Search className="absolute left-3 top-3 h-4 w-4 text-muted" />
                  <Input
                    placeholder="운용자 ID, 이메일로 검색..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="pl-10 bg-surface-container-high/50 border-border text-foreground"
                  />
                </div>
              </div>

              {/* 필터 */}
              <div className="flex gap-2">
                <Select value={roleFilter} onValueChange={setRoleFilter}>
                  <SelectTrigger className="w-32 bg-surface-container-high/50 border-border text-foreground">
                    <SelectValue placeholder="권한" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">전체 권한</SelectItem>
                    <SelectItem value="admin">관리자</SelectItem>
                    <SelectItem value="user">일반 운용자</SelectItem>
                  </SelectContent>
                </Select>

                <Select value={statusFilter} onValueChange={setStatusFilter}>
                  <SelectTrigger className="w-32 bg-surface-container-high/50 border-border text-foreground">
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
                <Dialog open={showCreateDialog} onOpenChange={(open) => {
                  setShowCreateDialog(open)
                  if (!open) { // Reset form and messages when dialog is closed
                    setNewUser({
                      username: '',
                      email: '',
                      password: '',
                      confirmPassword: '',
                      role: 'user'
                    })
                    setPasswordErrors([])
                    setPasswordValidationStatus({ type: null, message: null })
                    setCreateAccountMessage({ type: null, message: null })
                  }
                }}>
                  <DialogTrigger asChild>
                    <Button className="ds-btn-primary">
                      <UserPlus className="w-4 h-4 mr-2" />
                      계정 생성
                    </Button>
                  </DialogTrigger>
                  <DialogContent className="max-w-lg">
                    <DialogHeader>
                      <DialogTitle className="text-foreground">신규 운용자 생성</DialogTitle>
                      <DialogDescription className="text-muted">
                        새로운 운용자 계정을 생성합니다
                      </DialogDescription>
                    </DialogHeader>
                    {createAccountMessage.message && (
                      <Alert className={createAccountMessage.type === 'error' ? "bg-error-container border-error" : "bg-tertiary-container border-tertiary"}>
                        {createAccountMessage.type === 'error' ? <AlertCircle className="h-4 w-4 text-error" /> : <Check className="h-4 w-4 text-tertiary" />}
                        <AlertDescription className={createAccountMessage.type === 'error' ? "text-error" : "text-tertiary"}>
                          {createAccountMessage.message}
                        </AlertDescription>
                      </Alert>
                    )}
                    <div className="space-y-4">
                      <div>
                        <Label className="text-foreground">운용자 ID *</Label>
                        <Input
                          value={newUser.username}
                          onChange={(e) => setNewUser({...newUser, username: e.target.value})}
                          className="bg-surface-container-high/50 border-border text-foreground"
                          placeholder="영문, 숫자 조합"
                        />
                      </div>
                      <div>
                        <Label className="text-foreground">이메일 (선택)</Label>
                        <Input
                          type="email"
                          value={newUser.email}
                          onChange={(e) => setNewUser({...newUser, email: e.target.value})}
                          className="bg-surface-container-high/50 border-border text-foreground"
                          placeholder="example@army.mil.kr"
                        />
                      </div>
                      <div>
                        <Label className="text-foreground">비밀번호 *</Label>
                        <Input
                          type="password"
                          value={newUser.password}
                          onChange={(e) => handlePasswordChange(e.target.value)}
                          onFocus={() => setShowPasswordHelp(true)}
                          className="bg-surface-container-high/50 border-border text-foreground"
                          placeholder="9자 이상, 숫자+문자+특수문자"
                        />
                        {passwordValidationStatus.message && (
                          <Alert className={`mt-2 ${passwordValidationStatus.type === 'error' ? "bg-error-container border-error" : "bg-tertiary-container border-tertiary"}`}>
                            {passwordValidationStatus.type === 'error' ? <AlertCircle className="h-4 w-4 text-error" /> : <Check className="h-4 w-4 text-tertiary" />}
                            <AlertDescription className={passwordValidationStatus.type === 'error' ? "text-error" : "text-tertiary"}>
                              {passwordValidationStatus.message}
                            </AlertDescription>
                          </Alert>
                        )}
                        {showPasswordHelp && (
                          <div className="mt-2 p-3 bg-surface-container-high/30 rounded-md border border-border">
                            <p className="text-xs text-muted font-semibold mb-2">비밀번호 정책:</p>
                            <ul className="space-y-1 text-xs">
                              <li className={`flex items-center gap-2 ${!passwordErrors.some(e => e.includes('9자리')) && newUser.password.length > 0 ? 'text-tertiary' : 'text-muted'}`}>
                                {!passwordErrors.some(e => e.includes('9자리')) && newUser.password.length > 0 ? <Check className="w-3 h-3" /> : <X className="w-3 h-3" />}
                                숫자, 문자, 특수문자 포함 9자리 이상
                              </li>
                              <li className={`flex items-center gap-2 ${!passwordErrors.some(e => e.includes('ID')) && newUser.password.length > 0 ? 'text-tertiary' : 'text-muted'}`}>
                                {!passwordErrors.some(e => e.includes('ID')) && newUser.password.length > 0 ? <Check className="w-3 h-3" /> : <X className="w-3 h-3" />}
                                운용자 ID와 다르게 설정
                              </li>
                              <li className={`flex items-center gap-2 ${!passwordErrors.some(e => e.includes('연속')) && newUser.password.length > 0 ? 'text-tertiary' : 'text-muted'}`}>
                                {!passwordErrors.some(e => e.includes('연속')) && newUser.password.length > 0 ? <Check className="w-3 h-3" /> : <X className="w-3 h-3" />}
                                동일 문자 3회 이상 반복 금지
                              </li>
                              <li className={`flex items-center gap-2 ${!passwordErrors.some(e => e.includes('오름') || e.includes('내림')) && newUser.password.length > 0 ? 'text-tertiary' : 'text-muted'}`}>
                                {!passwordErrors.some(e => e.includes('오름') || e.includes('내림')) && newUser.password.length > 0 ? <Check className="w-3 h-3" /> : <X className="w-3 h-3" />}
                                연속 오름차순/내림차순 금지
                              </li>
                            </ul>
                          </div>
                        )}
                      </div>
                      <div>
                        <Label className="text-foreground">비밀번호 확인 *</Label>
                        <Input
                          type="password"
                          value={newUser.confirmPassword}
                          onChange={(e) => setNewUser({...newUser, confirmPassword: e.target.value})}
                          className="bg-surface-container-high/50 border-border text-foreground"
                          placeholder="비밀번호 재입력"
                        />
                        {newUser.password && newUser.confirmPassword && newUser.password !== newUser.confirmPassword && (
                          <p className="text-xs text-error mt-1 flex items-center gap-1">
                            <X className="w-3 h-3" />
                            비밀번호가 일치하지 않습니다
                          </p>
                        )}
                        {newUser.password && newUser.confirmPassword && newUser.password === newUser.confirmPassword && (
                          <p className="text-xs text-tertiary mt-1 flex items-center gap-1">
                            <Check className="w-3 h-3" />
                            비밀번호가 일치합니다
                          </p>
                        )}
                      </div>
                      <div>
                        <Label className="text-foreground">권한</Label>
                        <Select value={newUser.role} onValueChange={(value) => setNewUser({...newUser, role: value})}>
                          <SelectTrigger className="bg-surface-container-high/50 border-border text-foreground">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="user">일반 운용자</SelectItem>
                            <SelectItem value="admin">관리자</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                    </div>
                    <DialogFooter className="mt-6">
                      <Button
                        onClick={() => setShowCreateDialog(false)}
                        className="ds-btn-outline"
                      >
                        취소
                      </Button>
                      <Button
                        onClick={handleCreateUser}
                        className="ds-btn-primary"
                      >
                        <UserPlus className="w-4 h-4 mr-2" />
                        생성
                      </Button>
                    </DialogFooter>
                  </DialogContent>
                </Dialog>

                <Dialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
                  <DialogTrigger asChild>
                    <Button
                      disabled={selectedUsers.length === 0}
                      className="ds-btn-primary"
                    >
                      <Trash2 className="w-4 h-4 mr-2" />
                      선택 삭제 ({selectedUsers.length})
                    </Button>
                  </DialogTrigger>
                  <DialogContent className="max-w-lg">
                    <DialogHeader>
                      <DialogTitle className="text-foreground">계정 삭제 확인</DialogTitle>
                      <DialogDescription className="text-muted">
                        선택한 {selectedUsers.length}개의 계정을 삭제하시겠습니까?
                        이 작업은 되돌릴 수 없습니다.
                      </DialogDescription>
                    </DialogHeader>
                    <Alert className="bg-warning-container border-warning">
                      <AlertCircle className="h-4 w-4 text-tertiary" />
                      <AlertDescription className="text-tertiary">
                        삭제된 계정은 복구할 수 없습니다. 계속하시겠습니까?
                      </AlertDescription>
                    </Alert>
                    <DialogFooter className="mt-6">
                      <Button
                        onClick={() => setShowDeleteDialog(false)}
                        className="ds-btn-primary"
                      >
                        취소
                      </Button>
                      <Button
                        onClick={handleDeleteUsers}
                        className="ds-btn-primary"
                      >
                        <Trash2 className="w-4 h-4 mr-2" />
                        삭제
                      </Button>
                    </DialogFooter>
                  </DialogContent>
                </Dialog>

                {/* 수정 다이얼로그 */}
                <Dialog open={showEditDialog} onOpenChange={(open) => {
                  setShowEditDialog(open)
                  if (!open) {
                    setSelectedUserForEdit(null)
                    setEditForm({
                      password: '',
                      confirmPassword: '',
                      role: 'user',
                      isActive: true
                    })
                    setEditPasswordErrors([])
                    setEditMessage({ type: null, message: null })
                  }
                }}>
                  <DialogContent className="max-w-lg">
                    <DialogHeader>
                      <DialogTitle className="text-foreground">운용자 정보 수정</DialogTitle>
                      <DialogDescription className="text-muted">
                        {selectedUserForEdit?.username}의 정보를 수정합니다
                      </DialogDescription>
                    </DialogHeader>

                    <div className="space-y-4">
                      {/* Error/Success Messages */}
                      {editMessage.message && (
                        <Alert className={editMessage.type === 'error' ? "bg-error-container border-error" : "bg-tertiary-container border-tertiary"}>
                          {editMessage.type === 'error' ? <AlertCircle className="h-4 w-4 text-error" /> : <Check className="h-4 w-4 text-tertiary" />}
                          <AlertDescription className={editMessage.type === 'error' ? "text-error" : "text-tertiary"}>
                            {editMessage.message}
                          </AlertDescription>
                        </Alert>
                      )}

                      {/* 권한 및 상태 설정 */}
                      <div className="grid md:grid-cols-2 gap-4">
                        <div className="space-y-2">
                          <Label htmlFor="edit-role" className="text-foreground">
                            권한 <span className="text-error">*</span>
                          </Label>
                          <Select value={editForm.role} onValueChange={(value) => setEditForm({...editForm, role: value})}>
                            <SelectTrigger id="edit-role" className="bg-surface-container-high/50 border-border text-foreground">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="user">일반 운용자</SelectItem>
                              <SelectItem value="admin">관리자</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>

                        <div className="space-y-2">
                          <Label htmlFor="edit-status" className="text-foreground">
                            상태 <span className="text-error">*</span>
                          </Label>
                          <Select value={editForm.isActive ? 'active' : 'inactive'} onValueChange={(value) => setEditForm({...editForm, isActive: value === 'active'})}>
                            <SelectTrigger id="edit-status" className="bg-surface-container-high/50 border-border text-foreground">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="active">활성</SelectItem>
                              <SelectItem value="inactive">비활성</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>
                      </div>

                      {/* 비밀번호 변경 섹션 */}
                      <div className="space-y-2 border-t border-border pt-4">
                        <Label htmlFor="edit-password" className="text-foreground">
                          새 비밀번호
                        </Label>
                        <p className="text-xs text-muted">비밀번호를 변경하려면 입력하세요 (선택사항)</p>
                        <Input
                          id="edit-password"
                          type="password"
                          value={editForm.password}
                          onChange={(e) => handleEditPasswordChange(e.target.value)}
                          className="bg-surface-container-high/50 border-border text-foreground"
                          placeholder="새 비밀번호 (선택사항)"
                        />
                        {editPasswordErrors.length > 0 && editForm.password && (
                          <Alert variant="destructive" className="bg-error-container border-error">
                            <AlertCircle className="h-4 w-4 text-error" />
                            <AlertDescription className="text-error">
                              <div className="font-semibold mb-1">비밀번호 정책 위반</div>
                              <ul className="space-y-1 text-xs">
                                {editPasswordErrors.map((error, idx) => (
                                  <li key={idx} className="flex items-center gap-1">
                                    <X className="w-3 h-3" />
                                    {error}
                                  </li>
                                ))}
                              </ul>
                            </AlertDescription>
                          </Alert>
                        )}
                      </div>

                      {/* 비밀번호 확인 */}
                      {editForm.password && (
                        <div className="space-y-2">
                          <Label htmlFor="edit-confirm-password" className="text-foreground">
                            새 비밀번호 확인
                          </Label>
                          <Input
                            id="edit-confirm-password"
                            type="password"
                            value={editForm.confirmPassword}
                            onChange={(e) => setEditForm({...editForm, confirmPassword: e.target.value})}
                            className="bg-surface-container-high/50 border-border text-foreground"
                            placeholder="새 비밀번호 확인"
                          />
                          {editForm.password && editForm.confirmPassword && editForm.password !== editForm.confirmPassword && (
                            <p className="text-xs text-error mt-1 flex items-center gap-1">
                              <X className="w-3 h-3" />
                              비밀번호가 일치하지 않습니다
                            </p>
                          )}
                          {editForm.password && editForm.confirmPassword && editForm.password === editForm.confirmPassword && (
                            <p className="text-xs text-tertiary mt-1 flex items-center gap-1">
                              <Check className="w-3 h-3" />
                              비밀번호가 일치합니다
                            </p>
                          )}
                        </div>
                      )}
                    </div>

                    <DialogFooter className="mt-6">
                      <Button
                        onClick={() => setShowEditDialog(false)}
                        className="ds-btn-outline"
                      >
                        취소
                      </Button>
                      <Button
                        onClick={handleUpdateUser}
                        className="ds-btn-primary"
                      >
                        <Check className="w-4 h-4 mr-2" />
                        수정
                      </Button>
                    </DialogFooter>
                  </DialogContent>
                </Dialog>
                </div>
              </div>
              </CardContent>
            </Card>

            {/* 계정 목록 */}
            <Card className="bg-surface-container/50 border-border">
              <CardHeader className="pb-3">
                <CardTitle className="text-primary">계정 목록</CardTitle>
                <CardDescription className="text-muted">
                  총 {filteredUsers.length}개의 계정이 검색되었습니다
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ScrollArea className="h-[600px]">
                  <Table>
                <TableHeader>
                  <TableRow className="border-border">
                    <TableHead className="w-12 text-muted">
                      <Checkbox
                        checked={selectedUsers.length === filteredUsers.filter(u => u.id !== currentUser?.id).length && filteredUsers.length > 0}
                        onCheckedChange={handleSelectAll}
                      />
                    </TableHead>
                    <TableHead className="text-muted">운용자 ID</TableHead>
                    <TableHead className="text-muted">이메일</TableHead>
                    <TableHead className="text-muted">권한</TableHead>
                    <TableHead className="text-muted">상태</TableHead>
                    <TableHead className="text-muted">생성일</TableHead>
                    <TableHead className="text-muted">수정일</TableHead>
                    <TableHead className="text-muted">작업</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filteredUsers.map((user) => (
                    <TableRow key={user.id} className="border-border">
                      <TableCell>
                        <Checkbox
                          checked={selectedUsers.includes(user.id)}
                          onCheckedChange={(checked) => handleSelectUser(user.id, checked as boolean)}
                          disabled={user.id === currentUser?.id}
                        />
                      </TableCell>
                      <TableCell className="font-medium text-foreground">{user.username}</TableCell>
                      <TableCell className="text-muted">{user.email || '-'}</TableCell>
                      <TableCell>
                        <Badge className={user.role === 'admin'
                          ? 'bg-secondary-container text-on-secondary-container border-secondary/30'
                          : 'bg-surface-container-high text-on-surface-variant border-outline-variant'}>
                          {user.role === 'admin' ? (
                            <>
                              <Shield className="w-3 h-3 mr-1" />
                              관리자
                            </>
                          ) : (
                            <>
                              <Users className="w-3 h-3 mr-1" />
                              일반 운용자
                            </>
                          )}
                        </Badge>
                      </TableCell>
                      <TableCell>
                        <Badge className={user.isActive
                          ? 'bg-tertiary-container text-on-tertiary-container border-tertiary/30'
                          : 'bg-error-container text-on-error-container border-error/30'}>
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
                      <TableCell className="text-muted text-sm">{formatDate(user.createdAt)}</TableCell>
                      <TableCell className="text-muted text-sm">{formatDate(user.updatedAt)}</TableCell>
                      <TableCell>
                        <div className="flex items-center gap-2">
                          <Button
                            onClick={() => openEditDialog(user)}
                            disabled={user.id === currentUser?.id}
                            className="ds-btn-icon"
                            title="수정"
                          >
                            <Edit className="w-4 h-4" />
                          </Button>
                        </div>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>

                {filteredUsers.length === 0 && (
                  <div className="text-center py-8">
                    <Users className="w-12 h-12 mx-auto mb-4 text-muted" />
                    <p className="text-muted">검색 결과가 없습니다</p>
                    <p className="text-muted text-sm">검색 조건을 변경해보세요</p>
                  </div>
                )}
                </ScrollArea>
              </CardContent>
            </Card>
          </div>
        )
      }}
    />
  )
}