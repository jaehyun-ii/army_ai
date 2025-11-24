'use client'

import React, { createContext, useContext, useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'

interface User {
  id: string
  username: string
  email: string
  name: string
  rank?: string
  unit?: string
  role: string
  lastLoginAt?: string | null
}

interface AuthContextType {
  user: User | null
  isAuthenticated: boolean
  login: (username: string, password: string) => Promise<void>
  logout: () => Promise<void>
  loading: boolean
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [loading, setLoading] = useState(true)
  const router = useRouter()

  useEffect(() => {
    checkAuth()
  }, [])

  const checkAuth = async () => {
    try {
      const token = localStorage.getItem('token')
      if (token) {
        const response = await fetch('/api/auth/me', {
          headers: {
            'Authorization': `Bearer ${token}`
          },
          credentials: 'include'
        })
        if (response.ok) {
          const userData = await response.json()
          setUser(userData)
        } else {
          localStorage.removeItem('token')
        }
      }
    } catch (error) {
      console.error('Auth check failed:', error)
    } finally {
      setLoading(false)
    }
  }

  const login = async (username: string, password: string) => {
    console.log('🔐 Login attempt started for:', username)
    try {
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({ username, password }),
      })

      console.log('📡 API response status:', response.status)
      const data = await response.json()
      console.log('📦 API response data:', data)

      if (!response.ok || !data.success) {
        console.error('❌ Login failed:', data.error)
        throw new Error(data.error || 'Login failed')
      }

      console.log('✅ Login successful, saving token and user data')
      localStorage.setItem('token', data.token)
      setUser(data.user)

      console.log('🚀 Redirecting to dashboard...')
      // 즉시 리다이렉션 (setTimeout 제거)
      router.push('/dashboard')
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Login failed'
      console.error('❌ Login error:', error)
      throw new Error(message)
    }
  }

  const logout = async () => {
    const token = localStorage.getItem('token')

    try {
      await fetch('/api/auth/logout', {
        method: 'POST',
        headers: token
          ? {
              'Authorization': `Bearer ${token}`
            }
          : undefined,
        credentials: 'include'
      })
    } catch (error) {
      console.error('Logout request failed:', error)
    }

    localStorage.removeItem('token')
    setUser(null)
    window.location.href = '/login'
  }

  return (
    <AuthContext.Provider value={{
      user,
      isAuthenticated: !!user,
      login,
      logout,
      loading
    }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}
