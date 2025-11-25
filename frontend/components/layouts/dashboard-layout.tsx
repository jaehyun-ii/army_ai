"use client"

import type { ReactNode } from 'react'
import { DashboardTopbar } from './dashboard-topbar'
import { DashboardSidebar, type DashboardMenuItem } from './dashboard-sidebar'
import { AdminSidebar, type AdminMenuItem } from './admin-sidebar'
import { DashboardBottomBar } from './dashboard-bottom-bar'
import { useAuth } from '@/contexts/AuthContext'

export type { DashboardMenuItem }

interface DashboardLayoutProps {
  menuItems: DashboardMenuItem[]
  activeSection: string
  onSelectSection: (section: string) => void
  expandedMenus: string[]
  onToggleMenu: (menuName: string) => void
  children: ReactNode
}

export function DashboardLayout({
  menuItems,
  activeSection,
  onSelectSection,
  expandedMenus,
  onToggleMenu,
  children
}: DashboardLayoutProps) {
  const { user, loading } = useAuth()
  const isAdmin = user?.role === 'admin'

  console.log('Dashboard Layout - User:', user)
  console.log('Dashboard Layout - User Role:', user?.role)
  console.log('Dashboard Layout - Is Admin:', isAdmin)
  console.log('Dashboard Layout - Loading:', loading)

  return (
    <div className="h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex flex-col overflow-hidden">
      <DashboardTopbar />

      <div className="flex-1 flex flex-col overflow-hidden">
        <div className="flex-1 flex overflow-hidden">
          {loading ? (
            // Loading skeleton for sidebar
            <aside className="w-72 bg-gradient-to-b from-slate-900/95 via-slate-800/95 to-slate-900/95 backdrop-blur-sm border-r border-white/10 flex flex-col">
              <div className="flex-1 overflow-y-auto p-3">
                <div className="space-y-1">
                  <div className="h-4 bg-slate-700/50 rounded w-24 mb-3 animate-pulse" />
                  {[...Array(5)].map((_, i) => (
                    <div key={i} className="h-10 bg-slate-700/30 rounded animate-pulse" />
                  ))}
                </div>
              </div>
            </aside>
          ) : isAdmin ? (
            <AdminSidebar
              menuItems={menuItems as AdminMenuItem[]}
              activeSection={activeSection}
              expandedMenus={expandedMenus}
              onSelectSection={onSelectSection}
              onToggleMenu={onToggleMenu}
            />
          ) : (
            <DashboardSidebar
              menuItems={menuItems}
              activeSection={activeSection}
              expandedMenus={expandedMenus}
              onSelectSection={onSelectSection}
              onToggleMenu={onToggleMenu}
            />
          )}

          <main className="flex-1 p-6 overflow-hidden bg-gradient-to-br from-slate-900/50 via-slate-800/50 to-slate-900/50">
            <div className="h-full flex flex-col">{children}</div>
          </main>
        </div>

        <DashboardBottomBar />
      </div>
    </div>
  )
}
