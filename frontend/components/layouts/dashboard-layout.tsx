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
  const { user } = useAuth()
  const isAdmin = user?.role === 'admin'

  console.log('Dashboard Layout - User:', user)
  console.log('Dashboard Layout - User Role:', user?.role)
  console.log('Dashboard Layout - Is Admin:', isAdmin)

  return (
    <div className="h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex flex-col overflow-hidden">
      <DashboardTopbar />

      <div className="flex-1 flex flex-col overflow-hidden">
        <div className="flex-1 flex overflow-hidden">
          {isAdmin ? (
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
