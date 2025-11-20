"use client"

import NextImage from 'next/image'
import type { LucideIcon } from 'lucide-react'
import { ChevronDown } from 'lucide-react'

export interface AdminMenuItem {
  name: string
  icon: LucideIcon
  content?: string
  children?: Array<{ name: string; content: string }>
}

interface AdminSidebarProps {
  menuItems: AdminMenuItem[]
  activeSection: string
  expandedMenus: string[]
  onSelectSection: (section: string) => void
  onToggleMenu: (menuName: string) => void
}

export function AdminSidebar({
  menuItems,
  activeSection,
  expandedMenus,
  onSelectSection,
  onToggleMenu
}: AdminSidebarProps) {
  return (
    <aside className="w-72 bg-gradient-to-b from-slate-900/95 via-slate-800/95 to-slate-900/95 backdrop-blur-sm border-r border-white/10 flex flex-col relative">
      <div className="flex-1 overflow-y-auto p-3 scrollbar-thin">
        <div className="space-y-1">
          <p className="text-red-400 text-xs font-medium uppercase tracking-wider mb-3">관리자 메뉴</p>

          {menuItems.map((item) => {
            const IconComponent = item.icon
            const isActive = activeSection === item.name
            const isExpanded = expandedMenus.includes(item.name)
            const hasChildren = (item.children?.length ?? 0) > 0

            return (
              <div key={item.name} className="space-y-1">
                <button
                  className={`w-full flex items-center gap-2 px-3 py-2 rounded-lg text-left transition-all duration-300 ${
                    isActive && !hasChildren
                      ? 'bg-gradient-to-r from-red-600/20 to-orange-600/20 text-white border border-red-500/30 shadow-md'
                      : 'text-slate-300 hover:text-white hover:bg-gradient-to-r hover:from-slate-800/60 hover:to-slate-700/60'
                  }`}
                  onClick={() => {
                    if (hasChildren) {
                      onToggleMenu(item.name)
                    } else {
                      onSelectSection(item.name)
                    }
                  }}
                >
                  <IconComponent className="w-4 h-4" />
                  <span className="text-sm flex-1">{item.name}</span>
                  {hasChildren && (
                    <ChevronDown
                      className={`w-4 h-4 transition-transform duration-200 ${
                        isExpanded ? 'rotate-180' : ''
                      }`}
                    />
                  )}
                </button>

                {hasChildren && isExpanded && (
                  <div className="ml-6 space-y-1">
                    {item.children!.map((child) => (
                      <button
                        key={child.name}
                        className={`w-full flex items-center gap-2 px-3 py-2 rounded-md text-sm transition-all duration-300 ${
                          activeSection === child.name
                            ? 'text-white bg-gradient-to-r from-red-600/15 to-orange-600/15 border-l-2 border-red-500/50'
                            : 'text-slate-400 hover:text-white hover:bg-gradient-to-r hover:from-slate-800/40 hover:to-slate-700/40'
                        }`}
                        onClick={() => onSelectSection(child.name)}
                      >
                        <span>• {child.name}</span>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>

      <div className="p-4 border-t border-white/10">
        <div className="flex items-center gap-3 justify-center">
          <div className="w-10 h-10 relative flex items-center justify-center">
            <NextImage
              src="/army_logos.png"
              alt="육군 로고"
              width={40}
              height={40}
              className="object-contain"
            />
          </div>
          <div className="flex flex-col">
            <p className="text-lg font-bold text-white">육군인공지능센터</p>
            <p className="text-xs text-red-400">관리자 모드</p>
          </div>
        </div>
      </div>
    </aside>
  )
}