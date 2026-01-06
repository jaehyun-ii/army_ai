"use client"

import NextImage from 'next/image'
import type { LucideIcon } from 'lucide-react'
import { ChevronDown } from 'lucide-react'

export interface AdminMenuItem {
  name: string
  icon: LucideIcon
  content?: string
  children?: Array<{ name: string; content: string }>
  onClick?: () => void
}

interface AdminSidebarProps {
  menuItems: AdminMenuItem[]
  activeSection: string
  expandedMenus: string[]
  onSelectSection: (section: string) => void
  onToggleMenu: (menuName: string) => void
  disabled?: boolean
}

export function AdminSidebar({
  menuItems,
  activeSection,
  expandedMenus,
  onSelectSection,
  onToggleMenu,
  disabled = false
}: AdminSidebarProps) {
  return (
    <aside className={`w-48 md:w-56 lg:w-72 bg-gradient-to-b from-[var(--surface)]/94 via-[var(--surface-strong)]/96 to-[var(--surface)]/94 backdrop-blur-xl border-r border-border flex flex-col relative shadow-[14px_0_50px_-24px_rgba(0,0,0,0.35)] ${disabled ? 'pointer-events-none opacity-50' : ''}`}>
      <div className="flex-1 overflow-y-auto p-3 scrollbar-thin">
        <div className="space-y-1">
          <p className="text-error text-xs font-medium uppercase tracking-wider mb-3 px-1">관리자 메뉴</p>

          {menuItems.map((item) => {
            const IconComponent = item.icon
            const isActive = activeSection === item.name
            const isExpanded = expandedMenus.includes(item.name)
            const hasChildren = (item.children?.length ?? 0) > 0

            return (
              <div key={item.name} className="space-y-1">
                <button
                  className={`w-full flex items-center gap-3 px-3 py-3 rounded-xl text-left transition-all duration-200 ring-1 ring-transparent ${
                    isActive && !hasChildren
                      ? 'bg-gradient-to-r from-red-600/18 to-orange-500/18 text-foreground font-semibold shadow-[0_12px_30px_-18px_rgba(0,0,0,0.35)] ring-error/30'
                      : 'text-muted hover:text-on-surface hover:bg-[var(--surface-strong)]/80 hover:ring-1 hover:ring-border'
                  }`}
                  onClick={() => {
                    if (hasChildren) {
                      onToggleMenu(item.name)
                    } else if (item.onClick) {
                      item.onClick()
                    } else {
                      onSelectSection(item.name)
                    }
                  }}
                >
                  <IconComponent className="w-4 h-4" />
                  <span className="text-sm flex-1 truncate">{item.name}</span>
                  {hasChildren && (
                    <ChevronDown
                      className={`w-4 h-4 transition-transform duration-200 ${
                        isExpanded ? 'rotate-180' : ''
                      }`}
                    />
                  )}
                </button>

                {hasChildren && isExpanded && (
                  <div className="ml-4 space-y-1 border-l border-border pl-2 py-1 my-1">
                    {item.children!.map((child) => (
                      <button
                        key={child.name}
                        className={`w-full flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-all duration-200 ring-1 ring-transparent ${
                          activeSection === child.name
                            ? 'bg-gradient-to-r from-red-600/15 to-orange-600/15 text-foreground font-medium shadow-[0_10px_24px_-16px_rgba(0,0,0,0.32)] ring-error/30'
                            : 'text-muted hover:text-on-surface hover:bg-[var(--surface-strong)]/70 hover:ring-1 hover:ring-border'
                        }`}
                        onClick={() => onSelectSection(child.name)}
                      >
                        <span className="truncate">{child.name}</span>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>

      <div className="p-4 border-t border-border bg-[var(--surface)]/90 backdrop-blur-xl">
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
            <p className="text-lg font-bold text-secondary">육군시험평가단</p>
            <p className="text-xs text-error">관리자 모드</p>
          </div>
        </div>
      </div>
    </aside>
  )
}
