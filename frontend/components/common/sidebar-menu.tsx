import { ChevronDown } from "lucide-react"
import { LucideIcon } from "lucide-react"
import { cn } from "@/lib/utils"

export interface MenuItem {
  name: string
  icon?: LucideIcon
  content?: string
  children?: {
    name: string
    content: string
  }[]
}

interface SidebarMenuProps {
  items: MenuItem[]
  activeSection: string
  expandedMenus: string[]
  onMenuClick: (name: string) => void
  onToggleMenu: (name: string) => void
  className?: string
}

export function SidebarMenu({
  items,
  activeSection,
  expandedMenus,
  onMenuClick,
  onToggleMenu,
  className
}: SidebarMenuProps) {
  return (
    <div className={cn("space-y-1", className)}>
      <p className="text-slate-400 text-xs font-medium uppercase tracking-wider mb-3">
        신뢰성 검증 메뉴
      </p>

      {items.map((item, index) => {
        const IconComponent = item.icon
        const isActive = activeSection === item.name
        const isExpanded = expandedMenus.includes(item.name)
        const hasChildren = item.children && item.children.length > 0

        return (
          <div key={index} className="space-y-1">
            <button
              className={cn(
                "w-full flex items-center gap-2 px-3 py-2 rounded-lg text-left transition-all duration-300",
                isActive && !hasChildren
                  ? "bg-gradient-to-r from-primary/20 to-accent/20 text-white border border-primary/30 shadow-md"
                  : "text-slate-300 hover:text-white hover:bg-gradient-to-r hover:from-slate-800/60 hover:to-slate-700/60"
              )}
              onClick={() => {
                if (hasChildren) {
                  onToggleMenu(item.name)
                } else {
                  onMenuClick(item.name)
                }
              }}
            >
              {IconComponent && <IconComponent className="w-4 h-4" />}
              <span className="text-sm flex-1">{item.name}</span>
              {hasChildren && (
                <ChevronDown
                  className={cn(
                    "w-4 h-4 transition-transform duration-200",
                    isExpanded && "rotate-180"
                  )}
                />
              )}
            </button>

            {hasChildren && isExpanded && item.children && (
              <div className="ml-6 space-y-1">
                {item.children.map((child, childIndex) => (
                  <button
                    key={childIndex}
                    className={cn(
                      "w-full flex items-center gap-2 px-3 py-2 rounded-md text-sm transition-all duration-300",
                      activeSection === child.name
                        ? "text-white bg-gradient-to-r from-primary/15 to-accent/15 border-l-2 border-primary/50"
                        : "text-slate-400 hover:text-white hover:bg-gradient-to-r hover:from-slate-800/40 hover:to-slate-700/40"
                    )}
                    onClick={() => onMenuClick(child.name)}
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
  )
}