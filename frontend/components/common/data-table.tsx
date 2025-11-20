import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { LucideIcon } from "lucide-react"
import { styles } from "@/lib/styles"

export interface Column<T> {
  key: keyof T | string
  header: string
  width?: string
  align?: "left" | "center" | "right"
  render?: (value: any, item: T) => React.ReactNode
}

export interface DataTableProps<T> {
  title?: string
  description?: string
  data: T[]
  columns: Column<T>[]
  actions?: (item: T) => React.ReactNode
  onRowClick?: (item: T) => void
  selectedRow?: T
  emptyMessage?: string
  className?: string
  headerActions?: React.ReactNode
}

export function DataTable<T extends Record<string, any>>({
  title,
  description,
  data,
  columns,
  actions,
  onRowClick,
  selectedRow,
  emptyMessage = "데이터가 없습니다",
  className,
  headerActions
}: DataTableProps<T>) {
  const getValue = (item: T, key: string) => {
    const keys = key.split('.')
    let value: any = item
    for (const k of keys) {
      value = value?.[k]
    }
    return value
  }

  return (
    <Card className={cn(styles.card.base, className)}>
      {(title || headerActions) && (
        <CardHeader className="flex flex-row items-center justify-between">
          <div>
            {title && <CardTitle className="text-white">{title}</CardTitle>}
            {description && (
              <p className="text-slate-400 text-sm mt-1">{description}</p>
            )}
          </div>
          {headerActions}
        </CardHeader>
      )}
      <CardContent className="p-4">
        <div className="border border-white/10 rounded-lg overflow-hidden">
          <Table>
            <TableHeader>
              <TableRow className="border-white/10 bg-slate-800/50">
                {columns.map((column, index) => (
                  <TableHead
                    key={index}
                    className={cn(
                      "text-slate-200 font-semibold h-12 px-6",
                      column.width,
                      column.align === "center" && "text-center",
                      column.align === "right" && "text-right"
                    )}
                  >
                    {column.header}
                  </TableHead>
                ))}
                {actions && <TableHead className="text-right text-slate-200 font-semibold h-12 px-6">작업</TableHead>}
              </TableRow>
            </TableHeader>
            <TableBody>
              {data.length === 0 ? (
                <TableRow>
                  <TableCell
                    colSpan={columns.length + (actions ? 1 : 0)}
                    className="text-center text-slate-400 py-12 px-6"
                  >
                    {emptyMessage}
                  </TableCell>
                </TableRow>
              ) : (
                data.map((item, rowIndex) => (
                  <TableRow
                    key={rowIndex}
                    className={cn(
                      "border-white/10 hover:bg-slate-700/40 transition-colors",
                      onRowClick && "cursor-pointer",
                      selectedRow === item && "bg-slate-700/50",
                      rowIndex % 2 === 0 ? "bg-slate-900/20" : "bg-slate-900/40"
                    )}
                    onClick={() => onRowClick?.(item)}
                  >
                    {columns.map((column, colIndex) => {
                      const value = getValue(item, column.key as string)
                      return (
                        <TableCell
                          key={colIndex}
                          className={cn(
                            "text-slate-200 py-4 px-6",
                            column.align === "center" && "text-center",
                            column.align === "right" && "text-right"
                          )}
                        >
                          {column.render ? column.render(value, item) : value}
                        </TableCell>
                      )
                    })}
                    {actions && (
                      <TableCell className="text-right py-4 px-6">
                        {actions(item)}
                      </TableCell>
                    )}
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </div>
      </CardContent>
    </Card>
  )
}

export function StatusBadge({ status, variant }: { status: string; variant?: any }) {
  const getStatusStyle = (status: string) => {
    const lowerStatus = status.toLowerCase()

    if (lowerStatus === "완료" || lowerStatus === "completed" || lowerStatus === "success") {
      return "bg-green-900/30 text-green-400 border-green-500/30"
    }
    if (lowerStatus === "진행중" || lowerStatus === "active" || lowerStatus === "running") {
      return "bg-blue-900/30 text-blue-400 border-blue-500/30"
    }
    if (lowerStatus === "대기" || lowerStatus === "pending" || lowerStatus === "waiting") {
      return "bg-yellow-900/30 text-yellow-400 border-yellow-500/30"
    }
    if (lowerStatus === "실패" || lowerStatus === "failed" || lowerStatus === "error") {
      return "bg-red-900/30 text-red-400 border-red-500/30"
    }
    return "bg-slate-800/50 text-slate-400 border-slate-600/30"
  }

  return (
    <Badge className={getStatusStyle(status)}>
      {status}
    </Badge>
  )
}