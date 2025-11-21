'use client'

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ReactQueryDevtools } from '@tanstack/react-query-devtools'
import { useState, type ReactNode } from 'react'

export function Providers({ children }: { children: ReactNode }) {
  const [queryClient] = useState(() => new QueryClient({
    defaultOptions: {
      queries: {
        staleTime: 60 * 1000, // 1분: 데이터가 신선하다고 간주되는 시간
        gcTime: 5 * 60 * 1000, // 5분 (이전 cacheTime): 캐시 유지 시간
        retry: 1, // 실패 시 재시도 횟수
        refetchOnWindowFocus: false, // 윈도우 포커스 시 자동 리페치 비활성화
        refetchOnMount: true, // 컴포넌트 마운트 시 리페치
        refetchOnReconnect: true, // 재연결 시 리페치
      },
      mutations: {
        retry: 0, // 뮤테이션은 재시도하지 않음
      },
    },
  }))

  return (
    <QueryClientProvider client={queryClient}>
      {children}
      {process.env.NODE_ENV === 'development' && (
        <ReactQueryDevtools initialIsOpen={false} />
      )}
    </QueryClientProvider>
  )
}
