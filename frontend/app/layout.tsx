import type { Metadata } from 'next'
import { GeistSans } from 'geist/font/sans'
import { GeistMono } from 'geist/font/mono'
import { AuthProvider } from '@/contexts/AuthContext'
import { OperationProvider } from '@/contexts/OperationContext'
import { Toaster } from '@/components/ui/toaster'
import { Providers } from './providers'
import './globals.css'
import '../css/light.css'
import '../css/dark.css'
import '../css/light-hc.css'
import '../css/dark-hc.css'
import '../css/light-mc.css'
import '../css/dark-mc.css'

export const metadata: Metadata = {
  title: '객체식별 AI 모델 신뢰성 검증 실증 체계',
  description: 'AI Model Reliability Verification System',
  generator: 'Next.js',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="ko" className="light" suppressHydrationWarning>
      <body className={`font-sans ${GeistSans.variable} ${GeistMono.variable}`}>
        <Providers>
          <AuthProvider>
            <OperationProvider>
              {children}
            </OperationProvider>
          </AuthProvider>
        </Providers>
        <Toaster />
      </body>
    </html>
  )
}
