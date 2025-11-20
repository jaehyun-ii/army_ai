import { LoginFormDB } from "@/components/login-form-db"

export default function LoginPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-600 to-slate-900 flex items-center justify-center p-4 relative overflow-hidden">
      <div className="absolute inset-0 bg-[url('/subtle-military-pattern.jpg')] opacity-5"></div>
      <div className="absolute inset-0 bg-gradient-to-br from-primary/10 via-transparent to-accent/10"></div>

      <div className="w-full max-w-6xl grid gap-8 items-center relative z-10">
        <div className="w-full max-w-md mx-auto">
          <LoginFormDB />
        </div>
      </div>
    </div>
  )
}