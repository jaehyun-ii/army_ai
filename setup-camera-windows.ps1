# ============================================
# Army AI Platform - Camera Setup (Clean All)
# ============================================

Write-Host "--- Army AI Platform Camera Setup ---" -ForegroundColor Cyan
Write-Host ""

# 1. Admin Check
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "[ERROR] This script requires Administrator privileges." -ForegroundColor Red
    pause
    exit 1
}

# 2. Check usbipd
if (-not (Get-Command usbipd -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] usbipd-win is not installed." -ForegroundColor Red
    pause
    exit 1
}

# 3. Check WSL2
$wslStatus = wsl --status 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] WSL2 is not installed or running." -ForegroundColor Red
    pause
    exit 1
}

# ============================================
# 4. GLOBAL RESET (모든 장치 초기화)
# ============================================
Write-Host "--- 0. Cleaning All Connections ---" -ForegroundColor Cyan
Write-Host "Checking for active connections..." -ForegroundColor Yellow

# "Shared" 또는 "Attached" 상태인 모든 장치 검색
$activeLines = usbipd list | Select-String "Shared|Attached"

if ($activeLines) {
    foreach ($line in $activeLines) {
        # BUSID 추출 (예: 4-6, 6-1)
        if ($line -match '(\d+-\d+)') {
            $targetBusId = $matches[1]
            Write-Host "   ♻️  Resetting BUSID: $targetBusId ..." -NoNewline
            
            # Detach (WSL 연결 해제)
            usbipd detach --busid $targetBusId 2>&1 | Out-Null
            
            # Unbind (공유 해제)
            usbipd unbind --busid $targetBusId 2>&1 | Out-Null
            
            Write-Host " Done." -ForegroundColor Gray
        }
    }
    Write-Host "   ✅ All devices disconnected." -ForegroundColor Green
} else {
    Write-Host "   ℹ️  No active devices found. Clean start." -ForegroundColor Gray
}

# 잠시 대기 (WSL 커널이 장치 해제를 인식할 시간)
Start-Sleep -Seconds 2
Write-Host ""

# ============================================
# 5. Find & Connect Camera
# ============================================
Write-Host "--- 1. Selecting Camera ---" -ForegroundColor Cyan
Write-Host "Searching for camera..." -ForegroundColor Yellow
$devices = usbipd list | Select-String "Camera|Webcam|Video|Capture|USB.*Video|Integrated Camera|Logitech|BRIO"
$busId = ""

if ($devices.Count -eq 0) {
    usbipd list
    Write-Host "[WARN] No camera found automatically." -ForegroundColor Yellow
    $busId = Read-Host "Enter BUSID manually (e.g., 4-6)"
} else {
    Write-Host "Found devices:" -ForegroundColor Green
    $devices | ForEach-Object { Write-Host $_ -ForegroundColor White }
    
    $firstDevice = $devices[0].ToString()
    if ($firstDevice -match '(\d+-\d+)') {
        $foundBusId = $matches[1]
        Write-Host "Auto-detected BUSID: $foundBusId" -ForegroundColor Cyan
        $confirm = Read-Host "Use this device? (Enter for Yes, or type new BUSID)"
        
        if ($confirm -eq '' -or $confirm -eq 'y' -or $confirm -eq 'Y') {
            $busId = $foundBusId
        } else {
            $busId = $confirm
        }
    } else {
        $busId = Read-Host "Enter BUSID manually"
    }
}

if ([string]::IsNullOrWhiteSpace($busId)) {
    Write-Host "[ERROR] No BUSID provided." -ForegroundColor Red
    pause
    exit 1
}

# ============================================
# 6. New Connection
# ============================================
Write-Host ""
Write-Host "--- 2. Connecting to WSL2 ---" -ForegroundColor Cyan

# Bind
Write-Host "Binding device ($busId)..." -ForegroundColor Yellow
usbipd bind --busid $busId 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "   [OK] Bind success" -ForegroundColor Green
} else {
    Write-Host "   [INFO] Check if another app is using the camera." -ForegroundColor Yellow
}

# Attach
Write-Host "Attaching to WSL..." -ForegroundColor Yellow
usbipd attach --wsl --busid $busId
if ($LASTEXITCODE -eq 0) {
    Write-Host "   [OK] Attached successfully" -ForegroundColor Green
} else {
    Write-Host "   [ERROR] Failed to attach." -ForegroundColor Red
    pause
    exit 1
}

# Verify
Write-Host "Verifying in WSL..." -ForegroundColor Yellow
# WSL 내부에서 드라이버를 다시 로드하도록 유도 (선택 사항)
# wsl sudo modprobe -r uvcvideo; wsl sudo modprobe uvcvideo
wsl ls -la /dev/video*

if ($LASTEXITCODE -eq 0) {
    Write-Host "   [OK] /dev/video* exists." -ForegroundColor Green
} else {
    Write-Host "   [WARN] /dev/video* not found (Check drivers in WSL)." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "--- Setup Complete ---" -ForegroundColor Green
pause