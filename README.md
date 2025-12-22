# Army AI Platform

Adversarial Vision Platform for AI Security Research

## 빠른 시작

### Linux 환경
```bash
docker-compose up -d
```

### Windows 환경 (WSL2)
Windows에서 배포하는 경우 [Windows 배포 가이드](./WINDOWS_DEPLOYMENT.md)를 참조하세요.

특히 USB 카메라를 사용하는 경우:
1. **setup-camera-windows.ps1** 스크립트를 관리자 권한으로 실행
2. Docker Compose 실행

```powershell
# Windows PowerShell (관리자 권한)
.\setup-camera-windows.ps1

# WSL2 터미널
docker-compose up -d
```

## 문서

- [Windows 배포 가이드](./WINDOWS_DEPLOYMENT.md) - WSL2 + Docker Desktop 사용법
- Docker Compose 설정:
  - `docker-compose.yml` - Linux/WSL2 (카메라 있음)
  - `docker-compose.windows.yml` - Windows 전용 (카메라 없음)
  - `docker-compose.pro.yml` - 프로덕션


