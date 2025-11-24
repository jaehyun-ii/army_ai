# 개발 환경 가이드

## 🔥 Hot Reload 개발 환경

코드 변경 시 **자동으로 반영**되는 개발 환경 설정입니다.

### 특징
- ✅ 파일 저장 시 자동 새로고침 (Hot Module Replacement)
- ✅ 소스 코드 볼륨 마운트
- ✅ 더 자세한 로그 출력
- ✅ 개발자 도구 활성화

---

## 사용 방법

### 1. 개발 환경 시작

```bash
# 기존 프로덕션 컨테이너 중지 (선택사항)
docker-compose down

# 개발 환경 시작
docker-compose -f docker-compose.dev.yml up --build
```

### 2. 백그라운드로 실행

```bash
docker-compose -f docker-compose.dev.yml up --build -d
```

### 3. 로그 확인

```bash
# 모든 서비스 로그
docker-compose -f docker-compose.dev.yml logs -f

# Frontend만
docker-compose -f docker-compose.dev.yml logs -f frontend

# Backend만
docker-compose -f docker-compose.dev.yml logs -f backend
```

### 4. 중지

```bash
docker-compose -f docker-compose.dev.yml down
```

---

## 코드 수정 후

### ✅ 자동 반영되는 것들
- **Frontend (Next.js)**:
  - React 컴포넌트 (`.tsx`, `.jsx`)
  - 스타일 파일 (`.css`)
  - API Routes (`/app/api/**/*`)
  - 저장하면 **2-3초 내 브라우저에 자동 반영**

### ⚠️ 재시작 필요한 것들
- **환경 변수 변경** (`.env` 파일)
- **package.json 변경** (새 패키지 설치)
- **Docker 설정 변경** (`docker-compose.dev.yml`, `Dockerfile.dev`)

재시작 방법:
```bash
# Frontend만 재시작
docker-compose -f docker-compose.dev.yml restart frontend

# 전체 재빌드
docker-compose -f docker-compose.dev.yml up --build -d
```

---

## 접속 주소

- **Frontend**: http://localhost:54322
- **Backend API**: http://localhost:54321
- **PostgreSQL**: localhost:54320

---

## 프로덕션 환경으로 전환

```bash
# 개발 환경 중지
docker-compose -f docker-compose.dev.yml down

# 프로덕션 환경 시작
docker-compose up --build -d
```

---

## 문제 해결

### 1. 코드가 반영되지 않을 때

```bash
# 캐시 삭제 후 재빌드
docker-compose -f docker-compose.dev.yml down -v
docker-compose -f docker-compose.dev.yml up --build
```

### 2. "ENOSPC: no space left on device" 에러

```bash
# 파일 감시자 제한 늘리기 (Linux/WSL)
echo fs.inotify.max_user_watches=524288 | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

### 3. 포트 충돌

다른 서비스가 포트를 사용 중인 경우:
```bash
# 사용 중인 포트 확인
sudo lsof -i :54322  # Frontend
sudo lsof -i :54321  # Backend
sudo lsof -i :54320  # PostgreSQL

# 프로세스 종료 또는 .env에서 포트 변경
```

---

## 성능 최적화 팁

1. **node_modules 볼륨**:
   - Named volume으로 관리하여 I/O 성능 향상
   - 삭제 방법: `docker volume rm army_ai_frontend_node_modules`

2. **.next 캐시 볼륨**:
   - 빌드 캐시를 유지하여 재시작 속도 향상

3. **파일 감시 설정**:
   - `WATCHPACK_POLLING=true`: Docker 환경에서 파일 변경 감지
   - `CHOKIDAR_USEPOLLING=true`: 대체 파일 감시 방법
