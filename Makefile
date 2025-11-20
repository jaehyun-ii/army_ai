# ===============================================
# Army AI Platform - Makefile
# ===============================================

.PHONY: help cleanup dev prod down logs build test

# Default target
help:
	@echo "═══════════════════════════════════════════════════════"
	@echo "  Army AI Platform - Available Commands"
	@echo "═══════════════════════════════════════════════════════"
	@echo ""
	@echo "  Development:"
	@echo "    make cleanup       - Remove cache and temporary files"
	@echo "    make dev           - Start development environment"
	@echo "    make dev-logs      - View development logs"
	@echo ""
	@echo "  Production:"
	@echo "    make prod          - Start production environment (GPU)"
	@echo "    make build         - Build Docker images"
	@echo "    make logs          - View production logs"
	@echo ""
	@echo "  Maintenance:"
	@echo "    make down          - Stop all containers"
	@echo "    make down-clean    - Stop containers and remove volumes"
	@echo "    make restart       - Restart all containers"
	@echo ""
	@echo "  Testing:"
	@echo "    make test          - Run backend tests"
	@echo "    make test-cov      - Run tests with coverage"
	@echo ""
	@echo "═══════════════════════════════════════════════════════"

# Cleanup cache and temporary files
cleanup:
	@echo "🧹 Cleaning up cache and temporary files..."
	@./cleanup.sh

# Development environment
dev:
	@echo "🚀 Starting development environment..."
	@docker-compose -f docker-compose.dev.yml up -d
	@echo "✅ Development environment started!"
	@echo "   - Frontend: http://localhost:3000"
	@echo "   - Backend:  http://localhost:8000"
	@echo "   - API Docs: http://localhost:8000/api/v1/docs"

dev-logs:
	@docker-compose -f docker-compose.dev.yml logs -f

dev-down:
	@docker-compose -f docker-compose.dev.yml down

# Production environment (GPU-accelerated with NGC PyTorch)
prod:
	@echo "🚀 Starting GPU-accelerated environment (NVIDIA NGC PyTorch)..."
	@docker-compose up -d
	@echo "✅ Environment started!"
	@echo "   - Frontend: http://localhost:3000"
	@echo "   - Backend:  http://localhost:8000"
	@echo "   - API Docs: http://localhost:8000/api/v1/docs"
	@echo ""
	@echo "🎮 Verifying GPU..."
	@sleep 5
	@docker-compose exec backend python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" || echo "⚠️  GPU check failed"

build:
	@echo "🔨 Building Docker images with NVIDIA NGC PyTorch..."
	@docker-compose build

logs:
	@docker-compose logs -f

# Stop containers
down:
	@echo "🛑 Stopping all containers..."
	@docker-compose down
	@docker-compose -f docker-compose.dev.yml down 2>/dev/null || true
	@echo "✅ All containers stopped"

down-clean:
	@echo "🛑 Stopping all containers and removing volumes..."
	@docker-compose down -v
	@docker-compose -f docker-compose.dev.yml down -v 2>/dev/null || true
	@echo "✅ All containers and volumes removed"

restart:
	@echo "🔄 Restarting containers..."
	@docker-compose restart
	@echo "✅ Containers restarted"

# Testing
test:
	@echo "🧪 Running backend tests..."
	@cd backend && python -m pytest

test-cov:
	@echo "🧪 Running tests with coverage..."
	@cd backend && python -m pytest --cov=app --cov-report=html --cov-report=term

# Database
db-shell:
	@docker-compose exec postgres psql -U admin -d armydb

db-backup:
	@echo "💾 Backing up database..."
	@docker-compose exec postgres pg_dump -U admin armydb > backup_$$(date +%Y%m%d_%H%M%S).sql
	@echo "✅ Database backed up"

# Status
status:
	@echo "📊 Container Status:"
	@docker-compose ps
	@echo ""
	@echo "📦 Volume Usage:"
	@docker volume ls | grep army_ai
