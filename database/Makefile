.PHONY: help up down restart logs clean backup restore shell psql test

# Default target
help:
	@echo "Adversarial Vision Platform - Database Management"
	@echo ""
	@echo "Available targets:"
	@echo "  make up         - Start PostgreSQL and pgAdmin containers"
	@echo "  make down       - Stop and remove containers"
	@echo "  make restart    - Restart all containers"
	@echo "  make logs       - Show container logs"
	@echo "  make clean      - Remove containers and volumes (WARNING: deletes all data)"
	@echo "  make backup     - Backup database to SQL file"
	@echo "  make restore    - Restore database from backup file"
	@echo "  make shell      - Open bash shell in PostgreSQL container"
	@echo "  make psql       - Open PostgreSQL interactive terminal"
	@echo "  make test       - Test database connection"
	@echo ""

# Start services
up:
	@echo "Starting PostgreSQL and pgAdmin..."
	docker-compose up -d
	@echo "Services started. PostgreSQL: localhost:5432, pgAdmin: http://localhost:5050"

# Stop services
down:
	@echo "Stopping services..."
	docker-compose down

# Restart services
restart:
	@echo "Restarting services..."
	docker-compose restart

# Show logs
logs:
	docker-compose logs -f

# Clean up (removes volumes)
clean:
	@echo "WARNING: This will delete all database data!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		docker-compose down -v; \
		echo "All data removed."; \
	else \
		echo "Cancelled."; \
	fi

# Backup database
backup:
	@echo "Creating backup..."
	@mkdir -p backups
	@TIMESTAMP=$$(date +%Y%m%d_%H%M%S); \
	docker-compose exec -T postgres pg_dump -U admin adversarial_vision > backups/backup_$$TIMESTAMP.sql
	@echo "Backup created: backups/backup_$$TIMESTAMP.sql"

# Restore database from backup
restore:
	@if [ -z "$(FILE)" ]; then \
		echo "Usage: make restore FILE=backups/backup_YYYYMMDD_HHMMSS.sql"; \
		exit 1; \
	fi
	@echo "Restoring from $(FILE)..."
	docker-compose exec -T postgres psql -U admin adversarial_vision < $(FILE)
	@echo "Database restored successfully."

# Shell access to PostgreSQL container
shell:
	docker-compose exec postgres /bin/bash

# PostgreSQL interactive terminal
psql:
	docker-compose exec postgres psql -U admin -d adversarial_vision

# Test database connection
test:
	@echo "Testing database connection..."
	@docker-compose exec postgres pg_isready -U admin -d adversarial_vision && \
		echo "Database is ready!" || \
		echo "Database is not ready."
