"""
Admin endpoints for system management.
"""
import logging
import asyncio
import os
import shutil
import subprocess
import tarfile
import zipfile
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status, UploadFile, File, Form
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert

from app import crud, schemas
from app.database import get_db, AsyncSessionLocal, engine
from app.core.config import settings
from app.models.backup import Backup

logger = logging.getLogger(__name__)

router = APIRouter()

# Prevent concurrent backup/restore jobs from clobbering each other
_backup_restore_lock = asyncio.Lock()


def get_database_config():
    """Extract database configuration from DATABASE_URL."""
    db_url = settings.DATABASE_URL

    # Parse PostgreSQL URL: postgresql+asyncpg://user:password@host:port/dbname
    # or postgresql://user:password@host:port/dbname
    if "postgresql" in db_url:
        # Remove asyncpg suffix if present
        db_url = db_url.replace("+asyncpg", "")

        # Parse URL
        parts = db_url.replace("postgresql://", "").split("@")
        user_pass = parts[0].split(":")
        host_db = parts[1].split("/")
        host_port = host_db[0].split(":")

        return {
            "type": "postgresql",
            "user": user_pass[0],
            "password": user_pass[1] if len(user_pass) > 1 else "",
            "host": host_port[0],
            "port": host_port[1] if len(host_port) > 1 else "5432",
            "database": host_db[1].split("?")[0] if len(host_db) > 1 else ""
        }
    else:
        raise ValueError("Unsupported database type. Only PostgreSQL is supported for backups.")


async def create_db_backup(backup_id: str, backup_dir: Path) -> tuple[str, int]:
    """Create a database backup using pg_dump."""
    try:
        db_config = get_database_config()

        # Create backup filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        db_backup_file = backup_dir / f"db_{backup_id}_{timestamp}.sql"

        # Set PostgreSQL password environment variable
        env = os.environ.copy()
        env["PGPASSWORD"] = db_config["password"]

        # Run pg_dump
        cmd = [
            "pg_dump",
            "-h", db_config["host"],
            "-p", db_config["port"],
            "-U", db_config["user"],
            "-d", db_config["database"],
            "-F", "c",  # Custom format (compressed)
            "-f", str(db_backup_file)
        ]

        logger.info(f"Running pg_dump: {' '.join(cmd[:-2])} -f {db_backup_file}")

        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=settings.BACKUP_TIMEOUT
        )

        if result.returncode != 0:
            error_msg = f"pg_dump failed: {result.stderr}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Get file size
        file_size = db_backup_file.stat().st_size
        logger.info(f"Database backup created: {db_backup_file} ({file_size} bytes)")

        return str(db_backup_file), file_size

    except Exception as e:
        logger.error(f"Failed to create database backup: {e}", exc_info=True)
        raise


async def create_storage_backup(backup_id: str, backup_dir: Path) -> tuple[str, int]:
    """Create a storage backup by compressing the storage directory."""
    try:
        # Create backup filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        storage_backup_file = backup_dir / f"storage_{backup_id}_{timestamp}.tar.gz"

        # Get storage root path
        storage_root = settings.STORAGE_ROOT

        logger.info(f"Creating storage backup from: {storage_root}")

        # Create tar.gz archive
        # No need to exclude backups since they're now outside storage directory
        with tarfile.open(storage_backup_file, "w:gz") as tar:
            tar.add(storage_root, arcname="storage")

        # Get file size
        file_size = storage_backup_file.stat().st_size
        logger.info(f"Storage backup created: {storage_backup_file} ({file_size} bytes)")

        return str(storage_backup_file), file_size

    except Exception as e:
        logger.error(f"Failed to create storage backup: {e}", exc_info=True)
        raise


async def perform_backup(backup_id: str):
    """Background task to perform the actual backup."""
    logger.info(f"Starting backup task for backup_id: {backup_id}")

    # Track backup results
    backup_success = False
    db_file_path = None
    storage_file_path = None
    total_size_bytes = 0
    error_message = None

    async with _backup_restore_lock:
        async with AsyncSessionLocal() as db:
            try:
                # Cleanup any stale in-progress backups (older than 10 minutes)
                from sqlalchemy import select, and_
                ten_minutes_ago = datetime.now(timezone.utc) - timedelta(minutes=10)
                stale_stmt = select(Backup).where(
                    and_(
                        Backup.status == "in-progress",
                        Backup.created_at < ten_minutes_ago
                    )
                )
                result = await db.execute(stale_stmt)
                stale_backups = result.scalars().all()
                for stale in stale_backups:
                    logger.warning(f"Cleaning up stale backup: {stale.id}")
                    stale.status = "failed"
                    stale.error_message = "Backup timed out or was interrupted"
                    stale.completed_at = datetime.now(timezone.utc)
                await db.commit()

                # Get backup record
                backup_record = await crud.backup.get(db, id=backup_id)
                if not backup_record:
                    logger.error(f"Backup record not found: {backup_id}")
                    return

                # Create backup directory OUTSIDE storage to avoid circular dependency
                # Storage is at /storage, backups at /backups (same level)
                storage_root = Path(settings.STORAGE_ROOT)
                backup_dir = storage_root.parent / "backups"
                backup_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Using backup directory: {backup_dir}")

                # Create database backup
                db_file_path, db_size = await create_db_backup(backup_id, backup_dir)

                # Create storage backup
                storage_file_path, storage_size = await create_storage_backup(backup_id, backup_dir)

                # Mark as successful
                total_size_bytes = db_size + storage_size
                backup_success = True
                logger.info(f"Backup completed successfully: {backup_id}")

            except Exception as e:
                logger.error(f"Backup task failed: {e}", exc_info=True)
                backup_success = False
                error_message = str(e)

    # Update status in a separate session (outside the main session)
    # This ensures status is updated even if the main session fails during commit
    try:
        async with AsyncSessionLocal() as status_db:
            backup_record = await crud.backup.get(status_db, id=backup_id)
            if backup_record:
                if backup_success:
                    # Backup succeeded
                    update_data = schemas.BackupUpdate(
                        status="success",
                        db_file_path=db_file_path,
                        storage_file_path=storage_file_path,
                        total_size_bytes=total_size_bytes,
                        completed_at=datetime.now(timezone.utc)
                    )
                    logger.info(f"✅ Marking backup as success: {backup_id}")
                else:
                    # Backup failed
                    update_data = schemas.BackupUpdate(
                        status="failed",
                        error_message=error_message or "Unknown error occurred",
                        completed_at=datetime.now(timezone.utc)
                    )
                    logger.error(f"❌ Marking backup as failed: {backup_id}")

                await crud.backup.update(status_db, db_obj=backup_record, obj_in=update_data)
                await status_db.commit()
                logger.info(f"Status update committed for backup: {backup_id}")
    except Exception as status_error:
        logger.error(f"Failed to update backup status (will retry on next list): {status_error}", exc_info=True)


async def terminate_db_connections(db_config: dict):
    """Terminate all active database connections except the current one."""
    try:
        env = os.environ.copy()
        env["PGPASSWORD"] = db_config["password"]

        # Connect to postgres database to terminate connections to target database
        terminate_sql = f"""
        SELECT pg_terminate_backend(pid)
        FROM pg_stat_activity
        WHERE datname = '{db_config["database"]}'
        AND pid <> pg_backend_pid();
        """

        cmd = [
            "psql",
            "-h", db_config["host"],
            "-p", db_config["port"],
            "-U", db_config["user"],
            "-d", "postgres",  # Connect to postgres database
            "-c", terminate_sql
        ]

        logger.info("Terminating active database connections...")
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            logger.warning(f"Failed to terminate connections: {result.stderr}")
        else:
            logger.info("Active database connections terminated")

        # Wait for connections to close
        await asyncio.sleep(2)

    except Exception as e:
        logger.warning(f"Error terminating connections: {e}")


async def perform_restore(backup_id: str):
    """Background task to perform the actual restore with comprehensive error handling."""
    logger.info(f"Starting restore task for backup_id: {backup_id}")

    # Import maintenance mode functions
    from app.core.middleware import set_maintenance_mode

    db_rollback_file = None
    storage_backup_temp = None
    restore_failed = False
    existing_backup_dicts: list[dict] = []
    restored_backup_snapshot: dict | None = None

    async with _backup_restore_lock:
        # Enable maintenance mode FIRST
        set_maintenance_mode(True)
        logger.info("🔒 Maintenance mode ENABLED - All API requests will be blocked")

        # Wait a moment for any in-flight requests to complete
        await asyncio.sleep(2)

        async with AsyncSessionLocal() as db:
            try:
                # Get backup record
                backup_record = await crud.backup.get(db, id=backup_id)
                if not backup_record:
                    raise ValueError(f"Backup not found: {backup_id}")

                if backup_record.status != "success":
                    raise ValueError(f"Cannot restore from backup with status: {backup_record.status}")

                if not backup_record.db_file_path or not backup_record.storage_file_path:
                    raise ValueError("Backup files not found")

                # Snapshot current backups so we can merge them back after restore
                existing_backups = await crud.backup.get_all_active(db, skip=0, limit=1_000_000)
                logger.info(f"📸 Snapshotting {len(existing_backups)} backup records before restore")
                for b in existing_backups:
                    snapshot = {
                        "id": b.id,
                        "name": b.name,
                        "type": b.type,
                        "status": b.status,
                        "db_file_path": b.db_file_path,
                        "storage_file_path": b.storage_file_path,
                        "total_size_bytes": b.total_size_bytes,
                        "metadata_": b.metadata_,
                        "error_message": b.error_message,
                        "created_at": b.created_at,
                        "completed_at": b.completed_at,
                        "restored_at": b.restored_at,
                        "deleted_at": b.deleted_at,
                    }
                    existing_backup_dicts.append(snapshot)
                    if str(b.id) == str(backup_id):
                        restored_backup_snapshot = snapshot
                    logger.debug(f"  - Snapshot: {b.name} (ID: {b.id})")

                db_file = Path(backup_record.db_file_path)
                storage_file = Path(backup_record.storage_file_path)

                if not db_file.exists():
                    raise FileNotFoundError(f"Database backup file not found: {db_file}")
                if not storage_file.exists():
                    raise FileNotFoundError(f"Storage backup file not found: {storage_file}")

                # Mark restore as in-progress
                meta = backup_record.metadata_ or {}
                meta.update({
                    "restore_status": "in-progress",
                    "restore_started_at": datetime.now(timezone.utc).isoformat()
                })
                await crud.backup.update(
                    db,
                    db_obj=backup_record,
                    obj_in=schemas.BackupUpdate(metadata_=meta)
                )
                await db.commit()

                # === DATABASE RESTORE WITH ROLLBACK ===
                db_config = get_database_config()
                env = os.environ.copy()
                env["PGPASSWORD"] = db_config["password"]

                # Step 1: Create pre-restore backup for rollback
                logger.info("📦 Creating pre-restore database backup for rollback...")
                # Use backup directory OUTSIDE storage
                storage_root = Path(settings.STORAGE_ROOT)
                backup_dir = storage_root.parent / "backups"
                backup_dir.mkdir(parents=True, exist_ok=True)

                try:
                    db_rollback_file, _ = await create_db_backup(f"pre_restore_{backup_id}", backup_dir)
                    logger.info(f"✅ Pre-restore backup created: {db_rollback_file}")
                except Exception as e:
                    logger.error(f"Failed to create pre-restore backup: {e}")
                    raise RuntimeError("Cannot proceed without pre-restore backup for rollback")

                # Step 2: Close current database connection
                logger.info("🔌 Closing database connections...")
                await db.close()

                # Step 3: Terminate all active connections to the database
                await terminate_db_connections(db_config)

                # Step 4: Perform database restore
                logger.info("🔄 Restoring database...")
                cmd = [
                    "pg_restore",
                    "-h", db_config["host"],
                    "-p", db_config["port"],
                    "-U", db_config["user"],
                    "-d", db_config["database"],
                    "--clean",
                    "--if-exists",
                    "--exclude-table=backups",  # 백업 메타 테이블은 복구 대상에서 제외
                    str(db_file)
                ]

                result = subprocess.run(
                    cmd,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=settings.RESTORE_TIMEOUT
                )

                # Check for critical errors
                critical_errors = ["ERROR", "FATAL", "PANIC"]
                if result.returncode != 0:
                    has_critical_error = any(err in result.stderr.upper() for err in critical_errors)

                    if has_critical_error:
                        error_msg = f"pg_restore failed with critical error: {result.stderr}"
                        logger.error(error_msg)
                        restore_failed = True
                        raise RuntimeError(error_msg)
                    elif result.stderr.strip():
                        logger.warning(f"pg_restore completed with warnings: {result.stderr}")

                logger.info("✅ Database restored successfully")

                # === STORAGE RESTORE WITH ROLLBACK ===
                logger.info("📁 Restoring storage...")
                storage_root = Path(settings.STORAGE_ROOT)

                # Backup existing storage contents (not the directory itself)
                storage_backup_temp = storage_root.parent / f"storage_temp_{backup_id}"
                if storage_root.exists() and any(storage_root.iterdir()):
                    logger.info(f"Backing up existing storage contents to: {storage_backup_temp}")
                    try:
                        storage_backup_temp.mkdir(parents=True, exist_ok=True)
                        # Copy contents, not the directory itself (for Docker volume compatibility)
                        for item in storage_root.iterdir():
                            if item.is_dir():
                                shutil.copytree(item, storage_backup_temp / item.name)
                            else:
                                shutil.copy2(item, storage_backup_temp / item.name)
                        logger.info("✅ Storage contents backed up successfully")
                    except Exception as e:
                        logger.error(f"Failed to backup existing storage: {e}")
                        restore_failed = True
                        raise

                try:
                    # Clear existing storage contents
                    if storage_root.exists():
                        logger.info("Clearing existing storage contents...")
                        for item in storage_root.iterdir():
                            if item.is_dir():
                                shutil.rmtree(item)
                            else:
                                item.unlink()

                    # Extract storage backup
                    logger.info(f"Extracting storage backup...")
                    with tarfile.open(storage_file, "r:gz") as tar:
                        tar.extractall(path=storage_root.parent)

                    # Remove temporary backup on success
                    if storage_backup_temp and storage_backup_temp.exists():
                        logger.info("Removing temporary storage backup")
                        shutil.rmtree(str(storage_backup_temp))
                        storage_backup_temp = None

                    logger.info("✅ Storage restored successfully")

                except Exception as e:
                    logger.error(f"Storage restore failed: {e}")
                    restore_failed = True

                    # Rollback storage
                    if storage_backup_temp and storage_backup_temp.exists():
                        logger.warning("⚠️ Rolling back storage...")
                        # Clear failed restore contents
                        if storage_root.exists():
                            for item in storage_root.iterdir():
                                if item.is_dir():
                                    shutil.rmtree(item)
                                else:
                                    item.unlink()
                        # Restore from backup
                        for item in storage_backup_temp.iterdir():
                            if item.is_dir():
                                shutil.copytree(item, storage_root / item.name)
                            else:
                                shutil.copy2(item, storage_root / item.name)
                        logger.info("✅ Storage rollback completed")
                    raise

                # Cleanup rollback file on success
                if db_rollback_file and Path(db_rollback_file).exists():
                    logger.info("Cleaning up pre-restore backup")
                    Path(db_rollback_file).unlink()

                logger.info(f"🎉 Restore completed successfully: {backup_id}")

                # Mark restore success - need new DB session
                async with AsyncSessionLocal() as new_db:
                    backup_record = await crud.backup.get(new_db, id=backup_id)

                    if backup_record:
                        meta = (backup_record.metadata_ or {}).copy()
                        meta.update({
                            "restore_status": "success",
                            "restore_completed_at": datetime.now(timezone.utc).isoformat()
                        })
                        update_data = schemas.BackupUpdate(
                            status="success",
                            db_file_path=backup_record.db_file_path,
                            storage_file_path=backup_record.storage_file_path,
                            total_size_bytes=backup_record.total_size_bytes,
                            completed_at=backup_record.completed_at or datetime.now(timezone.utc),
                            restored_at=datetime.now(timezone.utc),  # Record when this backup was restored
                            metadata_=meta,
                        )
                        await crud.backup.update(
                            new_db,
                            db_obj=backup_record,
                            obj_in=update_data
                        )
                        await new_db.commit()
            except Exception as e:
                logger.error(f"❌ Restore task failed: {e}", exc_info=True)
                restore_failed = True

                # Rollback database if we have a rollback file
                if db_rollback_file and Path(db_rollback_file).exists():
                    logger.warning("⚠️ Rolling back database...")
                    try:
                        await terminate_db_connections(db_config)

                        rollback_cmd = [
                            "pg_restore",
                            "-h", db_config["host"],
                            "-p", db_config["port"],
                            "-U", db_config["user"],
                            "-d", db_config["database"],
                            "--clean",
                            "--if-exists",
                            db_rollback_file
                        ]

                        rollback_result = subprocess.run(
                            rollback_cmd,
                            env=env,
                            capture_output=True,
                            text=True,
                            timeout=settings.RESTORE_TIMEOUT
                        )

                        if rollback_result.returncode == 0:
                            logger.info("✅ Database rollback completed")
                        else:
                            logger.error(f"Database rollback failed: {rollback_result.stderr}")
                    except Exception as rollback_error:
                        logger.error(f"Failed to rollback database: {rollback_error}")

                # Record failure status
                try:
                    async with AsyncSessionLocal() as new_db:
                        backup_record = await crud.backup.get(new_db, id=backup_id)
                        if backup_record:
                            meta = backup_record.metadata_ or {}
                            meta.update({
                                "restore_status": "failed",
                                "restore_error": str(e),
                                "restore_failed_at": datetime.now(timezone.utc).isoformat()
                            })
                            await crud.backup.update(
                                new_db,
                                db_obj=backup_record,
                                obj_in=schemas.BackupUpdate(metadata_=meta)
                            )
                            await new_db.commit()
                except Exception:
                    logger.error("Failed to record restore failure status")

                raise

            finally:
                # Reset connection pool to clear enum type cache
                # This is crucial after database restore to avoid "cache lookup failed for type" errors
                try:
                    logger.info("♻️ Resetting database connection pool...")
                    await engine.dispose()
                    # Wait briefly for existing connections to complete
                    await asyncio.sleep(1)
                    logger.info("✅ Connection pool reset successfully")
                except Exception as e:
                    logger.error(f"Failed to reset connection pool: {e}")

                # Run VACUUM ANALYZE to update database statistics after restore
                # This prevents slow queries due to outdated query planner statistics
                if not restore_failed:
                    try:
                        logger.info("📊 Running VACUUM ANALYZE to update database statistics...")
                        db_config = get_database_config()
                        env = os.environ.copy()
                        env["PGPASSWORD"] = db_config["password"]

                        vacuum_cmd = [
                            "psql",
                            "-h", db_config["host"],
                            "-p", db_config["port"],
                            "-U", db_config["user"],
                            "-d", db_config["database"],
                            "-c", "VACUUM (VERBOSE, ANALYZE);"
                        ]

                        vacuum_result = subprocess.run(
                            vacuum_cmd,
                            env=env,
                            capture_output=True,
                            text=True,
                            timeout=300  # 5 minutes timeout
                        )

                        if vacuum_result.returncode == 0:
                            logger.info("✅ VACUUM ANALYZE completed successfully")
                        else:
                            logger.warning(f"VACUUM ANALYZE completed with warnings: {vacuum_result.stderr}")
                    except Exception as vacuum_error:
                        logger.warning(f"Failed to run VACUUM ANALYZE (non-critical): {vacuum_error}")

                # ALWAYS disable maintenance mode
                set_maintenance_mode(False)
                logger.info("🔓 Maintenance mode DISABLED - API requests restored")

                # Cleanup temporary files
                if storage_backup_temp and storage_backup_temp.exists():
                    try:
                        shutil.rmtree(str(storage_backup_temp))
                    except Exception as e:
                        logger.warning(f"Failed to cleanup temp storage: {e}")


@router.post("/backups", response_model=schemas.BackupResponse)
async def create_backup(
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    backup_in: schemas.BackupCreate = schemas.BackupCreate()
):
    """
    Create a new backup of the database and storage.
    The backup process runs in the background.
    """
    try:
        # Generate backup name if not provided
        if not backup_in.name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_in.name = f"backup_{timestamp}"

        # Create backup record
        backup_record = await crud.backup.create(db, obj_in=backup_in)
        await db.commit()
        await db.refresh(backup_record)

        # Start backup in background
        background_tasks.add_task(perform_backup, backup_record.id)

        logger.info(f"Backup creation initiated: {backup_record.id}")

        return schemas.BackupResponse.from_backup(backup_record)

    except Exception as e:
        logger.error(f"Failed to create backup: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create backup: {str(e)}"
        )


@router.get("/backups", response_model=schemas.BackupListResponse)
async def list_backups(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db)
):
    """Get list of all backups."""
    try:
        # Cleanup any stale in-progress backups (older than 10 minutes)
        from sqlalchemy import select, and_
        ten_minutes_ago = datetime.now(timezone.utc) - timedelta(minutes=10)
        stale_stmt = select(Backup).where(
            and_(
                Backup.status == "in-progress",
                Backup.created_at < ten_minutes_ago
            )
        )
        result = await db.execute(stale_stmt)
        stale_backups = result.scalars().all()
        if stale_backups:
            for stale in stale_backups:
                logger.warning(f"Cleaning up stale backup in list_backups: {stale.id}")
                stale.status = "failed"
                stale.error_message = "Backup timed out or was interrupted"
                stale.completed_at = datetime.now(timezone.utc)
            await db.commit()

        backups = await crud.backup.get_all_active(db, skip=skip, limit=limit)
        total_size = await crud.backup.get_total_size(db)

        # Convert to response format
        backup_responses = [schemas.BackupResponse.from_backup(b) for b in backups]

        # Calculate human-readable total size
        if total_size < 1024:
            total_size_str = f"{total_size} B"
        elif total_size < 1024 * 1024:
            total_size_str = f"{total_size / 1024:.1f} KB"
        elif total_size < 1024 * 1024 * 1024:
            total_size_str = f"{total_size / (1024 * 1024):.1f} MB"
        else:
            total_size_str = f"{total_size / (1024 * 1024 * 1024):.2f} GB"

        return schemas.BackupListResponse(
            backups=backup_responses,
            total=len(backups),
            total_size_bytes=total_size,
            total_size=total_size_str
        )

    except Exception as e:
        logger.error(f"Failed to list backups: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list backups: {str(e)}"
        )


@router.post("/backups/{backup_id}/restore")
async def restore_backup(
    backup_id: str,
    background_tasks: BackgroundTasks
):
    """
    Restore database and storage from a backup.
    WARNING: This will overwrite all current data!
    """
    # Use manual session management to avoid connection issues with background tasks
    async with AsyncSessionLocal() as db:
        try:
            # Check if backup exists
            backup_record = await crud.backup.get(db, id=backup_id)
            if not backup_record:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Backup not found: {backup_id}"
                )

            if backup_record.status != "success":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Cannot restore from backup with status: {backup_record.status}"
                )

            # Commit and close session before starting background task
            await db.commit()

        except HTTPException:
            await db.rollback()
            raise
        except Exception as e:
            await db.rollback()
            logger.error(f"Failed to validate backup: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to validate backup: {str(e)}"
            )

    # Start restore in background (after session is closed)
    background_tasks.add_task(perform_restore, backup_id)

    logger.info(f"Restore initiated for backup: {backup_id}")

    return {
        "status": "started",
        "message": f"Restore process started for backup {backup_id}",
        "backup_id": backup_id
    }


@router.delete("/backups/{backup_id}")
async def delete_backup(
    backup_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Delete a backup and its associated files."""
    try:
        # Get backup record
        backup_record = await crud.backup.get(db, id=backup_id)
        if not backup_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Backup not found: {backup_id}"
            )

        # Delete backup files
        if backup_record.db_file_path:
            db_file = Path(backup_record.db_file_path)
            if db_file.exists():
                db_file.unlink()
                logger.info(f"Deleted database backup file: {db_file}")

        if backup_record.storage_file_path:
            storage_file = Path(backup_record.storage_file_path)
            if storage_file.exists():
                storage_file.unlink()
                logger.info(f"Deleted storage backup file: {storage_file}")

        # Soft delete the backup record
        await crud.backup.soft_delete(db, id=backup_id)
        await db.commit()

        logger.info(f"Backup deleted: {backup_id}")

        return {"status": "success", "message": f"Backup {backup_id} deleted"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete backup: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete backup: {str(e)}"
        )


@router.get("/backups/{backup_id}/download")
async def download_backup(
    backup_id: str,
    file_type: str = "db",  # 'db' or 'storage'
    db: AsyncSession = Depends(get_db)
):
    """Download a backup file."""
    try:
        # Get backup record
        backup_record = await crud.backup.get(db, id=backup_id)
        if not backup_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Backup not found: {backup_id}"
            )

        # Get file path based on type
        if file_type == "db":
            file_path = backup_record.db_file_path
        elif file_type == "storage":
            file_path = backup_record.storage_file_path
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file_type. Must be 'db' or 'storage'"
            )

        if not file_path:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Backup file not found for type: {file_type}"
            )

        file = Path(file_path)
        if not file.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Backup file does not exist: {file}"
            )

        return FileResponse(
            path=str(file),
            filename=file.name,
            media_type="application/octet-stream"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download backup: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to download backup: {str(e)}"
        )


@router.get("/backups/{backup_id}/export")
async def export_backup(
    backup_id: str,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """
    Export a backup as a ZIP file containing both database and storage files.
    The ZIP file can be imported later using the import endpoint.
    """
    try:
        # Get backup record
        backup_record = await crud.backup.get(db, id=backup_id)
        if not backup_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Backup not found: {backup_id}"
            )

        if backup_record.status != "success":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot export backup with status: {backup_record.status}"
            )

        if not backup_record.db_file_path or not backup_record.storage_file_path:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Backup files not found"
            )

        db_file = Path(backup_record.db_file_path)
        storage_file = Path(backup_record.storage_file_path)

        if not db_file.exists():
            raise FileNotFoundError(f"Database backup file not found: {db_file}")
        if not storage_file.exists():
            raise FileNotFoundError(f"Storage backup file not found: {storage_file}")

        # Create ZIP file in backup directory (not temp directory)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"backup_export_{backup_record.name}_{timestamp}.zip"

        # Use backup directory to avoid temp directory cleanup issues
        storage_root = Path(settings.STORAGE_ROOT)
        backup_dir = storage_root.parent / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        zip_path = backup_dir / zip_filename

        # Create the ZIP file
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add database file
            zipf.write(db_file, arcname=f"database_{db_file.name}")
            logger.info(f"Added database file to export: {db_file.name}")

            # Add storage file
            zipf.write(storage_file, arcname=f"storage_{storage_file.name}")
            logger.info(f"Added storage file to export: {storage_file.name}")

            # Add metadata file with backup info
            metadata = {
                "backup_id": str(backup_record.id),
                "backup_name": backup_record.name,
                "created_at": backup_record.created_at.isoformat() if backup_record.created_at else None,
                "export_date": datetime.now(timezone.utc).isoformat(),
                "total_size_bytes": backup_record.total_size_bytes,
                "db_filename": f"database_{db_file.name}",
                "storage_filename": f"storage_{storage_file.name}"
            }
            import json
            metadata_json = json.dumps(metadata, indent=2)
            zipf.writestr("metadata.json", metadata_json)
            logger.info("Added metadata to export")

        logger.info(f"Export created: {zip_path} ({zip_path.stat().st_size} bytes)")

        # Schedule cleanup of the export ZIP file after response is sent
        def cleanup_export_file():
            try:
                if zip_path.exists():
                    zip_path.unlink()
                    logger.info(f"Cleaned up export file: {zip_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup export file {zip_path}: {e}")

        background_tasks.add_task(cleanup_export_file)

        # Return the ZIP file
        return FileResponse(
            path=str(zip_path),
            filename=zip_filename,
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename={zip_filename}"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export backup: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export backup: {str(e)}"
        )


@router.post("/backups/import")
async def import_backup(
    zip_file: UploadFile = File(..., description="Exported backup ZIP file"),
    backup_name: str = Form(None, description="Optional backup name"),
    db: AsyncSession = Depends(get_db)
):
    """
    Import a backup from an exported ZIP file.
    The ZIP file should contain database and storage files from export.
    Creates a new backup record with 'export_' prefix in the name.
    """
    try:
        # Validate file type
        if not zip_file.filename or not zip_file.filename.endswith('.zip'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be a ZIP archive"
            )

        # Create temporary directory for extraction
        with tempfile.TemporaryDirectory() as temp_extract_dir:
            extract_path = Path(temp_extract_dir)

            # Save uploaded ZIP temporarily
            temp_zip_path = extract_path / "upload.zip"
            logger.info(f"Saving uploaded ZIP to: {temp_zip_path}")
            with open(temp_zip_path, "wb") as f:
                content = await zip_file.read()
                f.write(content)

            # Extract ZIP file
            logger.info("Extracting ZIP archive...")
            with zipfile.ZipFile(temp_zip_path, 'r') as zipf:
                zipf.extractall(extract_path)

            # Find database and storage files
            db_file_path = None
            storage_file_path = None
            metadata_path = extract_path / "metadata.json"

            for file in extract_path.iterdir():
                if file.name.startswith("database_") and file.suffix == ".sql":
                    db_file_path = file
                elif file.name.startswith("storage_") and file.suffix == ".gz":
                    storage_file_path = file

            if not db_file_path or not storage_file_path:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid backup ZIP: missing database or storage files"
                )

            logger.info(f"Found database file: {db_file_path.name}")
            logger.info(f"Found storage file: {storage_file_path.name}")

            # Read metadata if exists
            metadata = {}
            if metadata_path.exists():
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"Loaded metadata: {metadata.get('backup_name', 'unknown')}")

            # Create backup directory
            storage_root = Path(settings.STORAGE_ROOT)
            backup_dir = storage_root.parent / "backups"
            backup_dir.mkdir(parents=True, exist_ok=True)

            # Generate backup name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if backup_name:
                final_backup_name = f"export_{backup_name}"
            elif metadata.get('backup_name'):
                final_backup_name = f"export_{metadata['backup_name']}"
            else:
                final_backup_name = f"export_imported_{timestamp}"

            # Generate unique backup ID
            import uuid
            backup_id = str(uuid.uuid4())

            # Copy database file to backup directory
            db_backup_filename = f"db_{backup_id}_{timestamp}.sql"
            db_backup_path = backup_dir / db_backup_filename

            logger.info(f"Copying database file to: {db_backup_path}")
            shutil.copy2(db_file_path, db_backup_path)
            db_file_size = db_backup_path.stat().st_size

            # Copy storage file to backup directory
            storage_backup_filename = f"storage_{backup_id}_{timestamp}.tar.gz"
            storage_backup_path = backup_dir / storage_backup_filename

            logger.info(f"Copying storage file to: {storage_backup_path}")
            shutil.copy2(storage_file_path, storage_backup_path)
            storage_file_size = storage_backup_path.stat().st_size

            # Create backup record in database
            total_size = db_file_size + storage_file_size
            backup_metadata = {
                "source": "import",
                "imported_at": datetime.now(timezone.utc).isoformat(),
                "original_backup_name": metadata.get('backup_name'),
                "original_backup_id": metadata.get('backup_id'),
                "original_created_at": metadata.get('created_at')
            }

            backup_data = schemas.BackupCreate(
                name=final_backup_name,
                type="manual",
                metadata_=backup_metadata
            )

            backup_record = await crud.backup.create(db, obj_in=backup_data)

            # Update with file paths and status
            update_data = schemas.BackupUpdate(
                status="success",
                db_file_path=str(db_backup_path),
                storage_file_path=str(storage_backup_path),
                total_size_bytes=total_size,
                completed_at=datetime.now(timezone.utc)
            )
            await crud.backup.update(db, db_obj=backup_record, obj_in=update_data)
            await db.commit()
            await db.refresh(backup_record)

            logger.info(f"Backup imported successfully: {backup_record.id} - {final_backup_name}")

            return {
                "status": "success",
                "message": f"Backup imported successfully as '{final_backup_name}'",
                "backup_id": str(backup_record.id),
                "backup_name": final_backup_name,
                "db_size_bytes": db_file_size,
                "storage_size_bytes": storage_file_size,
                "total_size_bytes": total_size,
                "original_metadata": metadata
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to import backup: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to import backup: {str(e)}"
        )
