"""
CARLA backend proxy endpoints.

프런트엔드 -> 메인 백엔드 -> CARLA 백엔드 흐름을 위해
CARLA API를 프록시하고, 필요 메타데이터를 DB(SystemLog)에 남긴다.
"""
import json
import os
import hashlib
import mimetypes
from pathlib import Path
from typing import Dict, Optional, List
from uuid import UUID
from datetime import datetime

import httpx
from fastapi import APIRouter, HTTPException, Query, Depends, Body, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import insert, select
from pydantic import BaseModel, Field

from app.database import get_db
from app.models.system_log import SystemLog, LogLevel
from app.models.model_repo import ODModel, ODModelArtifact, ArtifactType
from app.models.dataset_3d import Patch3D, AttackDataset3D, Dataset3D, Image3D
from app.models.dataset_2d import AttackType
from app.models.annotation import Annotation
from app.core.logging import get_logger
from app.core.config import settings
from app import crud
from app.services.carla import (
    Dataset3DService,
    Patch3DService,
    AttackDataset3DService,
    AnnotationParser,
)

router = APIRouter(prefix="/carla", tags=["CARLA Proxy"])

logger = get_logger(__name__)

# Upstream CARLA backend (build from host/port if URL not provided)
CARLA_HOST = os.getenv("CARLA_BACKEND_HOST", "localhost")
CARLA_PORT = os.getenv("CARLA_BACKEND_PORT", "8888")
CARLA_BASE_URL = os.getenv("CARLA_BACKEND_URL", f"http://{CARLA_HOST}:{CARLA_PORT}")


# Request schemas
class PatchGenerationRequest(BaseModel):
    """Request schema for patch generation."""
    patch_name: str
    object_name: str
    attack_method_name: str
    model_id: Optional[UUID] = None  # 메인 백엔드 DB의 모델 ID
    object_detection_name: Optional[str] = None  # 또는 직접 이름 지정 (내장 모델용)
    # 아래는 선택적 (메인 백엔드에서 자동으로 채워짐)
    model_path: str = ""
    model_type: str = ""
    class_name: List[str] = Field(default_factory=lambda: ["car"])
    input_size: List[int] = Field(default_factory=lambda: [640, 640])
    device_type: str = ""


# Helpers (now delegated to services)


async def _save_dataset_3d(
  dataset_name: str,
  storage_path: str,
  metadata: Dict,
  db: AsyncSession,
  owner_id: Optional[UUID] = None
) -> Optional["Dataset3D"]:
  """Wrapper for Dataset3DService.create_dataset"""
  return await Dataset3DService.create_dataset(
    dataset_name=dataset_name,
    storage_path=storage_path,
    metadata=metadata,
    db=db,
    owner_id=owner_id
  )


def _convert_carla_path_to_storage(image_path: str) -> Optional[str]:
  """Wrapper for Dataset3DService.convert_carla_path_to_storage"""
  return Dataset3DService.convert_carla_path_to_storage(image_path)


def _infer_image_count(metadata: Optional[Dict]) -> int:
  """Wrapper for Dataset3DService.infer_image_count"""
  return Dataset3DService.infer_image_count(metadata)


async def _save_patch_metadata(
  image_path: str,
  patch_name: str,
  method: str,
  target_class: str,
  model_id: Optional[UUID],
  db: AsyncSession,
  source_dataset_id: Optional[UUID] = None,
  training_metadata: Optional[Dict] = None
) -> Optional[Patch3D]:
  """Wrapper for Patch3DService.save_patch_metadata"""
  return await Patch3DService.save_patch_metadata(
    image_path=image_path,
    patch_name=patch_name,
    method=method,
    target_class=target_class,
    model_id=model_id,
    db=db,
    source_dataset_id=source_dataset_id,
    training_metadata=training_metadata
  )


async def _log_system(db: AsyncSession, action: str, message: str, details: Optional[Dict] = None, level: LogLevel = LogLevel.INFO):
  """Persist a lightweight system log entry."""
  try:
    stmt = insert(SystemLog).values(
      log_level=level,
      action=action,
      module="carla_proxy",
      message=message,
      details=details or {},
    )
    await db.execute(stmt)
    await db.commit()
  except Exception as e:
    logger.warning(f"Failed to write system log: {e}")


async def _proxy_get(path: str, params: Optional[Dict] = None) -> JSONResponse:
  url = f"{CARLA_BASE_URL}{path}"
  logger.info(f"🔄 CARLA GET: {url}")
  if params:
    logger.debug(f"   Params: {params}")

  try:
    async with httpx.AsyncClient(timeout=30.0) as client:
      resp = await client.get(url, params=params)

    logger.info(f"✅ CARLA Response: {resp.status_code} for {path}")

    if resp.status_code >= 400:
      try:
        detail = resp.json()
      except Exception:
        detail = {"message": resp.text}
      logger.error(f"❌ CARLA Error: {resp.status_code} - {detail}")
      raise HTTPException(status_code=resp.status_code, detail=detail)
    return JSONResponse(content=resp.json(), status_code=resp.status_code)

  except httpx.ReadTimeout:
    logger.error(f"❌ CARLA Backend timeout: {url}")
    return JSONResponse(
      status_code=503,
      content={
        "state": 503,
        "message": "CARLA 백엔드가 응답하지 않습니다. CARLA 시뮬레이터가 실행 중인지 확인하세요.",
        "result": []
      }
    )
  except httpx.ConnectError as e:
    logger.error(f"❌ CARLA Backend connection failed: {e}")
    return JSONResponse(
      status_code=503,
      content={
        "state": 503,
        "message": f"CARLA 백엔드({CARLA_BASE_URL})에 연결할 수 없습니다.",
        "result": []
      }
    )
  except Exception as e:
    logger.error(f"❌ Unexpected error: {e}")
    return JSONResponse(
      status_code=500,
      content={
        "state": 500,
        "message": f"예상치 못한 오류: {str(e)}",
        "result": []
      }
    )


async def _proxy_stream_post(
  path: str,
  payload: Dict,
  db: AsyncSession,
  log_action: str,
  save_patch: bool = False,
  save_dataset: bool = False,
  save_attack_dataset: bool = False,
  model_id: Optional[UUID] = None,
) -> StreamingResponse:
  """
  Stream CARLA POST responses and tee minimal logs.

  Args:
    path: CARLA backend endpoint path
    payload: Request payload
    db: Database session
    log_action: Action name for system log
    save_patch: 완료 시 Patch3D 저장 여부
    save_dataset: 완료 시 Dataset3D 저장 여부
    save_attack_dataset: 완료 시 AttackDataset3D 저장 여부
    model_id: 모델 ID (패치 저장 시 사용)
  """
  url = f"{CARLA_BASE_URL}{path}"

  # 로깅: 시작
  logger.info(f"🔄 CARLA POST Stream: {url}")
  logger.debug(f"   Payload: {payload}")

  # 상태 플래그
  patch_saved = False
  dataset_saved = False
  attack_dataset_saved = False

  async def event_generator():
    nonlocal patch_saved, dataset_saved, attack_dataset_saved

    async with httpx.AsyncClient(timeout=None) as client:
      try:
        upstream_ctx = client.stream("POST", url, json=payload)
        async with upstream_ctx as upstream:
          if upstream.status_code >= 400:
            try:
              error_body = await upstream.aread()
              error_json = json.loads(error_body.decode() or "{}")
            except Exception:
              error_json = {"message": "Upstream error"}
            logger.error(f"❌ CARLA Stream Error: {upstream.status_code} - {error_json}")
            raise HTTPException(status_code=upstream.status_code, detail=error_json)

          logger.info(f"📡 CARLA Stream started (status: {upstream.status_code})")
          logger.info(f"Response headers: {dict(upstream.headers)}")

          # aiter_raw()로 바이트 스트림을 직접 받아 디코딩 (버퍼링 없음)
          buffer = b""
          async for raw_chunk in upstream.aiter_raw():
            if not raw_chunk:
              continue

            logger.debug(f"🔵 Received chunk: {len(raw_chunk)} bytes")
            logger.debug(f"   Chunk content: {raw_chunk[:200]}")
            buffer += raw_chunk
            logger.debug(f"   Buffer size: {len(buffer)} bytes, has newline: {b'\\n' in buffer}")

            # 줄 단위로 분리 처리
            while b"\n" in buffer:
              line_bytes, buffer = buffer.split(b"\n", 1)

              if not line_bytes.strip():
                continue

              # UTF-8 디코딩
              try:
                line = line_bytes.decode('utf-8')
              except UnicodeDecodeError as e:
                logger.error(f"Failed to decode line: {e}")
                continue

              try:
                # 원본 라인 로깅 (디버깅용)
                logger.debug(f"📨 Raw stream line: {line[:200]}...")

                data = json.loads(line)

                # 스트림 데이터 로깅 (상세 정보는 DEBUG 레벨)
                if data.get("state") == 200 and "result" in data:
                  result = data["result"]
                  logger.debug(f"📦 Stream data: {result}")
                elif data.get("state") != 200:
                  logger.warning(f"⚠️ Stream warning: {data}")

                await _log_system(
                  db,
                  action=log_action,
                  message="stream",
                  details={"payload": payload, "result": data},
                  level=LogLevel.INFO,
                )

                # 모든 이미지 경로를 CARLA 경로에서 storage 경로로 변환
                if data.get("state") == 200 and "result" in data:
                  result = data["result"]
                  image_path = result.get("image_path")

                  if image_path:
                    logger.info(f"🔄 Processing image_path from CARLA: {image_path}")
                    storage_path = _convert_carla_path_to_storage(image_path)

                    if storage_path:
                      data["result"]["image_path"] = storage_path
                      # storage_key 추가 (프론트엔드에서 /api/storage/{storage_key} 형식으로 사용)
                      # /storage/3d/k2_tank/image.jpg -> 3d/k2_tank/image.jpg
                      storage_key = storage_path.replace("/storage/", "")
                      data["result"]["storage_key"] = storage_key
                      logger.info(f"✅ Image path converted: {image_path} -> {storage_path}, storage_key: {storage_key}")
                    else:
                      logger.error(f"❌ Failed to convert image_path: {image_path}")
                  else:
                    logger.debug(f"No image_path in result: {result}")

              except Exception as e:
                logger.warning(f"Failed to process stream line: {e}")
                yield line + "\n"
                continue

              # 완료 시점 처리 - Dataset3D 저장
              if save_dataset and not dataset_saved and data.get("state") == 200 and "result" in data:
                result = data["result"]
                loc_idx = result.get("loc_idx")
                loc_total = result.get("loc_total")
                is_complete = loc_idx is not None and loc_total is not None and loc_idx == loc_total
                if is_complete:
                  # storage_path는 메인 백엔드의 경로 사용 (이미지 경로가 업데이트된 경우 그것을 사용)
                  # 예: /storage/3d/k2_tank 또는 /storage/3d/exports/k2_tank_tmpkkkkksad
                  image_path = result.get("image_path", "")
                  if image_path and image_path.startswith("/storage/3d/"):
                    # 이미지 경로에서 폴더 경로 추출 (마지막 파일명 제거)
                    # /storage/3d/k2_tank/etron_car_eval/image.jpg -> /storage/3d/k2_tank
                    # /storage/3d/exports/k2_tank_tmpkkkkksad/etron_car_eval/image.jpg -> /storage/3d/exports/k2_tank_tmpkkkkksad
                    path_parts = image_path.split("/")
                    if len(path_parts) >= 4:
                      # exports 폴더인 경우 한 단계 더 깊이 포함
                      if len(path_parts) >= 5 and path_parts[3] == "exports":
                        storage_path = "/".join(path_parts[:5])  # /storage/3d/exports/folder_name
                      else:
                        storage_path = "/".join(path_parts[:4])  # /storage/3d/folder_name
                    else:
                      storage_path = f"/storage/3d/{payload.get('dataset_name', 'unknown')}"
                  else:
                    storage_path = f"/storage/3d/{payload.get('dataset_name', 'unknown')}"

                  metadata = {
                    "map_name": payload.get("map_name"),
                    "weather_name": payload.get("weather_name"),
                    "time_name": payload.get("time_name"),
                    "object_name": payload.get("object_name"),
                    "image_width": payload.get("image_width"),
                    "image_height": payload.get("image_height"),
                    "loc_total": loc_total,
                    "batch_total": result.get("batch_total"),
                    "source": "carla_simulator",
                  }
                  dataset = await _save_dataset_3d(
                    dataset_name=payload.get("dataset_name", "unknown_dataset"),
                    storage_path=storage_path,
                    metadata=metadata,
                    db=db,
                  )
                  if dataset:
                    dataset_saved = True

                    # Image3D 레코드 저장
                    image_count, annotation_count = await Dataset3DService.save_images_and_annotations(
                      dataset=dataset,
                      storage_path=storage_path,
                      db=db,
                      parse_annotations=False  # 먼저 어노테이션 파싱 없이 이미지만 저장
                    )
                    logger.info(f"Saved {image_count} images for dataset {dataset.id}")

                    # CARLA가 annotations JSON을 생성할 때까지 대기 (폴링)
                    import asyncio
                    storage_key = storage_path.replace("/storage/", "")
                    base_dir = Path(settings.STORAGE_ROOT) / storage_key
                    annotations_dir = base_dir / "annotations"
                    json_files = []

                    max_wait_seconds = 30
                    poll_interval = 0.5
                    waited_seconds = 0

                    logger.info(f"Waiting for annotations JSON in {annotations_dir}")
                    while waited_seconds < max_wait_seconds:
                      if annotations_dir.exists():
                        json_files = list(annotations_dir.glob("*.json"))
                        if json_files:
                          logger.info(f"Found annotations JSON: {json_files[0]}")
                          break
                      await asyncio.sleep(poll_interval)
                      waited_seconds += poll_interval

                    # 어노테이션 파싱 시도
                    if json_files:
                      # file_name -> image_3d_id 매핑 생성
                      from sqlalchemy import select
                      stmt = select(Image3D).where(Image3D.dataset_id == dataset.id)
                      result_imgs = await db.execute(stmt)
                      images = result_imgs.scalars().all()
                      image_id_map = {img.file_name: img.id for img in images}

                      annotation_count = await AnnotationParser.parse_and_save(
                        annotations_dir=annotations_dir,
                        dataset_id=dataset.id,
                        image_id_map=image_id_map,
                        db=db
                      )
                      logger.info(f"Saved {annotation_count} annotations for dataset {dataset.id}")
                    else:
                      logger.warning(
                        f"Annotations not found after {max_wait_seconds}s wait. "
                        f"Use POST /api/v1/carla/datasets_3d/{dataset.id}/parse-annotations later."
                      )

                    completion_data = {
                      "state": 200,
                      "result": {
                        **result,
                        "dataset_id": str(dataset.id),
                        "storage_path": dataset.storage_path,
                        "image_count": image_count,
                        "annotation_count": annotation_count
                      },
                      "message": f"Dataset saved with {image_count} images and {annotation_count} annotations",
                    }
                    yield json.dumps(completion_data, ensure_ascii=False) + "\n"
                    continue

              # 완료 시점 처리 - Patch3D 저장
              if save_patch and not patch_saved and data.get("state") == 200 and "result" in data:
                result = data["result"]
                epoch = result.get("epoch")
                epoch_total = result.get("epoch_total")
                image_path = result.get("image_path")
                is_complete = (
                  (epoch is not None and epoch_total is not None and epoch == epoch_total - 1)
                  or (image_path and "best" in image_path)
                )
                if is_complete and image_path:
                  # image_path는 이미 위에서 변환되어 /storage/3d/... 형식일 것
                  training_metadata = {
                    "epoch": epoch,
                    "epoch_total": epoch_total,
                    "loss": result.get("loss"),
                    "detection_score": result.get("detection_score"),
                  }

                  patch = await _save_patch_metadata(
                    image_path=image_path,
                    patch_name=payload.get("patch_name", "unknown_patch"),
                    method=payload.get("attack_method_name", payload.get("attack_method", "UNKNOWN")),
                    target_class=payload.get("object_name", "unknown"),
                    model_id=model_id,
                    db=db,
                    source_dataset_id=None,
                    training_metadata=training_metadata,
                  )

                  if patch:
                    patch_saved = True
                    completion_data = {
                      "state": 200,
                      "result": {**result, "patch_id": str(patch.id), "storage_key": patch.storage_key},
                      "message": "Patch saved to database",
                    }
                    yield json.dumps(completion_data, ensure_ascii=False) + "\n"
                    continue

              # 완료 시점 처리 - AttackDataset3D 저장
              if save_attack_dataset and not attack_dataset_saved and data.get("state") == 200 and "result" in data:
                result = data["result"]
                loc_idx = result.get("loc_idx")
                loc_total = result.get("loc_total")
                batch_idx = result.get("batch_idx")
                batch_total = result.get("batch_total")
                # 기본 완주 조건 (이전 로직 유지)
                is_complete = (
                  loc_idx is not None
                  and loc_total is not None
                  and batch_idx is not None
                  and batch_total is not None
                  and loc_idx == loc_total
                  and batch_idx == batch_total
                )
                # loc_total/batch_total 이 제공되지 않는 경우를 대비한 완화 조건
                has_progress_keys = any(
                  key in result for key in ("loc_idx", "loc_total", "batch_idx", "batch_total", "progress", "processed")
                )
                completion_flags = [
                  result.get("is_complete"),
                  result.get("completed"),
                  result.get("is_final"),
                  result.get("final"),
                  result.get("done"),
                ]
                if not is_complete:
                  # 명시적 완료 플래그가 있으면 완료로 간주
                  if any(flag is True for flag in completion_flags):
                    is_complete = True
                  # 진행률 정보가 전혀 없는 단일 청크 응답은 완료로 간주
                  elif not has_progress_keys:
                    is_complete = True

                if is_complete:
                  # storage_path는 메인 백엔드의 경로 사용
                  # 예: /storage/3d/adv_k2_tank
                  image_path = result.get("image_path", "")
                  if image_path and image_path.startswith("/storage/3d/"):
                    # 이미지 경로에서 폴더 경로 추출
                    # /storage/3d/adv_k2_tank/path/image.jpg -> /storage/3d/adv_k2_tank
                    path_parts = image_path.split("/")
                    if len(path_parts) >= 4:
                      storage_path = "/".join(path_parts[:4])  # /storage/3d/folder_name
                    else:
                      storage_path = f"/storage/3d/adv_{payload.get('dataset_name', 'unknown')}"
                  else:
                    storage_path = f"/storage/3d/adv_{payload.get('dataset_name', 'unknown')}"

                  processed_images = (
                    result.get("processed_images")
                    or result.get("processed")
                    or result.get("successful")
                    or result.get("total")
                  )

                  # AttackDataset3D 및 출력 Dataset3D 생성
                  timestamp_suffix = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                  base_dataset_name = payload.get("dataset_name", "unknown")
                  attack_dataset_name = f"{base_dataset_name}_attack_{timestamp_suffix}"
                  output_dataset_name = f"{base_dataset_name}_output_{timestamp_suffix}"

                  # 1) 출력 Dataset3D 생성 (이미지 레코드 없이 메타데이터만 저장)
                  dataset_metadata = {
                    "attack_method": payload.get("attack_method"),
                    "patch_name": payload.get("patch_name"),
                    "map_name": payload.get("map_name"),
                    "weather_name": payload.get("weather_name"),
                    "time_name": payload.get("time_name"),
                    "object_name": payload.get("object_name"),
                    "image_width": payload.get("image_width"),
                    "image_height": payload.get("image_height"),
                    "loc_total": loc_total,
                    "batch_total": batch_total,
                    "processed_images": processed_images,
                    "storage_path": storage_path,
                    "source": "carla_simulator",
                  }

                  output_dataset = await _save_dataset_3d(
                    dataset_name=output_dataset_name,
                    storage_path=storage_path,
                    metadata=dataset_metadata,
                    db=db,
                  )

                  output_dataset_id = str(output_dataset.id) if output_dataset else None
                  storage_key_base = storage_path.replace("/storage/", "") if storage_path.startswith("/storage/") else None

                  # 1-1) Image3D 레코드 저장 (경로 스캔)
                  saved_image_count = 0
                  saved_annotation_count = 0
                  if output_dataset and storage_key_base:
                    try:
                      base_dir = Path(settings.STORAGE_ROOT) / storage_key_base
                      if base_dir.exists():
                        from PIL import Image as PILImage
                        image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
                        image_records = []
                        for file_path in base_dir.rglob("*"):
                          if not file_path.is_file():
                            continue
                          if file_path.suffix.lower() not in image_exts:
                            continue

                          storage_key = str(file_path.relative_to(settings.STORAGE_ROOT))
                          mime_type, _ = mimetypes.guess_type(file_path.name)

                          # 이미지 파일 열어서 크기 가져오기
                          width, height = None, None
                          try:
                            with PILImage.open(file_path) as img:
                              width, height = img.size
                          except Exception as img_err:
                            logger.warning(f"Failed to read image size for {file_path.name}: {img_err}")

                          image_records.append(Image3D(
                            dataset_id=output_dataset.id,
                            file_name=file_path.name,
                            storage_key=storage_key,
                            mime_type=mime_type or "image/png",
                            width=width,
                            height=height,
                          ))

                        if image_records:
                          db.add_all(image_records)
                          await db.commit()
                          await db.refresh(output_dataset)
                          saved_image_count = len(image_records)
                          logger.info(f"Saved {saved_image_count} Image3D records for dataset {output_dataset_id}")

                          # Image3D 레코드 저장 후 어노테이션 파싱 및 저장
                          # file_name -> image_3d_id 매핑 생성
                          image_id_map = {img.file_name: img.id for img in image_records}

                          # CARLA가 annotations JSON을 생성할 때까지 대기 (폴링)
                          import asyncio
                          annotations_dir = base_dir / "annotations"
                          json_files = []

                          max_wait_seconds = 30
                          poll_interval = 0.5
                          waited_seconds = 0

                          logger.info(f"Waiting for annotations JSON in {annotations_dir}")
                          while waited_seconds < max_wait_seconds:
                            if annotations_dir.exists():
                              json_files = list(annotations_dir.glob("*.json"))
                              if json_files:
                                logger.info(f"Found annotations JSON: {json_files[0]}")
                                break
                            await asyncio.sleep(poll_interval)
                            waited_seconds += poll_interval

                          # 어노테이션 파싱 시도
                          if json_files:
                            saved_annotation_count = await AnnotationParser.parse_and_save(
                              annotations_dir=annotations_dir,
                              dataset_id=output_dataset.id,
                              image_id_map=image_id_map,
                              db=db
                            )
                            logger.info(f"Saved {saved_annotation_count} annotations for attack dataset {output_dataset_id}")
                          else:
                            logger.warning(
                              f"Annotations not found after {max_wait_seconds}s wait. "
                              f"Use POST /api/v1/carla/datasets_3d/{output_dataset_id}/parse-annotations later."
                            )
                        else:
                          logger.warning(f"No image files found under {base_dir} for dataset {output_dataset_id}")
                      else:
                        logger.warning(f"Image base directory does not exist: {base_dir}")
                    except Exception as e:
                      logger.error(f"Failed to save Image3D records: {e}", exc_info=True)

                  # 2) AttackDataset3D 생성 (output_dataset_id 연결)
                  attack_dataset = AttackDataset3D(
                    name=attack_dataset_name,
                    description=f"Attack dataset with {payload.get('attack_method')} patch applied",
                    attack_type=AttackType.PATCH,
                    output_dataset_id=output_dataset_id,
                    target_class=payload.get("object_name"),
                    patch_id=None,
                    parameters={
                      "attack_method": payload.get("attack_method"),
                      "patch_name": payload.get("patch_name"),
                      "map_name": payload.get("map_name"),
                      "weather_name": payload.get("weather_name"),
                      "time_name": payload.get("time_name"),
                      "object_name": payload.get("object_name"),
                      "image_width": payload.get("image_width"),
                      "image_height": payload.get("image_height"),
                      "loc_total": loc_total,
                      "batch_total": batch_total,
                      "storage_path": storage_path,
                      "processed_images": processed_images,
                      "output_dataset_name": output_dataset_name,
                      "output_dataset_id": output_dataset_id,
                      "source": "carla_simulator",
                    },
                  )
                  try:
                    db.add(attack_dataset)
                    await db.commit()
                    await db.refresh(attack_dataset)
                    attack_dataset_saved = True
                    completion_data = {
                      "state": 200,
                      "result": {
                        **result,
                        "attack_dataset_id": str(attack_dataset.id),
                        "output_dataset_id": output_dataset_id,
                        "storage_path": storage_path,
                        "image_count": saved_image_count,
                        "annotation_count": saved_annotation_count
                      },
                      "message": f"Attack dataset saved with {saved_image_count} images and {saved_annotation_count} annotations",
                    }
                    yield json.dumps(completion_data, ensure_ascii=False) + "\n"
                    continue
                  except Exception as e:
                    await db.rollback()
                    logger.error(f"Failed to save attack dataset: {e}")

              # 수정된 데이터 전달 (이미지 경로가 업데이트되었을 수 있음)
              yield json.dumps(data, ensure_ascii=False) + "\n"
      except httpx.HTTPError as e:
        logger.error(f"Failed to connect to CARLA backend: {e}")
        error_payload = {"state": 500, "message": f"CARLA backend connection failed: {e}"}
        yield json.dumps(error_payload) + "\n"

  return StreamingResponse(event_generator(), media_type="application/x-ndjson")


# GET proxy endpoints


@router.get("/sim_map_list")
async def get_maps(db: AsyncSession = Depends(get_db)):
  logger.info("🎯 CARLA endpoint hit: /sim_map_list")
  await _log_system(db, "sim_map_list", "fetch maps")
  return await _proxy_get("/sim_map_list")


@router.get("/sim_weather_list")
async def get_weathers(db: AsyncSession = Depends(get_db)):
  logger.info("🎯 CARLA endpoint hit: /sim_weather_list")
  await _log_system(db, "sim_weather_list", "fetch weathers")
  return await _proxy_get("/sim_weather_list")


@router.get("/sim_time_list")
async def get_times(db: AsyncSession = Depends(get_db)):
  logger.info("🎯 CARLA endpoint hit: /sim_time_list")
  await _log_system(db, "sim_time_list", "fetch times")
  return await _proxy_get("/sim_time_list")


@router.get("/sim_object_list")
async def get_objects(db: AsyncSession = Depends(get_db)):
  logger.info("🎯 CARLA endpoint hit: /sim_object_list")
  await _log_system(db, "sim_object_list", "fetch objects")
  return await _proxy_get("/sim_object_list")


@router.get("/sim_attack_list")
async def get_attack_methods(db: AsyncSession = Depends(get_db)):
  await _log_system(db, "sim_attack_list", "fetch attacks")
  return await _proxy_get("/sim_attack_list")


@router.get("/get_detector_list")
async def get_detectors(db: AsyncSession = Depends(get_db)):
  """
  메인 백엔드 DB에서 탐지 모델 목록 조회.

  반환 형식: {state: 200, message: "Success", result: [{id: "...", name: "..."}]}
  """
  await _log_system(db, "get_detector_list", "fetch detectors from main DB")

  try:
    # 메인 백엔드 DB에서 모델 목록 조회
    models = await crud.od_model.get_multi(db, skip=0, limit=100)

    # 카를라 백엔드 형식에 맞게 변환
    result = [{"id": str(model.id), "name": model.name} for model in models]

    return JSONResponse(content={
      "state": 200,
      "message": "Success",
      "result": result
    })
  except Exception as e:
    logger.error(f"Failed to fetch detector list: {e}")
    raise HTTPException(status_code=500, detail=str(e))


@router.get("/sim_dataset_list")
async def get_datasets(db: AsyncSession = Depends(get_db)):
  await _log_system(db, "sim_dataset_list", "fetch datasets")
  return await _proxy_get("/sim_dataset_list")


@router.get("/sim_patch_list")
async def get_patch_list(
  object_name: str = Query(..., description="CARLA object name"),
  attack_method: str = Query(..., description="Attack method"),
  db: AsyncSession = Depends(get_db),
):
  params = {"object_name": object_name, "attack_method": attack_method}
  await _log_system(db, "sim_patch_list", "fetch patch list", params)
  return await _proxy_get("/sim_patch_list", params=params)


# POST streaming proxy endpoints


@router.post("/sim_generate")
async def generate_dataset(payload: Dict, db: AsyncSession = Depends(get_db)):
  """
  3D 데이터셋 생성 요청 처리.

  완료 시 Dataset3D 테이블에 메타데이터 저장.
  """
  return await _proxy_stream_post("/sim_generate", payload, db, log_action="sim_generate", save_dataset=True)


@router.post("/sim_gen_patch")
async def generate_patch(request: PatchGenerationRequest, db: AsyncSession = Depends(get_db)):
  """
  패치 생성 요청 처리.

  1. 프론트엔드에서 model_id 또는 object_detection_name 수신
  2. model_id가 있으면 DB에서 모델 정보 조회
  3. 모델 정보(model_path, class_name, input_size 등)를 payload에 추가
  4. 카를라 백엔드로 전달
  """
  payload = request.dict()

  # model_id가 있으면 DB에서 모델 정보 조회
  if request.model_id:
    try:
      # 모델 조회
      model = await crud.od_model.get(db, id=request.model_id)
      if not model:
        raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found")

      # object_detection_name 설정
      payload["object_detection_name"] = model.name

      # 내장 모델이 아닌 경우에만 상세 정보 추가
      if model.name not in ["yolov11", "yolov8", "rt-detr", "efficientdet_d2_tank"]:
        # WEIGHTS artifact 조회
        stmt = select(ODModelArtifact).where(
          ODModelArtifact.model_id == request.model_id,
          ODModelArtifact.artifact_type == ArtifactType.WEIGHTS
        )
        result = await db.execute(stmt)
        artifact = result.scalar_one_or_none()

        if artifact:
          payload["model_path"] = artifact.storage_path
        else:
          logger.warning(f"No WEIGHTS artifact found for model {request.model_id}")
          payload["model_path"] = ""

        # framework를 model_type으로 변환
        payload["model_type"] = model.framework.value if model.framework else ""

        # labelmap에서 class_name 추출
        if model.labelmap:
          payload["class_name"] = list(model.labelmap.values())
        else:
          payload["class_name"] = request.class_name

        # input_spec에서 input_size 추출
        if model.input_spec and "size" in model.input_spec:
          input_size_val = model.input_spec["size"]
          if isinstance(input_size_val, list):
            payload["input_size"] = input_size_val
          elif isinstance(input_size_val, dict):
            # {width: 640, height: 640} 형식 지원
            payload["input_size"] = [input_size_val.get("width", 640), input_size_val.get("height", 640)]
        else:
          payload["input_size"] = request.input_size

        payload["device_type"] = request.device_type or ""
      else:
        # 내장 모델은 카를라 백엔드가 알아서 처리
        payload["model_path"] = ""
        payload["model_type"] = ""
        payload["class_name"] = request.class_name
        payload["input_size"] = request.input_size
        payload["device_type"] = ""

      await _log_system(db, "sim_gen_patch", f"Model {model.name} loaded", {"model_id": str(request.model_id)})

    except HTTPException:
      raise
    except Exception as e:
      logger.error(f"Failed to load model {request.model_id}: {e}")
      raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

  elif request.object_detection_name:
    # object_detection_name이 직접 제공된 경우 (하위 호환성)
    payload["object_detection_name"] = request.object_detection_name
  else:
    raise HTTPException(status_code=400, detail="Either model_id or object_detection_name must be provided")

  # 카를라 백엔드로 전달하기 전에 model_id 저장 (나중에 Patch3D 저장 시 사용)
  original_model_id = request.model_id

  # model_id는 카를라 백엔드로 전달하지 않음 (카를라는 UUID를 이해하지 못함)
  payload.pop("model_id", None)

  # save_patch=True로 설정하여 완료 시 DB에 저장
  # model_id는 별도 파라미터로 전달 (payload에 포함하지 않음)
  return await _proxy_stream_post("/sim_gen_patch", payload, db, log_action="sim_gen_patch", save_patch=True, model_id=original_model_id)


@router.post("/sim_apply_texture")
async def apply_texture(payload: Dict, db: AsyncSession = Depends(get_db)):
  """
  텍스처(패치) 적용 데이터셋 생성 요청 처리.

  완료 시 AttackDataset3D 및 output Dataset3D 테이블에 저장.
  """
  return await _proxy_stream_post("/sim_apply_texture", payload, db, log_action="sim_apply_texture", save_attack_dataset=True)


# ============================================
# 3D Dataset CRUD Endpoints
# ============================================

@router.get("/datasets_3d", response_model=List[Dict])
async def list_datasets_3d(
  skip: int = Query(0, ge=0),
  limit: int = Query(100, le=1000),
  exclude_attack_output: bool = Query(True, description="Exclude attack output datasets from results"),
  db: AsyncSession = Depends(get_db)
):
  """List all 3D datasets.

  By default, excludes datasets that are outputs of adversarial attacks.
  Set exclude_attack_output=false to include all datasets.
  """
  from app.models.dataset_3d import Dataset3D
  from sqlalchemy import select, func

  # Create subquery for attack output dataset IDs
  attack_output_subq = select(AttackDataset3D.output_dataset_id).where(
    AttackDataset3D.deleted_at.is_(None),
    AttackDataset3D.output_dataset_id.is_not(None)
  ).scalar_subquery()

  # Query datasets with image count
  stmt = (
    select(
      Dataset3D,
      func.count(Image3D.id).label('image_count')
    )
    .outerjoin(Image3D, Dataset3D.id == Image3D.dataset_id)
    .where(Dataset3D.deleted_at.is_(None))
  )

  # Exclude attack output datasets if requested (default behavior)
  if exclude_attack_output:
    stmt = stmt.where(Dataset3D.id.not_in(attack_output_subq))

  stmt = stmt.group_by(Dataset3D.id).offset(skip).limit(limit)

  result = await db.execute(stmt)
  rows = result.all()

  # Collect attack output IDs for is_attack_output flag (useful when exclude_attack_output=false)
  attack_output_rows = await db.execute(
    select(AttackDataset3D.output_dataset_id).where(
      AttackDataset3D.deleted_at.is_(None),
      AttackDataset3D.output_dataset_id.is_not(None)
    )
  )
  attack_output_ids = {str(row[0]) for row in attack_output_rows if row[0]}

  return [
    {
      "id": str(row[0].id),
      "name": row[0].name,
      "description": row[0].description,
      "storage_path": row[0].storage_path,
      # DB에 이미지가 없다면 메타데이터 기반 추론값으로 보완
      "image_count": row[1] or _infer_image_count(row[0].metadata_),
      "metadata": row[0].metadata_,
      "created_at": row[0].created_at.isoformat() if row[0].created_at else None,
      "type": "3D_IMAGE",
      "is_attack_output": str(row[0].id) in attack_output_ids
    }
    for row in rows
  ]


@router.get("/datasets_3d/{dataset_id}")
async def get_dataset_3d(
  dataset_id: UUID,
  db: AsyncSession = Depends(get_db)
):
  """Get a 3D dataset by ID."""
  from app.models.dataset_3d import Dataset3D
  from sqlalchemy import select

  stmt = select(Dataset3D).where(
    Dataset3D.id == dataset_id,
    Dataset3D.deleted_at.is_(None)
  )
  result = await db.execute(stmt)
  dataset = result.scalar_one_or_none()

  if not dataset:
    raise HTTPException(status_code=404, detail="Dataset not found")

  return {
    "id": str(dataset.id),
    "name": dataset.name,
    "description": dataset.description,
    "storage_path": dataset.storage_path,
    "metadata": dataset.metadata_,
    "created_at": dataset.created_at.isoformat() if dataset.created_at else None,
  }


@router.get("/datasets_3d/{dataset_id}/images")
async def get_dataset_3d_images(
  dataset_id: UUID,
  skip: int = Query(0, ge=0),
  limit: int = Query(100, le=1000),
  db: AsyncSession = Depends(get_db)
):
  """Get images from a 3D dataset."""
  from app.models.dataset_3d import Dataset3D, Image3D
  from sqlalchemy import select

  # Check if dataset exists
  stmt = select(Dataset3D).where(
    Dataset3D.id == dataset_id,
    Dataset3D.deleted_at.is_(None)
  )
  result = await db.execute(stmt)
  dataset = result.scalar_one_or_none()

  if not dataset:
    raise HTTPException(status_code=404, detail="Dataset not found")

  # Get images
  stmt = (
    select(Image3D)
    .where(
      Image3D.dataset_id == dataset_id,
      Image3D.deleted_at.is_(None)
    )
    .offset(skip)
    .limit(limit)
  )
  result = await db.execute(stmt)
  images = result.scalars().all()

  return {
    "total": len(images),
    "images": [
      {
        "id": str(img.id),
        "dataset_id": str(img.dataset_id),
        "file_name": img.file_name,
        "storage_key": img.storage_key,
        "width": img.width,
        "height": img.height,
        "depth": img.depth,
        "mime_type": img.mime_type,
        "metadata": img.metadata_,
        "created_at": img.created_at.isoformat() if img.created_at else None,
      }
      for img in images
    ]
  }


@router.delete("/datasets_3d/{dataset_id}", status_code=204)
async def delete_dataset_3d(
  dataset_id: UUID,
  db: AsyncSession = Depends(get_db)
):
  """Soft delete a 3D dataset."""
  from app.models.dataset_3d import Dataset3D
  from sqlalchemy import select
  from datetime import datetime

  stmt = select(Dataset3D).where(
    Dataset3D.id == dataset_id,
    Dataset3D.deleted_at.is_(None)
  )
  result = await db.execute(stmt)
  dataset = result.scalar_one_or_none()

  if not dataset:
    raise HTTPException(status_code=404, detail="Dataset not found")

  dataset.deleted_at = datetime.utcnow()
  await db.commit()

  return None


@router.post("/datasets_3d/{dataset_id}/update-image-sizes")
async def update_image_sizes(
  dataset_id: UUID,
  db: AsyncSession = Depends(get_db)
):
  """
  Update image sizes for all Image3D records in a dataset.

  Reads actual image files and updates width/height in database.
  Useful for datasets created before image size tracking was added.
  """
  from sqlalchemy import select
  from pathlib import Path
  from PIL import Image as PILImage

  # Get dataset
  stmt = select(Dataset3D).where(
    Dataset3D.id == dataset_id,
    Dataset3D.deleted_at.is_(None)
  )
  result = await db.execute(stmt)
  dataset = result.scalar_one_or_none()

  if not dataset:
    raise HTTPException(status_code=404, detail="Dataset not found")

  # Get all images
  stmt = select(Image3D).where(
    Image3D.dataset_id == dataset_id,
    Image3D.deleted_at.is_(None)
  )
  result = await db.execute(stmt)
  images = result.scalars().all()

  if not images:
    raise HTTPException(status_code=400, detail="No images in dataset")

  # Update sizes
  updated_count = 0
  failed_count = 0

  for img in images:
    if img.width and img.height:
      continue  # Skip if already has size

    try:
      file_path = Path(settings.STORAGE_ROOT) / img.storage_key
      if not file_path.exists():
        logger.warning(f"Image file not found: {file_path}")
        failed_count += 1
        continue

      with PILImage.open(file_path) as pil_img:
        img.width, img.height = pil_img.size
        updated_count += 1
        logger.info(f"Updated size for {img.file_name}: {img.width}x{img.height}")

    except Exception as e:
      logger.error(f"Failed to update size for {img.file_name}: {e}")
      failed_count += 1

  if updated_count > 0:
    await db.commit()

  return {
    "dataset_id": str(dataset_id),
    "dataset_name": dataset.name,
    "total_images": len(images),
    "updated": updated_count,
    "failed": failed_count,
    "message": f"Updated {updated_count} images, {failed_count} failed"
  }


@router.post("/datasets_3d/{dataset_id}/parse-annotations")
async def parse_annotations_manually(
  dataset_id: UUID,
  db: AsyncSession = Depends(get_db)
):
  """
  Manually trigger annotation parsing for a 3D dataset.

  Useful when annotations were created after the dataset was saved,
  or when annotation parsing failed during dataset creation.
  """
  from sqlalchemy import select
  from pathlib import Path

  # Get dataset
  stmt = select(Dataset3D).where(
    Dataset3D.id == dataset_id,
    Dataset3D.deleted_at.is_(None)
  )
  result = await db.execute(stmt)
  dataset = result.scalar_one_or_none()

  if not dataset:
    raise HTTPException(status_code=404, detail="Dataset not found")

  # Get images
  stmt = select(Image3D).where(
    Image3D.dataset_id == dataset_id,
    Image3D.deleted_at.is_(None)
  )
  result = await db.execute(stmt)
  images = result.scalars().all()

  if not images:
    raise HTTPException(status_code=400, detail="No images in dataset")

  # Parse annotations
  image_id_map = {img.file_name: img.id for img in images}

  # Calculate base directory from storage_path
  storage_path = dataset.storage_path
  if storage_path.startswith("/storage/"):
    storage_key = storage_path.replace("/storage/", "")
  else:
    storage_key = storage_path

  base_dir = Path(settings.STORAGE_ROOT) / storage_key
  annotations_dir = base_dir / "annotations"

  logger.info(f"Attempting to parse annotations from {annotations_dir}")

  if not annotations_dir.exists():
    raise HTTPException(
      status_code=404,
      detail=f"Annotations directory not found: {annotations_dir}"
    )

  annotation_count = await AnnotationParser.parse_and_save(
    annotations_dir=annotations_dir,
    dataset_id=dataset_id,
    image_id_map=image_id_map,
    db=db
  )

  return {
    "dataset_id": str(dataset_id),
    "dataset_name": dataset.name,
    "annotations_parsed": annotation_count,
    "message": f"Successfully parsed {annotation_count} annotations"
  }


# ============================================
# 3D Patch CRUD Endpoints
# ============================================

@router.get("/patches_3d", response_model=List[Dict])
async def list_patches_3d(
  skip: int = Query(0, ge=0),
  limit: int = Query(100, le=1000),
  target_class: Optional[str] = Query(None),
  db: AsyncSession = Depends(get_db)
):
  """List all 3D patches."""
  from sqlalchemy import select

  stmt = select(Patch3D).where(Patch3D.deleted_at.is_(None))

  if target_class:
    stmt = stmt.where(Patch3D.target_class == target_class)

  stmt = stmt.offset(skip).limit(limit)

  result = await db.execute(stmt)
  patches = result.scalars().all()

  return [
    {
      "id": str(patch.id),
      "name": patch.name,
      "description": patch.description,
      "target_model_id": str(patch.target_model_id) if patch.target_model_id else None,
      "source_dataset_id": str(patch.source_dataset_id) if patch.source_dataset_id else None,
      "target_class": patch.target_class,
      "method": patch.method,
      "hyperparameters": patch.hyperparameters,
      "patch_metadata": patch.patch_metadata,
      "storage_key": patch.storage_key,
      "file_name": patch.file_name,
      "size_bytes": patch.size_bytes,
      "created_at": patch.created_at.isoformat() if patch.created_at else None,
    }
    for patch in patches
  ]


@router.get("/patches_3d/{patch_id}")
async def get_patch_3d(
  patch_id: UUID,
  db: AsyncSession = Depends(get_db)
):
  """Get a 3D patch by ID."""
  from sqlalchemy import select

  stmt = select(Patch3D).where(
    Patch3D.id == patch_id,
    Patch3D.deleted_at.is_(None)
  )
  result = await db.execute(stmt)
  patch = result.scalar_one_or_none()

  if not patch:
    raise HTTPException(status_code=404, detail="Patch not found")

  return {
    "id": str(patch.id),
    "name": patch.name,
    "description": patch.description,
    "target_model_id": str(patch.target_model_id) if patch.target_model_id else None,
    "source_dataset_id": str(patch.source_dataset_id) if patch.source_dataset_id else None,
    "target_class": patch.target_class,
    "method": patch.method,
    "hyperparameters": patch.hyperparameters,
    "patch_metadata": patch.patch_metadata,
    "storage_key": patch.storage_key,
    "file_name": patch.file_name,
    "size_bytes": patch.size_bytes,
    "created_at": patch.created_at.isoformat() if patch.created_at else None,
  }


@router.delete("/patches_3d/{patch_id}", status_code=204)
async def delete_patch_3d(
  patch_id: UUID,
  db: AsyncSession = Depends(get_db)
):
  """Soft delete a 3D patch."""
  from sqlalchemy import select
  from datetime import datetime

  stmt = select(Patch3D).where(
    Patch3D.id == patch_id,
    Patch3D.deleted_at.is_(None)
  )
  result = await db.execute(stmt)
  patch = result.scalar_one_or_none()

  if not patch:
    raise HTTPException(status_code=404, detail="Patch not found")

  patch.deleted_at = datetime.utcnow()
  await db.commit()

  return None


# ============================================
# 3D Attack Dataset CRUD Endpoints
# ============================================

@router.get("/attack_datasets_3d", response_model=List[Dict])
async def list_attack_datasets_3d(
  skip: int = Query(0, ge=0),
  limit: int = Query(100, le=1000),
  target_class: Optional[str] = Query(None),
  db: AsyncSession = Depends(get_db)
):
  """List all 3D attack datasets."""
  from sqlalchemy import select, func

  stmt = select(AttackDataset3D).where(AttackDataset3D.deleted_at.is_(None))

  if target_class:
    stmt = stmt.where(AttackDataset3D.target_class == target_class)

  stmt = stmt.offset(skip).limit(limit)

  result = await db.execute(stmt)
  datasets = result.scalars().all()

  # Get actual image counts from output datasets
  output_list = []
  for ds in datasets:
    # Calculate actual image count from output_dataset_id
    image_count = 0
    if ds.output_dataset_id:
      try:
        count_stmt = (
          select(func.count(Image3D.id))
          .where(
            Image3D.dataset_id == ds.output_dataset_id,
            Image3D.deleted_at.is_(None)
          )
        )
        count_result = await db.execute(count_stmt)
        image_count = count_result.scalar() or 0
      except Exception as e:
        logger.warning(f"Failed to count images for attack dataset {ds.id}: {e}")
        # Fallback to parameters.processed_images if count fails
        image_count = ds.parameters.get("processed_images", 0) if ds.parameters else 0

    # Update parameters with actual image count
    updated_parameters = ds.parameters.copy() if ds.parameters else {}
    updated_parameters["processed_images"] = image_count

    output_list.append({
      "id": str(ds.id),
      "name": ds.name,
      "description": ds.description,
      "attack_type": ds.attack_type.value if ds.attack_type else None,
      "target_model_id": str(ds.target_model_id) if ds.target_model_id else None,
      "output_dataset_id": str(ds.output_dataset_id) if ds.output_dataset_id else None,
      "target_class": ds.target_class,
      "patch_id": str(ds.patch_id) if ds.patch_id else None,
      "parameters": updated_parameters,
      "image_count": image_count,  # Add top-level image_count field
      "created_at": ds.created_at.isoformat() if ds.created_at else None,
    })

  return output_list


@router.get("/attack_datasets_3d/{dataset_id}")
async def get_attack_dataset_3d(
  dataset_id: UUID,
  db: AsyncSession = Depends(get_db)
):
  """Get a 3D attack dataset by ID."""
  from sqlalchemy import select

  stmt = select(AttackDataset3D).where(
    AttackDataset3D.id == dataset_id,
    AttackDataset3D.deleted_at.is_(None)
  )
  result = await db.execute(stmt)
  ds = result.scalar_one_or_none()

  if not ds:
    raise HTTPException(status_code=404, detail="Attack dataset not found")

  return {
    "id": str(ds.id),
    "name": ds.name,
    "description": ds.description,
    "attack_type": ds.attack_type.value if ds.attack_type else None,
    "target_model_id": str(ds.target_model_id) if ds.target_model_id else None,
    "output_dataset_id": str(ds.output_dataset_id) if ds.output_dataset_id else None,
    "target_class": ds.target_class,
    "patch_id": str(ds.patch_id) if ds.patch_id else None,
    "parameters": ds.parameters,
    "created_at": ds.created_at.isoformat() if ds.created_at else None,
  }


@router.delete("/attack_datasets_3d/{dataset_id}", status_code=204)
async def delete_attack_dataset_3d(
  dataset_id: UUID,
  db: AsyncSession = Depends(get_db)
):
  """Soft delete a 3D attack dataset."""
  from sqlalchemy import select
  from datetime import datetime

  stmt = select(AttackDataset3D).where(
    AttackDataset3D.id == dataset_id,
    AttackDataset3D.deleted_at.is_(None)
  )
  result = await db.execute(stmt)
  ds = result.scalar_one_or_none()

  if not ds:
    raise HTTPException(status_code=404, detail="Attack dataset not found")

  ds.deleted_at = datetime.utcnow()
  await db.commit()

  return None


# ============================================
# Download Endpoints
# ============================================

@router.get("/datasets_3d/{dataset_id}/download")
async def download_dataset_3d(
  dataset_id: UUID,
  db: AsyncSession = Depends(get_db)
):
  """
  Download 3D dataset as ZIP file.

  Returns a ZIP file containing all images and annotations from the dataset.
  """
  from sqlalchemy import select
  from fastapi.responses import StreamingResponse
  import zipfile
  import io
  from pathlib import Path

  # Get dataset
  stmt = select(Dataset3D).where(
    Dataset3D.id == dataset_id,
    Dataset3D.deleted_at.is_(None)
  )
  result = await db.execute(stmt)
  dataset = result.scalar_one_or_none()

  if not dataset:
    raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")

  # Get images
  stmt = select(Image3D).where(
    Image3D.dataset_id == dataset_id,
    Image3D.deleted_at.is_(None)
  )
  result = await db.execute(stmt)
  images = result.scalars().all()

  if not images:
    raise HTTPException(status_code=404, detail="No images found in dataset")

  # Create ZIP in memory
  zip_buffer = io.BytesIO()

  with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
    # Add images
    for img in images:
      try:
        file_path = Path(settings.STORAGE_ROOT) / img.storage_key
        if file_path.exists():
          zip_file.write(file_path, f"images/{img.file_name}")
      except Exception as e:
        logger.warning(f"Failed to add image {img.file_name} to ZIP: {e}")

    # Add metadata
    import json
    metadata = {
      "dataset_id": str(dataset.id),
      "dataset_name": dataset.name,
      "created_at": dataset.created_at.isoformat() if dataset.created_at else None,
      "image_count": len(images),
      "metadata": dataset.metadata_
    }
    zip_file.writestr("metadata.json", json.dumps(metadata, indent=2, ensure_ascii=False))

  zip_buffer.seek(0)

  # Return as streaming response
  return StreamingResponse(
    iter([zip_buffer.getvalue()]),
    media_type="application/zip",
    headers={
      "Content-Disposition": f'attachment; filename="{dataset.name}.zip"'
    }
  )


@router.get("/patches_3d/{patch_id}/download")
async def download_patch_3d(
  patch_id: UUID,
  db: AsyncSession = Depends(get_db)
):
  """
  Download 3D patch file.

  Returns the patch image file.
  """
  from sqlalchemy import select
  from fastapi.responses import FileResponse
  from pathlib import Path

  # Get patch
  stmt = select(Patch3D).where(
    Patch3D.id == patch_id,
    Patch3D.deleted_at.is_(None)
  )
  result = await db.execute(stmt)
  patch = result.scalar_one_or_none()

  if not patch:
    raise HTTPException(status_code=404, detail=f"Patch {patch_id} not found")

  if not patch.storage_key:
    raise HTTPException(status_code=404, detail="Patch file not found")

  file_path = Path(settings.STORAGE_ROOT) / patch.storage_key

  if not file_path.exists():
    raise HTTPException(status_code=404, detail=f"Patch file does not exist: {file_path}")

  # Determine media type
  media_type = patch.file_name.split('.')[-1] if patch.file_name else "png"
  media_type_map = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "bmp": "image/bmp",
  }

  return FileResponse(
    path=file_path,
    media_type=media_type_map.get(media_type, "application/octet-stream"),
    filename=patch.file_name or f"{patch.name}.png"
  )


# ==========================================
# Vehicles 3D Model Upload
# ==========================================

from fastapi import UploadFile, File
import shutil
import zipfile
import tempfile

@router.post("/vehicles/upload")
async def upload_vehicles_folder(
  files: List[UploadFile] = File(..., description="Vehicles 폴더의 모든 파일 (ZIP 또는 개별 파일)")
):
  """
  CARLA Vehicles 폴더 업로드 (전체 덮어쓰기)
  
  - ZIP 파일로 업로드하거나 폴더 내 모든 파일을 개별 업로드
  - 기존 storage/simulator/Vehicle 폴더를 완전히 교체
  - 업로드 후 CARLA 시뮬레이터 재시작 필요
  """
  try:
    logger.info(f"Received {len(files)} files for Vehicles upload")
    
    # Target directory: storage/simulator/Vehicle
    target_dir = Path(settings.STORAGE_ROOT) / "simulator" / "Vehicle"
    
    # Backup existing folder (optional)
    if target_dir.exists():
      backup_dir = Path(settings.STORAGE_ROOT) / "simulator" / f"Vehicle_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
      logger.info(f"Backing up existing Vehicles to: {backup_dir}")
      shutil.copytree(target_dir, backup_dir)
      # Remove old folder
      shutil.rmtree(target_dir)
    
    # Create fresh directory
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Process files
    total_size = 0
    file_count = 0
    
    for upload_file in files:
      file_name = upload_file.filename
      
      # Check if it's a ZIP file
      if file_name.endswith('.zip'):
        logger.info(f"Extracting ZIP file: {file_name}")
        # Save ZIP to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
          content = await upload_file.read()
          tmp_file.write(content)
          tmp_zip_path = tmp_file.name
        
        # Extract ZIP
        try:
          with zipfile.ZipFile(tmp_zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
          file_count += len(zipfile.ZipFile(tmp_zip_path).namelist())
          total_size += len(content)
        finally:
          # Clean up temp file
          Path(tmp_zip_path).unlink()
      
      else:
        # Save individual file
        # Preserve directory structure if file_name contains paths
        file_path = target_dir / file_name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        content = await upload_file.read()
        with open(file_path, 'wb') as f:
          f.write(content)
        
        total_size += len(content)
        file_count += 1
        logger.debug(f"Saved file: {file_name} ({len(content)} bytes)")
    
    logger.info(f"Vehicles upload complete: {file_count} files, {total_size} bytes")
    
    return {
      "state": 200,
      "message": "Vehicles folder uploaded successfully. Please restart CARLA simulator to apply changes.",
      "result": {
        "file_count": file_count,
        "total_size": total_size,
        "target_path": str(target_dir),
        "warning": "CARLA 시뮬레이터를 재시작해야 변경사항이 반영됩니다."
      }
    }
  
  except Exception as e:
    logger.error(f"Vehicles upload failed: {e}", exc_info=True)
    raise HTTPException(
      status_code=500,
      detail=f"Vehicles upload failed: {str(e)}"
    )


@router.get("/vehicles/list")
async def list_vehicles():
  """
  CARLA 차량 목록 조회 (CARLA 백엔드 프록시)
  """
  try:
    async with httpx.AsyncClient(timeout=30.0) as client:
      response = await client.get(f"{CARLA_BASE_URL}/sim_object_list")
      response.raise_for_status()
      data = response.json()
      
      return {
        "state": 200,
        "message": "Success",
        "result": data.get("result", [])
      }
  
  except httpx.HTTPError as e:
    logger.error(f"Failed to fetch vehicle list from CARLA: {e}")
    raise HTTPException(
      status_code=502,
      detail=f"Failed to communicate with CARLA backend: {str(e)}"
    )
  except Exception as e:
    logger.error(f"Unexpected error fetching vehicle list: {e}")
    raise HTTPException(
      status_code=500,
      detail=f"Failed to fetch vehicle list: {str(e)}"
    )
