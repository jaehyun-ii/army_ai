"""
적대적 공격 평가 보고서 생성 서비스

평가 결과를 기반으로 PDF 보고서를 자동 생성합니다.
WeasyPrint를 사용하여 HTML 템플릿을 PDF로 변환합니다.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from uuid import UUID
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app import crud
from app.core.config import settings
from app.core.exceptions import NotFoundError, ValidationError

logger = logging.getLogger(__name__)


class ReportGenerationService:
    """적대적 공격 보고서 생성 서비스"""

    def __init__(self):
        # Use STORAGE_ROOT for evaluate directory (shared between 2D/3D)
        self.storage_root = Path(settings.STORAGE_ROOT)
        self.reports_dir = self.storage_root / "evaluate"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # 템플릿 경로
        self.template_dir = Path(__file__).parent.parent / "templates"
        self.adversarial_template_path = self.template_dir / "adversarial_report" / "report_template.html"
        self.base_template_path = self.template_dir / "base_evaluation" / "report_template.html"
        self.style_path = self.template_dir / "style.css"

        # 임시 차트 디렉토리
        self.temp_charts_dir = self.reports_dir / "temp_charts"
        self.temp_charts_dir.mkdir(exist_ok=True)

        # WeasyPrint와 matplotlib는 필요할 때만 import (선택적 의존성)
        self._weasyprint_available = False
        self._matplotlib_available = False

        try:
            from weasyprint import HTML, CSS
            from weasyprint.text.fonts import FontConfiguration
            self._HTML = HTML
            self._CSS = CSS
            self._FontConfiguration = FontConfiguration
            self._weasyprint_available = True
            logger.info("WeasyPrint 사용 가능")
        except ImportError:
            logger.warning("WeasyPrint를 사용할 수 없습니다. PDF 생성이 불가능합니다.")

        try:
            import matplotlib
            matplotlib.use('Agg')  # GUI 없이 사용
            import matplotlib.pyplot as plt
            import matplotlib.font_manager as fm
            import numpy as np

            # 한글 폰트 설정
            try:
                # matplotlib 폰트 캐시 재빌드
                fm._load_fontmanager(try_read_cache=False)

                # 시스템에 설치된 한글 폰트 찾기 (우선순위: Noto Sans CJK > Nanum > Malgun)
                font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
                korean_fonts = []

                # Noto Sans CJK 우선 검색
                korean_fonts = [f for f in font_list if 'NotoSansCJK' in f or 'NotoSansKR' in f]
                if not korean_fonts:
                    # Nanum 폰트 검색
                    korean_fonts = [f for f in font_list if 'Nanum' in f or 'NanumGothic' in f]
                if not korean_fonts:
                    # Malgun 폰트 검색
                    korean_fonts = [f for f in font_list if 'Malgun' in f]

                if korean_fonts:
                    font_path = korean_fonts[0]
                    font_prop = fm.FontProperties(fname=font_path)
                    plt.rcParams['font.family'] = font_prop.get_name()
                    logger.info(f"한글 폰트 설정 완료: {font_path}")
                else:
                    # 폰트 이름으로 직접 설정 시도
                    try:
                        plt.rcParams['font.family'] = 'Noto Sans CJK KR'
                        logger.info("한글 폰트 설정 완료: Noto Sans CJK KR (이름 기반)")
                    except:
                        plt.rcParams['font.family'] = 'DejaVu Sans'
                        logger.warning("한글 폰트를 찾을 수 없어 DejaVu Sans 사용")

                # 마이너스 기호 깨짐 방지
                plt.rcParams['axes.unicode_minus'] = False
            except Exception as font_error:
                logger.warning(f"폰트 설정 중 오류 발생: {font_error}")
                plt.rcParams['axes.unicode_minus'] = False

            self._plt = plt
            self._np = np
            self._matplotlib_available = True
            logger.info("Matplotlib 사용 가능")
        except ImportError:
            logger.warning("Matplotlib를 사용할 수 없습니다. 그래프 생성이 불가능합니다.")

    def check_dependencies(self) -> Dict[str, bool]:
        """의존성 확인"""
        return {
            "weasyprint": self._weasyprint_available,
            "matplotlib": self._matplotlib_available,
        }

    async def generate_evaluation_report(
        self,
        db: AsyncSession,
        evaluation_id: UUID,
        include_charts: bool = True,
    ) -> str:
        """
        평가 결과로부터 PDF 보고서 생성

        Args:
            db: Database session
            evaluation_id: 평가 ID (eval_runs.id)
            include_charts: 그래프 포함 여부

        Returns:
            생성된 PDF 파일 경로

        Raises:
            NotFoundError: 평가 결과를 찾을 수 없음
            ValidationError: 보고서 생성 실패
        """
        if not self._weasyprint_available:
            raise ValidationError(
                "WeasyPrint가 설치되지 않았습니다. "
                "pip install weasyprint를 실행하세요."
            )

        logger.info(f"평가 보고서 생성 시작: evaluation_id={evaluation_id}")

        # 1. 평가 결과 로드
        eval_run = await crud.evaluation.get_eval_run(db, evaluation_id)
        if not eval_run:
            raise NotFoundError(f"Evaluation run {evaluation_id} not found")

        # 2. 관련 데이터 로드
        model = await crud.od_model.get(db, id=eval_run.model_id)
        if not model:
            raise NotFoundError(f"Model {eval_run.model_id} not found")

        # Determine dimension and load dataset
        dimension = getattr(eval_run, "dimension", "2d") or "2d"
        dataset = None
        attack_dataset = None

        if dimension == "3d":
            if eval_run.base_dataset_3d_id:
                from sqlalchemy import select
                from app.models.dataset_3d import Dataset3D, AttackDataset3D
                dataset_row = await db.execute(
                    select(Dataset3D).where(Dataset3D.id == eval_run.base_dataset_3d_id, Dataset3D.deleted_at.is_(None))
                )
                dataset = dataset_row.scalar_one_or_none()
            # Load attack dataset if present
            if eval_run.attack_dataset_3d_id:
                from sqlalchemy import select
                from app.models.dataset_3d import AttackDataset3D
                attack_dataset_row = await db.execute(
                    select(AttackDataset3D).where(AttackDataset3D.id == eval_run.attack_dataset_3d_id, AttackDataset3D.deleted_at.is_(None))
                )
                attack_dataset = attack_dataset_row.scalar_one_or_none()
        else:
            if eval_run.base_dataset_id:
                dataset = await crud.dataset_2d.get(db, id=eval_run.base_dataset_id)
            # Load attack dataset if present
            if eval_run.attack_dataset_id:
                from sqlalchemy import select
                from app.models.dataset_2d import AttackDataset2D
                attack_dataset_row = await db.execute(
                    select(AttackDataset2D).where(AttackDataset2D.id == eval_run.attack_dataset_id, AttackDataset2D.deleted_at.is_(None))
                )
                attack_dataset = attack_dataset_row.scalar_one_or_none()


        # Determine evaluation type (attack vs base)
        is_attack_evaluation = bool(
            eval_run.attack_dataset_id or getattr(eval_run, "attack_dataset_3d_id", None)
        )

        # 3. 보고서 데이터 구성
        report_data = await self._prepare_report_data(
            db, eval_run, model, dataset, include_charts, dimension, is_attack_evaluation, attack_dataset
        )

        # 4. HTML 템플릿 로드 및 치환 (평가 유형에 따라 템플릿 선택)
        template_path = self.adversarial_template_path if is_attack_evaluation else self.base_template_path
        html_content = self._fill_template(report_data, template_path)

        # 5. PDF 생성
        output_filename = f"evaluation_{evaluation_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        output_path = self.reports_dir / output_filename

        # FontConfiguration 생성 (한글 폰트 지원)
        font_config = self._FontConfiguration()

        self._HTML(string=html_content, base_url=str(self.template_dir)).write_pdf(
            str(output_path),
            stylesheets=[self._CSS(filename=str(self.style_path))],
            font_config=font_config
        )

        logger.info(f"보고서 생성 완료: {output_path}")

        # 6. 임시 파일 정리
        self._cleanup_temp_files()

        # 상대 경로 반환 (storage root 기준)
        relative_path = output_path.relative_to(self.storage_root)
        return str(relative_path)

    async def _prepare_report_data(
        self,
        db: AsyncSession,
        eval_run: Any,
        model: Any,
        dataset: Any,
        include_charts: bool,
        dimension: str = "2d",
        is_attack_evaluation: bool = True,
        attack_dataset: Any = None
    ) -> Dict[str, Any]:
        """보고서 데이터 준비"""

        from app.schemas.evaluation import EvalDatasetType

        # 메트릭 추출: eval_dataset_results에서 로드
        base_result = await crud.evaluation.get_eval_dataset_result_by_run_and_type(
            db, eval_run_id=eval_run.id, dataset_type=EvalDatasetType.BASE
        )
        clean_metrics = base_result.metrics_summary if base_result else {}

        # Attack evaluation: load attack metrics
        if is_attack_evaluation:
            attack_result = await crud.evaluation.get_eval_dataset_result_by_run_and_type(
                db, eval_run_id=eval_run.id, dataset_type=EvalDatasetType.ATTACK
            )
            attacked_metrics = attack_result.metrics_summary if attack_result else {}

            # Robustness metrics (optional, stored in eval_run.metrics_summary)
            robustness_metrics = {}
            if eval_run.metrics_summary and 'robustness' in eval_run.metrics_summary:
                robustness_metrics = eval_run.metrics_summary['robustness']
        else:
            # Base evaluation: no attack metrics
            attacked_metrics = {}
            robustness_metrics = {}

        # Extract attack method from attack_dataset or params
        attack_method = 'N/A'
        attack_params = {}
        if attack_dataset:
            # Get attack info from attack dataset
            if hasattr(attack_dataset, 'parameters') and attack_dataset.parameters:
                attack_params = attack_dataset.parameters
                attack_method = attack_params.get('attack_method', 'N/A')
            # Fallback to attack_type for 2D attack datasets
            if attack_method == 'N/A' and hasattr(attack_dataset, 'attack_type'):
                attack_method = attack_dataset.attack_type
            # Fallback to method for 3D attack datasets
            if attack_method == 'N/A' and hasattr(attack_dataset, 'method'):
                attack_method = attack_dataset.method
        elif eval_run.params:
            # Fallback to eval_run params
            attack_method = eval_run.params.get('attack_method', 'N/A')
            attack_params = eval_run.params.get('attack_params', {})

        # Extract model info
        input_shape = model.input_spec.get('shape', [640, 640]) if model and model.input_spec else [640, 640]
        input_resolution = f"{input_shape[0]}×{input_shape[1]}" if len(input_shape) >= 2 else "640×640"

        # Extract dataset info
        dataset_name = dataset.name if dataset else 'N/A'
        total_images = 'N/A'
        if dataset:
            # Try multiple sources for image count
            if hasattr(dataset, 'image_count') and dataset.image_count:
                total_images = str(dataset.image_count)
            elif hasattr(dataset, 'metadata_') and dataset.metadata_ and 'image_count' in dataset.metadata_:
                total_images = str(dataset.metadata_['image_count'])
            elif hasattr(dataset, 'images') and dataset.images is not None:
                # Count from images relationship
                try:
                    total_images = str(len([img for img in dataset.images if img.deleted_at is None]))
                except Exception as e:
                    logger.debug(f"Failed to count images from relationship: {e}")

        # Extract target class info
        target_class = 'All Classes'
        if attack_dataset and hasattr(attack_dataset, 'target_class') and attack_dataset.target_class:
            target_class = attack_dataset.target_class
        elif attack_params and 'target_class' in attack_params:
            target_class = attack_params.get('target_class', 'All Classes')

        # Extract dataset classes (보유 클래스 목록)
        dataset_classes = 'N/A'
        if dataset and hasattr(dataset, 'metadata_') and dataset.metadata_ and 'classes' in dataset.metadata_:
            # Get from dataset metadata
            dataset_classes = ', '.join(dataset.metadata_['classes'])
        elif model and model.labelmap:
            # Fallback to model labelmap
            dataset_classes = ', '.join(list(model.labelmap.values()))

        # Extract hyperparameters (formatted by attack method)
        hyperparameters = self._format_hyperparameters(attack_method, attack_params)

        # 기본 정보
        data = {
            # 표지
            'EXPERIMENT_DATE': eval_run.created_at.strftime('%Y-%m-%d') if eval_run.created_at else datetime.now().strftime('%Y-%m-%d'),
            'EXPERIMENTER_NAME': 'AI Security Team',
            'SYSTEM_NAME': model.name if model else 'Unknown Model',
            'CLEAN_DATASET_NAME': dataset_name,
            'ATTACK_DATASET_NAME': attack_dataset.name if attack_dataset else '-',
            'REPORT_GENERATION_DATE': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),

            # 모델 정보
            'MODEL_NAME': model.name if model else 'Unknown Model',
            'MODEL_ARCHITECTURE': f"{model.framework.value.upper()}" if model and hasattr(model, 'framework') and model.framework else 'N/A',
            'INPUT_RESOLUTION': input_resolution,
            'NUM_CLASSES': str(len(model.labelmap)) if model and model.labelmap else 'N/A',
            'CLASS_NAMES': ', '.join(list(model.labelmap.values())) if model and model.labelmap and len(model.labelmap) > 0 else 'N/A',
            'INFERENCE_DEVICE': 'GPU (CUDA)' if model and getattr(model, 'device_type', None) == 'cuda' else 'CPU',
            'CONFIDENCE_THRESHOLD': '0.25',  # Default value

            # 데이터셋 정보
            'DATASET_NAME': dataset_name,
            'DATASET_DIMENSION': '3D' if dimension == '3d' else '2D',
            'TOTAL_IMAGES': total_images,
            'DATASET_CLASSES': dataset_classes,
            'TARGET_CLASS': target_class,
            'ANNOTATION_FORMAT': 'YOLO',

            # 공격 정보
            'ATTACK_TYPE': self._get_attack_type_display(attack_method),
            'ATTACK_DESCRIPTION': self._get_attack_description(attack_method),
            'ATTACK_INTENSITY': 'Medium',

            # 실험 환경
            'GPU_INFO': attack_params.get('device', eval_run.params.get('device', 'N/A')) if eval_run.params else 'N/A',
            'BATCH_SIZE': str(attack_params.get('batch_size', eval_run.params.get('batch_size', 'N/A'))) if eval_run.params else 'N/A',
            'TOTAL_TIME': self._format_duration(eval_run.started_at, eval_run.ended_at) if eval_run.started_at and eval_run.ended_at else 'N/A',
            'AVG_TIME_PER_IMAGE': self._calculate_avg_time_per_image(eval_run.started_at, eval_run.ended_at, total_images),
            'HYPERPARAMETERS': hyperparameters,
        }

        # 성능 지표
        if is_attack_evaluation:
            self._add_metrics_to_data(data, clean_metrics, attacked_metrics, robustness_metrics)
        else:
            self._add_base_metrics_to_data(data, clean_metrics)

        # 그래프 생성
        if include_charts and self._matplotlib_available:
            await self._generate_charts(db, data, clean_metrics, attacked_metrics, eval_run, is_attack_evaluation)

        # 추론 결과 이미지 생성 (공격 평가인 경우만)
        if is_attack_evaluation:
            await self._generate_inference_images(db, data, eval_run)
        else:
            data['CLEAN_INFERENCE_IMAGES'] = ''
            data['ATTACK_INFERENCE_IMAGES'] = ''

        # 결론
        data['FINDINGS_TEXT'] = self._generate_findings(clean_metrics, attacked_metrics, robustness_metrics, is_attack_evaluation)
        data['RECOMMENDATIONS_TEXT'] = self._generate_recommendations(clean_metrics, attacked_metrics, robustness_metrics, is_attack_evaluation)
        data['TOTAL_TEST_IMAGES'] = total_images

        return data

    def _add_metrics_to_data(
        self,
        data: Dict[str, Any],
        clean_metrics: Dict[str, float],
        attacked_metrics: Dict[str, float],
        robustness_metrics: Dict[str, float]
    ):
        """메트릭 데이터 추가"""

        # Handle target class scenario: use 'ap' if present, otherwise 'map'
        clean_map = clean_metrics.get('map', clean_metrics.get('ap', 0))
        clean_map50 = clean_metrics.get('map50', clean_metrics.get('ap50', 0))
        clean_map75 = clean_metrics.get('map75', clean_metrics.get('ap75', 0))
        attacked_map = attacked_metrics.get('map', attacked_metrics.get('ap', 0))
        attacked_map50 = attacked_metrics.get('map50', attacked_metrics.get('ap50', 0))
        attacked_map75 = attacked_metrics.get('map75', attacked_metrics.get('ap75', 0))

        # Clean 메트릭
        data['CLEAN_MAP50'] = f"{clean_map50:.4f}"
        data['CLEAN_MAP75'] = f"{clean_map75:.4f}"
        data['CLEAN_MAP'] = f"{clean_map:.4f}"
        data['CLEAN_PRECISION'] = f"{clean_metrics.get('precision', 0):.4f}"
        data['CLEAN_RECALL'] = f"{clean_metrics.get('recall', 0):.4f}"
        data['CLEAN_F1'] = f"{clean_metrics.get('f1', 0):.4f}"
        data['CLEAN_DETECTION_RATE'] = 'N/A'  # Not tracked

        # Attacked 메트릭
        data['ATTACKED_MAP50'] = f"{attacked_map50:.4f}"
        data['ATTACKED_MAP50_NOTE'] = self._get_performance_note(attacked_map50, clean_map50)
        data['ATTACKED_MAP75'] = f"{attacked_map75:.4f}"
        data['ATTACKED_MAP75_NOTE'] = self._get_performance_note(attacked_map75, clean_map75)
        data['ATTACKED_MAP'] = f"{attacked_map:.4f}"
        data['ATTACKED_MAP_NOTE'] = self._get_performance_note(attacked_map, clean_map)
        data['ATTACKED_PRECISION'] = f"{attacked_metrics.get('precision', 0):.4f}"
        data['ATTACKED_PRECISION_NOTE'] = self._get_performance_note(attacked_metrics.get('precision', 0), clean_metrics.get('precision', 0))
        data['ATTACKED_RECALL'] = f"{attacked_metrics.get('recall', 0):.4f}"
        data['ATTACKED_RECALL_NOTE'] = self._get_performance_note(attacked_metrics.get('recall', 0), clean_metrics.get('recall', 0))
        data['ATTACKED_F1'] = f"{attacked_metrics.get('f1', 0):.4f}"
        data['ATTACKED_F1_NOTE'] = self._get_performance_note(attacked_metrics.get('f1', 0), clean_metrics.get('f1', 0))
        data['ATTACKED_DETECTION_RATE'] = 'N/A'  # Not tracked
        data['ATTACKED_DETECTION_RATE_NOTE'] = 'N/A'

        # Delta 계산 (감소율만 표시, 마이너스 제거)
        metric_pairs = [
            ('map50', clean_map50, attacked_map50),
            ('map75', clean_map75, attacked_map75),
            ('map', clean_map, attacked_map),
            ('precision', clean_metrics.get('precision', 0), attacked_metrics.get('precision', 0)),
            ('recall', clean_metrics.get('recall', 0), attacked_metrics.get('recall', 0)),
            ('f1', clean_metrics.get('f1', 0), attacked_metrics.get('f1', 0)),
        ]

        for metric_name, clean_val, attacked_val in metric_pairs:

            if clean_val > 0:
                delta_percent = ((clean_val - attacked_val) / clean_val) * 100
                data[f'DELTA_{metric_name.upper()}'] = f"{delta_percent:.1f}%"
                data[f'DELTA_{metric_name.upper()}_CLASS'] = 'negative' if delta_percent > 0 else 'positive'
            else:
                data[f'DELTA_{metric_name.upper()}'] = 'N/A'
                data[f'DELTA_{metric_name.upper()}_CLASS'] = ''

        # 공격 효과 지표 (Use robustness metrics if available)
        if robustness_metrics:
            delta_map_percent = robustness_metrics.get('drop_percentage', 0)
            robustness_ratio = robustness_metrics.get('robustness_ratio', 1.0)
            delta_map_percent = ((clean_map - attacked_map) / clean_map) * 100
            robustness_ratio = attacked_map / clean_map if clean_map > 0 else 0.0
        else:
            delta_map_percent = 0.0
            robustness_ratio = 0.0

        if clean_map > 0 or robustness_metrics:
            data['DELTA_MAP_PERCENT'] = f"{delta_map_percent:.1f}%"
            data['DELTA_MAP_PERCENT_CLASS'] = 'negative' if delta_map_percent > 0 else 'positive'
            data['DELTA_MAP_PERCENT_EVAL'] = self._get_severity_evaluation(delta_map_percent)

            # ASR (Attack Success Rate) - mAP 50% 이상 감소 시 성공으로 간주
            asr = min(100, delta_map_percent * 2) if delta_map_percent > 0 else 0
            data['ASR_VALUE'] = f"{asr:.1f}%"
            data['ASR_CLASS'] = 'negative' if asr > 50 else 'positive'
            data['ASR_EVAL'] = '✓ 매우 효과적인 공격' if asr > 70 else '⚠️ 중간 수준 공격' if asr > 40 else '✗ 낮은 공격 효과'

            data['ROBUSTNESS_RATIO'] = f"{robustness_ratio:.3f}"
        else:
            data['DELTA_MAP_PERCENT'] = 'N/A'
            data['ASR_VALUE'] = 'N/A'
            data['ROBUSTNESS_RATIO'] = 'N/A'

        data['FNR_INCREASE'] = 'N/A'
        data['FNR_INCREASE_CLASS'] = ''
        data['FNR_INCREASE_EVAL'] = ''
        data['AVG_CONF_DECREASE'] = 'N/A'
        data['AVG_CONF_DECREASE_CLASS'] = ''
        data['AVG_CONF_DECREASE_EVAL'] = ''
        data['DELTA_DETECTION_RATE'] = 'N/A'
        data['DELTA_DETECTION_RATE_CLASS'] = ''

        # 그래프 데이터 (Python dict로 저장, 나중에 차트 생성에 사용)
        data['clean_metrics'] = clean_metrics
        data['attacked_metrics'] = attacked_metrics

    def _add_base_metrics_to_data(
        self,
        data: Dict[str, Any],
        metrics: Dict[str, float]
    ):
        """단일 데이터셋 평가용 메트릭 데이터 추가"""

        # Handle target class scenario: use 'ap' if present, otherwise 'map'
        map_value = metrics.get('map', metrics.get('ap', 0))
        map50_value = metrics.get('map50', metrics.get('ap50', 0))
        map75_value = metrics.get('map75', metrics.get('ap75', 0))

        # 기본 메트릭
        data['CLEAN_MAP50'] = f"{map50_value:.4f}"
        data['CLEAN_MAP75'] = f"{map75_value:.4f}"
        data['CLEAN_MAP'] = f"{map_value:.4f}"
        data['CLEAN_PRECISION'] = f"{metrics.get('precision', 0):.4f}"
        data['CLEAN_RECALL'] = f"{metrics.get('recall', 0):.4f}"
        data['CLEAN_F1'] = f"{metrics.get('f1', 0):.4f}"

        # 성능 등급 평가
        performance_grade, performance_eval = self._evaluate_performance_grade(map_value)
        data['PERFORMANCE_GRADE'] = performance_grade
        data['PERFORMANCE_EVALUATION'] = performance_eval

        # 그래프 데이터
        data['clean_metrics'] = metrics

    @staticmethod
    def _evaluate_performance_grade(map_value: float) -> tuple[str, str]:
        """mAP 값을 기준으로 성능 등급 평가"""
        if map_value >= 0.9:
            return "S등급 (Excellent)", "매우 우수한 성능. 프로덕션 배포 가능한 수준입니다."
        elif map_value >= 0.8:
            return "A등급 (Very Good)", "우수한 성능. 대부분의 실전 환경에서 활용 가능합니다."
        elif map_value >= 0.7:
            return "B등급 (Good)", "양호한 성능. 일부 환경에서 추가 튜닝이 필요할 수 있습니다."
        elif map_value >= 0.6:
            return "C등급 (Fair)", "보통 성능. 개선이 권장됩니다."
        elif map_value >= 0.5:
            return "D등급 (Poor)", "낮은 성능. 모델 재학습 또는 아키텍처 변경이 필요합니다."
        else:
            return "F등급 (Very Poor)", "매우 낮은 성능. 근본적인 개선이 필요합니다."

    async def _generate_charts(
        self,
        db: AsyncSession,
        data: Dict[str, Any],
        clean_metrics: Dict[str, float],
        attacked_metrics: Dict[str, float],
        eval_run: Any,
        is_attack_evaluation: bool = True
    ):
        """차트 생성"""
        if not self._matplotlib_available:
            return

        # Handle target class scenario: use 'ap' if present, otherwise 'map'
        clean_metrics_chart = {
            'map': clean_metrics.get('map', clean_metrics.get('ap', 0)),
            'map50': clean_metrics.get('map50', clean_metrics.get('ap50', 0)),
            'map75': clean_metrics.get('map75', clean_metrics.get('ap75', 0)),
            'precision': clean_metrics.get('precision', 0),
            'recall': clean_metrics.get('recall', 0),
            'f1': clean_metrics.get('f1', 0),
        }
        

        # 1. 막대 그래프
        bar_chart_path = self.temp_charts_dir / f"metrics_bar_{eval_run.id}.png"
        if is_attack_evaluation:
            attacked_metrics_chart = {
                'map': attacked_metrics.get('map', attacked_metrics.get('ap', 0)),
                'map50': attacked_metrics.get('map50', attacked_metrics.get('ap50', 0)),
                'map75': attacked_metrics.get('map75', attacked_metrics.get('ap75', 0)),
                'precision': attacked_metrics.get('precision', 0),
                'recall': attacked_metrics.get('recall', 0),
                'f1': attacked_metrics.get('f1', 0),
            }
            self._generate_bar_chart(clean_metrics_chart, attacked_metrics_chart, str(bar_chart_path))
        else:
            self._generate_single_bar_chart(clean_metrics_chart, str(bar_chart_path))
        data['METRICS_BAR_CHART'] = str(bar_chart_path)

        # 2. PR Curve 생성
        try:
            pr_curve_path = self.temp_charts_dir / f"pr_curve_{eval_run.id}.png"
            await self._generate_pr_curve_chart(db, eval_run, str(pr_curve_path), is_attack_evaluation)
            data['PR_CURVE_CHART'] = str(pr_curve_path)
        except Exception as e:
            logger.warning(f"PR Curve generation failed: {e}")
            data['PR_CURVE_CHART'] = ''

    def _generate_bar_chart(
        self,
        clean_metrics: Dict[str, float],
        attacked_metrics: Dict[str, float],
        output_path: str
    ):
        """막대 그래프 생성"""
        metrics_names = ['F1 Score', 'AP@50', 'AP@75', 'AP@[50:95]', 'Precision', 'Recall']
        metric_keys = ['f1', 'map50', 'map75', 'map', 'precision', 'recall']

        clean_values = [clean_metrics.get(k, 0.0) for k in metric_keys]
        attacked_values = [attacked_metrics.get(k, 0.0) for k in metric_keys]

        x = self._np.arange(len(metrics_names))
        width = 0.35

        _fig, ax = self._plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - width/2, clean_values, width, label='일반 데이터셋', color='#4C4CFF', alpha=0.8)
        bars2 = ax.bar(x + width/2, attacked_values, width, label='적대적 공격 데이터셋', color='#cc0000', alpha=0.8)

        # 값 표시 (폰트 크기 2배)
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=18)

        ax.set_ylabel('Score', fontsize=24)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names, fontsize=20)
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=22)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        self._plt.tight_layout()
        self._plt.savefig(output_path, dpi=300, bbox_inches='tight')
        self._plt.close()

        logger.info(f"막대 그래프 생성: {output_path}")

    def _generate_single_bar_chart(
        self,
        metrics: Dict[str, float],
        output_path: str
    ):
        """단일 데이터셋용 막대 그래프 생성"""
        metrics_names = ['F1 Score', 'AP@50', 'AP@75', 'AP@[50:95]', 'Precision', 'Recall']
        metric_keys = ['f1', 'map50', 'map75', 'map', 'precision', 'recall']

        values = [metrics.get(k, 0.0) for k in metric_keys]

        x = self._np.arange(len(metrics_names))
        width = 0.6

        _fig, ax = self._plt.subplots(figsize=(12, 6))
        bars = ax.bar(x, values, width, label='Performance', color='#4C4CFF', alpha=0.8)

        # 값 표시
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10)

        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names, fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        self._plt.tight_layout()
        self._plt.savefig(output_path, dpi=300, bbox_inches='tight')
        self._plt.close()

        logger.info(f"단일 막대 그래프 생성: {output_path}")

    async def _generate_pr_curve_chart(
        self,
        db: AsyncSession,
        eval_run: Any,
        output_path: str,
        is_attack_evaluation: bool = True
    ):
        """
        PR Curve 생성.

        저장된 pr_curves 데이터를 우선 사용하고, 없으면 eval_items에서 실시간 계산 (하위 호환성).
        """
        if not self._matplotlib_available or not self._np:
            return

        from app.schemas.evaluation import EvalDatasetType

        # Try to load PR curves from eval_dataset_results
        base_result = await crud.evaluation.get_eval_dataset_result_by_run_and_type(
            db, eval_run_id=eval_run.id, dataset_type=EvalDatasetType.BASE
        )

        base_pr_data = None
        attack_pr_data = None

        # Check if stored PR curves exist for BASE dataset
        if base_result and base_result.pr_curves:
            logger.info("Using stored PR curves for BASE dataset")
            # Use IoU 0.5 for visualization (standard)
            base_pr_iou50 = base_result.pr_curves.get('iou_0.50', {})
            if base_pr_iou50.get('precisions') and base_pr_iou50.get('recalls'):
                base_pr_data = {
                    'precisions': self._np.array(base_pr_iou50['precisions']),
                    'recalls': self._np.array(base_pr_iou50['recalls']),
                    'ap': base_pr_iou50.get('ap', 0.0)
                }
        else:
            # Fallback: Calculate from eval_items (existing logic)
            logger.info("No stored PR curves for BASE, calculating from eval_items...")
            base_items = await crud.evaluation.get_all_eval_items(
                db, run_id=eval_run.id, dataset_type=EvalDatasetType.BASE
            )
            base_pr_data = self._calculate_pr_curve_from_items(base_items) if base_items else None

        # For attack evaluation, get ATTACK dataset PR curves
        if is_attack_evaluation:
            attack_result = await crud.evaluation.get_eval_dataset_result_by_run_and_type(
                db, eval_run_id=eval_run.id, dataset_type=EvalDatasetType.ATTACK
            )

            if attack_result and attack_result.pr_curves:
                logger.info("Using stored PR curves for ATTACK dataset")
                attack_pr_iou50 = attack_result.pr_curves.get('iou_0.50', {})
                if attack_pr_iou50.get('precisions') and attack_pr_iou50.get('recalls'):
                    attack_pr_data = {
                        'precisions': self._np.array(attack_pr_iou50['precisions']),
                        'recalls': self._np.array(attack_pr_iou50['recalls']),
                        'ap': attack_pr_iou50.get('ap', 0.0)
                    }
            else:
                # Fallback: Calculate from eval_items
                logger.info("No stored PR curves for ATTACK, calculating from eval_items...")
                attack_items = await crud.evaluation.get_all_eval_items(
                    db, run_id=eval_run.id, dataset_type=EvalDatasetType.ATTACK
                )
                attack_pr_data = self._calculate_pr_curve_from_items(attack_items) if attack_items else None

        # Plot PR curves (existing plotting logic)
        if not base_pr_data:
            logger.warning("No PR curve data available for BASE dataset")
            return

        # Plot PR curves
        _fig, ax = self._plt.subplots(figsize=(10, 8))

        label_clean = "일반 데이터셋" if is_attack_evaluation else "성능"
        ax.plot(base_pr_data['recalls'], base_pr_data['precisions'],
               color='#4C4CFF', linewidth=2, label=f'{label_clean} (AP={base_pr_data["ap"]:.3f})')
        ax.fill_between(base_pr_data['recalls'], base_pr_data['precisions'], alpha=0.1, color='#4C4CFF')

        if attack_pr_data:
            ax.plot(attack_pr_data['recalls'], attack_pr_data['precisions'],
                   color='#cc0000', linewidth=2, label=f'적대적 공격 데이터셋 (AP={attack_pr_data["ap"]:.3f})')
            ax.fill_between(attack_pr_data['recalls'], attack_pr_data['precisions'], alpha=0.1, color='#cc0000')

        ax.set_xlabel('재현율 (Recall)', fontsize=24)
        ax.set_ylabel('정밀도 (Precision)', fontsize=24)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', fontsize=22)
        ax.tick_params(axis='both', which='major', labelsize=20)

        self._plt.tight_layout()
        self._plt.savefig(output_path, dpi=300, bbox_inches='tight')
        self._plt.close()

        logger.info(f"PR Curve 생성: {output_path}")

    def _calculate_pr_curve_from_items(self, eval_items: List[Any]) -> Optional[Dict[str, Any]]:
        """Calculate PR curve from eval_items"""
        if not eval_items:
            return None

        from app.services.metrics_calculator import parse_detection_to_bbox, calculate_iou

        # Collect all predictions and ground truths
        all_predictions = []
        all_ground_truths = []

        for item in eval_items:
            image_id = str(item.image_2d_id or item.image_3d_id)

            # Parse predictions
            for pred in item.prediction or []:
                try:
                    bbox = parse_detection_to_bbox(
                        pred,
                        image_id=image_id,
                        image_width=640,  # Default, actual size doesn't matter for normalized coords
                        image_height=640,
                    )
                    all_predictions.append(bbox)
                except Exception as e:
                    logger.debug(f"Failed to parse prediction: {e}")
                    continue

            # Parse ground truths
            for gt in item.ground_truth or []:
                try:
                    gt_detection = {
                        "bbox": gt.get("bbox", {}),
                        "class_name": gt.get("class_name", "unknown"),
                        "confidence": 1.0,
                    }
                    bbox = parse_detection_to_bbox(
                        gt_detection,
                        image_id=image_id,
                        image_width=640,
                        image_height=640,
                    )
                    all_ground_truths.append(bbox)
                except Exception as e:
                    logger.debug(f"Failed to parse GT: {e}")
                    continue

        if not all_predictions or not all_ground_truths:
            return None

        # Sort predictions by confidence (descending)
        all_predictions = sorted(all_predictions, key=lambda x: x.confidence, reverse=True)

        # Calculate TP/FP at each threshold (each prediction's confidence)
        gt_matched = [False] * len(all_ground_truths)
        tp = self._np.zeros(len(all_predictions))
        fp = self._np.zeros(len(all_predictions))

        iou_threshold = 0.5  # Standard IoU threshold for matching

        for pred_idx, pred in enumerate(all_predictions):
            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(all_ground_truths):
                if gt.class_name != pred.class_name or gt.image_id != pred.image_id:
                    continue
                if gt_matched[gt_idx]:
                    continue

                iou = calculate_iou(pred, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp[pred_idx] = 1
                gt_matched[best_gt_idx] = True
            else:
                fp[pred_idx] = 1

        # Calculate cumulative TP and FP
        tp_cumsum = self._np.cumsum(tp)
        fp_cumsum = self._np.cumsum(fp)

        # Calculate precision and recall at each threshold
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
        recalls = tp_cumsum / len(all_ground_truths)

        # Calculate AP using interpolation
        ap = self._calculate_interpolated_ap(precisions, recalls)

        return {
            'precisions': precisions,
            'recalls': recalls,
            'ap': ap,
        }

    def _calculate_interpolated_ap(self, precisions: Any, recalls: Any) -> float:
        """Calculate AP using all-point interpolation"""
        if len(precisions) == 0 or len(recalls) == 0:
            return 0.0

        # Add sentinel values
        mrec = self._np.concatenate(([0.0], recalls, [1.0]))
        mpre = self._np.concatenate(([0.0], precisions, [0.0]))

        # Compute precision envelope (monotonically decreasing)
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])

        # Find points where recall changes
        i_list = []
        for i in range(1, len(mrec)):
            if mrec[i] != mrec[i - 1]:
                i_list.append(i)

        # Calculate AP as area under curve
        ap = 0.0
        for i in i_list:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]

        return float(ap)

    async def _generate_inference_images(
        self,
        db: AsyncSession,
        data: Dict[str, Any],
        eval_run: Any
    ):
        """추론 결과 이미지 생성 (원본 5개, 공격 5개)"""
        from app.schemas.evaluation import EvalDatasetType
        import base64
        from io import BytesIO
        from PIL import Image, ImageDraw, ImageFont

        try:
            # Get eval items for BASE and ATTACK datasets
            base_items = await crud.evaluation.get_all_eval_items(
                db, run_id=eval_run.id, dataset_type=EvalDatasetType.BASE
            )
            attack_items = await crud.evaluation.get_all_eval_items(
                db, run_id=eval_run.id, dataset_type=EvalDatasetType.ATTACK
            )

            logger.info(f"BASE 이미지 수: {len(base_items) if base_items else 0}")
            logger.info(f"ATTACK 이미지 수: {len(attack_items) if attack_items else 0}")

            # Select top 5 images by attack success rate (highest detection drop)
            top_pairs = self._select_top_attack_success_pairs(base_items, attack_items, top_k=5)

            logger.info(f"선택된 이미지 쌍 수: {len(top_pairs)}")

            # Generate comparison table HTML for paired images
            comparison_table = self._generate_comparison_table_html_from_pairs(top_pairs)

            data['INFERENCE_COMPARISON_TABLE'] = comparison_table

            logger.info(f"비교 테이블 HTML 길이: {len(comparison_table)}")

        except Exception as e:
            logger.error(f"추론 이미지 생성 실패: {e}", exc_info=True)
            data['INFERENCE_COMPARISON_TABLE'] = '<p>이미지 생성 실패</p>'

    def _select_top_attack_success_pairs(self, base_items: List[Any], attack_items: List[Any], top_k: int = 5) -> List[tuple]:
        """
        공격 성공률이 높은 상위 K개의 이미지 쌍을 선택

        공격 성공률 = (원본 탐지 개수 - 공격 탐지 개수) / 원본 탐지 개수
        탐지 개수가 많이 줄어든 순서대로 정렬
        """
        # Build attack map by filename
        attack_map = {}
        for attack_item in attack_items:
            attack_filename = self._get_filename(attack_item)
            if attack_filename:
                original_filename = None
                if attack_filename.startswith('adv_'):
                    original_filename = attack_filename[4:]
                elif attack_filename.startswith('patched_'):
                    original_filename = attack_filename[8:]

                if original_filename:
                    attack_map[original_filename] = attack_item

        # Calculate attack success rate for each pair
        pairs_with_scores = []
        for base_item in base_items:
            base_filename = self._get_filename(base_item)
            if not base_filename:
                continue

            attack_item = attack_map.get(base_filename)
            if not attack_item:
                continue

            # Count detections
            base_detections = len(base_item.prediction) if base_item.prediction else 0
            attack_detections = len(attack_item.prediction) if attack_item.prediction else 0

            # Calculate success rate (higher is better for attack)
            if base_detections > 0:
                success_rate = (base_detections - attack_detections) / base_detections
                detection_drop = base_detections - attack_detections
            else:
                # If no base detections, success rate is 0
                success_rate = 0
                detection_drop = 0

            pairs_with_scores.append({
                'base_item': base_item,
                'attack_item': attack_item,
                'base_filename': base_filename,
                'base_detections': base_detections,
                'attack_detections': attack_detections,
                'success_rate': success_rate,
                'detection_drop': detection_drop
            })

        # Sort by detection drop (absolute number) first, then by success rate
        pairs_with_scores.sort(key=lambda x: (x['detection_drop'], x['success_rate']), reverse=True)

        # Select top K
        top_pairs = pairs_with_scores[:top_k]

        # Log selected pairs
        logger.info(f"=== 공격 성공률 상위 {len(top_pairs)}개 이미지 쌍 ===")
        for i, pair in enumerate(top_pairs):
            logger.info(f"#{i+1}: {pair['base_filename']} - "
                       f"탐지 개수 {pair['base_detections']} -> {pair['attack_detections']} "
                       f"(감소: {pair['detection_drop']}, 성공률: {pair['success_rate']:.1%})")

        return [(p['base_item'], p['attack_item']) for p in top_pairs]

    def _generate_comparison_table_html_from_pairs(self, pairs: List[tuple]) -> str:
        """이미지 쌍 리스트로부터 비교 테이블 HTML 생성"""
        from PIL import Image, ImageDraw, ImageFont
        import base64
        from io import BytesIO

        if not pairs:
            return '<p>비교할 이미지가 없습니다.</p>'

        html_parts = []
        html_parts.append('<table class="inference-comparison-table">')
        html_parts.append('<thead><tr><th>순위</th><th>원본 이미지 (Clean)</th><th>공격 이미지 (Adversarial)</th><th>탐지 변화</th></tr></thead>')
        html_parts.append('<tbody>')

        for idx, (base_item, attack_item) in enumerate(pairs):
            try:
                base_filename = self._get_filename(base_item)

                # Count detections for info
                base_detections = len(base_item.prediction) if base_item.prediction else 0
                attack_detections = len(attack_item.prediction) if attack_item.prediction else 0
                detection_drop = base_detections - attack_detections
                success_rate = (detection_drop / base_detections * 100) if base_detections > 0 else 0

                # Generate base image with bounding boxes
                base_img_html = self._generate_single_inference_image(base_item, idx, "clean")
                # Generate attack image with bounding boxes
                attack_img_html = self._generate_single_inference_image(attack_item, idx, "attack")

                if base_img_html and attack_img_html:
                    # Generate detection change info
                    change_html = f'''
                    <div style="text-align: center; padding: 10px;">
                        <div style="font-size: 14pt; font-weight: bold; margin-bottom: 10px;">
                            {base_detections} → {attack_detections}
                        </div>
                        <div style="font-size: 11pt; color: #cc0000; font-weight: bold;">
                            감소: {detection_drop}개 ({success_rate:.0f}%)
                        </div>
                    </div>
                    '''

                    html_parts.append('<tr>')
                    html_parts.append(f'<td class="image-number">#{idx + 1}</td>')
                    html_parts.append(f'<td class="image-cell">{base_img_html}</td>')
                    html_parts.append(f'<td class="image-cell">{attack_img_html}</td>')
                    html_parts.append(f'<td class="change-cell">{change_html}</td>')
                    html_parts.append('</tr>')
                    logger.info(f"이미지 쌍 {idx + 1} 생성 완료: {base_filename} (탐지: {base_detections} → {attack_detections})")

            except Exception as e:
                logger.error(f"이미지 쌍 {idx + 1} 생성 실패: {e}", exc_info=True)
                continue

        html_parts.append('</tbody></table>')

        logger.info(f"총 {len(pairs)}개의 이미지 쌍 테이블 생성 완료")
        return ''.join(html_parts)

    def _generate_comparison_table_html(self, base_items: List[Any], attack_items: List[Any]) -> str:
        """원본과 공격 이미지 쌍을 테이블 형식으로 비교하는 HTML 생성"""
        from PIL import Image, ImageDraw, ImageFont
        import base64
        from io import BytesIO

        html_parts = []
        html_parts.append('<table class="inference-comparison-table">')
        html_parts.append('<thead><tr><th>이미지 번호</th><th>원본 이미지 (Clean)</th><th>공격 이미지 (Adversarial)</th></tr></thead>')
        html_parts.append('<tbody>')

        # Build a mapping from attack items by their original filename
        # Attack filenames are: adv_<original> or patched_<original>
        attack_map = {}
        for attack_item in attack_items:
            attack_filename = self._get_filename(attack_item)
            if attack_filename:
                # Remove 'adv_' or 'patched_' prefix to get original filename
                original_filename = None
                if attack_filename.startswith('adv_'):
                    original_filename = attack_filename[4:]  # Remove 'adv_'
                elif attack_filename.startswith('patched_'):
                    original_filename = attack_filename[8:]  # Remove 'patched_'

                if original_filename:
                    attack_map[original_filename] = attack_item
                    logger.debug(f"Attack 매핑: {original_filename} -> {attack_filename}")

        logger.info(f"총 {len(attack_map)}개의 공격 이미지 매핑 완료")

        # Match base items with attack items by filename
        pair_count = 0
        for idx, base_item in enumerate(base_items):
            base_filename = self._get_filename(base_item)
            if not base_filename:
                logger.warning(f"Base item {idx}의 파일명을 찾을 수 없음")
                continue

            # Find matching attack item
            attack_item = attack_map.get(base_filename)
            if not attack_item:
                logger.warning(f"원본 이미지 '{base_filename}'에 대응하는 공격 이미지를 찾을 수 없음")
                continue

            try:
                # Generate base image with bounding boxes
                base_img_html = self._generate_single_inference_image(base_item, idx, "clean")
                # Generate attack image with bounding boxes
                attack_img_html = self._generate_single_inference_image(attack_item, idx, "attack")

                if base_img_html and attack_img_html:
                    pair_count += 1
                    html_parts.append('<tr>')
                    html_parts.append(f'<td class="image-number">#{pair_count}</td>')
                    html_parts.append(f'<td class="image-cell">{base_img_html}</td>')
                    html_parts.append(f'<td class="image-cell">{attack_img_html}</td>')
                    html_parts.append('</tr>')
                    logger.info(f"이미지 쌍 {pair_count} 생성 완료: {base_filename}")

            except Exception as e:
                logger.error(f"이미지 쌍 생성 실패 ({base_filename}): {e}", exc_info=True)
                continue

        html_parts.append('</tbody></table>')

        logger.info(f"총 {pair_count}개의 이미지 쌍 테이블 생성 완료")
        return ''.join(html_parts) if pair_count > 0 else '<p>비교할 이미지가 없습니다.</p>'

    def _get_filename(self, item: Any) -> Optional[str]:
        """eval_item으로부터 파일명 추출"""
        try:
            if hasattr(item, 'image_2d') and item.image_2d:
                return item.image_2d.file_name
            elif hasattr(item, 'image_3d') and item.image_3d:
                return item.image_3d.file_name
            else:
                return None
        except Exception as e:
            logger.warning(f"파일명 추출 실패: {e}")
            return None

    def _generate_single_inference_image(self, item: Any, idx: int, dataset_type: str) -> Optional[str]:
        """단일 eval_item으로부터 바운딩 박스가 그려진 이미지 HTML 생성"""
        from PIL import Image, ImageDraw, ImageFont
        import base64
        from io import BytesIO

        try:
            # Get image path
            image_path = None
            if hasattr(item, 'image_2d') and item.image_2d:
                # storage_key may already include the filename, so check both possibilities
                image_path = Path(settings.STORAGE_ROOT) / item.image_2d.storage_key
                if not image_path.exists():
                    # Try appending filename if storage_key is just the directory
                    image_path = Path(settings.STORAGE_ROOT) / item.image_2d.storage_key / item.image_2d.file_name
                logger.debug(f"{dataset_type} Image {idx}: 2D 이미지 경로 = {image_path}")
            elif hasattr(item, 'image_3d') and item.image_3d:
                # storage_key may already include the filename, so check both possibilities
                image_path = Path(settings.STORAGE_ROOT) / item.image_3d.storage_key
                if not image_path.exists():
                    # Try appending filename if storage_key is just the directory
                    image_path = Path(settings.STORAGE_ROOT) / item.image_3d.storage_key / item.image_3d.file_name
                logger.debug(f"{dataset_type} Image {idx}: 3D 이미지 경로 = {image_path}")
            else:
                logger.warning(f"{dataset_type} Image {idx}: image_2d와 image_3d 모두 없음")
                return None

            if not image_path or not image_path.exists():
                logger.warning(f"{dataset_type} Image {idx}: 파일이 존재하지 않음 - {image_path}")
                return None

            # Load image
            img = Image.open(image_path).convert('RGB')
            draw = ImageDraw.Draw(img)
            logger.debug(f"{dataset_type} Image {idx}: 이미지 로드 성공, 크기 = {img.size}")

            # Draw predictions
            pred_count = 0
            if item.prediction is not None and len(item.prediction) > 0:
                logger.debug(f"{dataset_type} Image {idx}: Processing {len(item.prediction)} predictions")

                # Load font once
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
                except Exception as font_error:
                    logger.debug(f"Failed to load TrueType font: {font_error}, using default")
                    font = ImageFont.load_default()

                for pred_idx, pred in enumerate(item.prediction):
                    try:
                        # Handle both dict and object types
                        if isinstance(pred, dict):
                            bbox = pred.get('bbox', {})
                            class_name = pred.get('class_name', 'unknown')
                            conf = pred.get('confidence', 0)
                        else:
                            bbox = getattr(pred, 'bbox', {})
                            class_name = getattr(pred, 'class_name', 'unknown')
                            conf = getattr(pred, 'confidence', 0)

                        if not bbox:
                            logger.debug(f"{dataset_type} Image {idx} Pred {pred_idx}: No bbox found")
                            continue

                        # Convert bbox to dict if it's an object
                        if not isinstance(bbox, dict):
                            if hasattr(bbox, '__dict__'):
                                bbox = bbox.__dict__
                            elif hasattr(bbox, 'model_dump'):
                                bbox = bbox.model_dump()
                            else:
                                logger.warning(f"{dataset_type} Image {idx} Pred {pred_idx}: Cannot convert bbox to dict")
                                continue

                        img_width, img_height = img.size

                        # Determine bbox format and convert to absolute x1, y1, x2, y2
                        if 'x_center' in bbox and 'y_center' in bbox:
                            # YOLO format (normalized x_center, y_center, width, height)
                            x_center = float(bbox.get('x_center', 0))
                            y_center = float(bbox.get('y_center', 0))
                            width = float(bbox.get('width', 0))
                            height = float(bbox.get('height', 0))

                            x1 = int((x_center - width / 2) * img_width)
                            y1 = int((y_center - height / 2) * img_height)
                            x2 = int((x_center + width / 2) * img_width)
                            y2 = int((y_center + height / 2) * img_height)

                        elif 'x1' in bbox and 'y1' in bbox and 'x2' in bbox and 'y2' in bbox:
                            # BBox format (x1, y1, x2, y2) - may be normalized or absolute
                            x1_val = float(bbox.get('x1', 0))
                            y1_val = float(bbox.get('y1', 0))
                            x2_val = float(bbox.get('x2', 0))
                            y2_val = float(bbox.get('y2', 0))

                            # Check if normalized (values between 0 and 1)
                            if 0 <= x1_val <= 1 and 0 <= y1_val <= 1 and 0 <= x2_val <= 1 and 0 <= y2_val <= 1:
                                x1 = int(x1_val * img_width)
                                y1 = int(y1_val * img_height)
                                x2 = int(x2_val * img_width)
                                y2 = int(y2_val * img_height)
                            else:
                                # Already absolute coordinates
                                x1 = int(x1_val)
                                y1 = int(y1_val)
                                x2 = int(x2_val)
                                y2 = int(y2_val)
                        else:
                            logger.warning(f"{dataset_type} Image {idx} Pred {pred_idx}: Unknown bbox format: {bbox.keys()}")
                            continue

                        # Ensure coordinates are within image bounds
                        x1 = max(0, min(x1, img_width - 1))
                        y1 = max(0, min(y1, img_height - 1))
                        x2 = max(0, min(x2, img_width - 1))
                        y2 = max(0, min(y2, img_height - 1))

                        # Skip invalid boxes
                        if x2 <= x1 or y2 <= y1:
                            logger.debug(f"{dataset_type} Image {idx} Pred {pred_idx}: Invalid box dimensions")
                            continue

                        # Draw rectangle
                        draw.rectangle([x1, y1, x2, y2], outline='red', width=3)

                        # Add label with background
                        label = f"{class_name} {conf:.2f}"

                        # Calculate label position (above the box if possible)
                        label_y = y1 - 20 if y1 >= 20 else y1 + 5

                        # Get text bounding box
                        try:
                            bbox_label = draw.textbbox((x1, label_y), label, font=font)
                            draw.rectangle(bbox_label, fill='red')
                            draw.text((x1, label_y), label, fill='white', font=font)
                        except Exception as text_error:
                            # Fallback: just draw text without background
                            logger.debug(f"Text rendering error: {text_error}")
                            draw.text((x1, label_y), label, fill='red', font=font)

                        pred_count += 1

                    except Exception as pred_error:
                        logger.warning(f"{dataset_type} Image {idx} Pred {pred_idx}: Error processing prediction: {pred_error}")
                        continue
            else:
                logger.debug(f"{dataset_type} Image {idx}: No predictions to draw")

            logger.debug(f"{dataset_type} Image {idx}: {pred_count}개의 바운딩 박스 그림")

            # Convert to base64
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            img_str = base64.b64encode(buffered.getvalue()).decode()

            # Create HTML with detection count
            html = f'''
            <div class="inference-image-container">
                <img src="data:image/jpeg;base64,{img_str}" alt="{dataset_type} Image {idx + 1}">
                <div class="detection-count">탐지 객체: {pred_count}개</div>
            </div>
            '''
            logger.debug(f"{dataset_type} Image {idx}: HTML 생성 완료")
            return html

        except Exception as e:
            logger.error(f"{dataset_type} Image {idx} 생성 실패: {e}", exc_info=True)
            return None

    def _generate_inference_html(self, eval_items: List[Any], dataset_type: str) -> str:
        """eval_items로부터 추론 결과 HTML 생성"""
        from PIL import Image, ImageDraw
        import base64
        from io import BytesIO

        html_parts = []

        logger.info(f"{dataset_type}: 처리할 eval_items 수: {len(eval_items)}")

        for idx, item in enumerate(eval_items):
            try:
                # Get image path
                image_path = None
                if hasattr(item, 'image_2d') and item.image_2d:
                    # storage_key may already include the filename, so check both possibilities
                    image_path = Path(settings.STORAGE_ROOT) / item.image_2d.storage_key
                    if not image_path.exists():
                        # Try appending filename if storage_key is just the directory
                        image_path = Path(settings.STORAGE_ROOT) / item.image_2d.storage_key / item.image_2d.file_name
                    logger.info(f"{dataset_type} Image {idx}: 2D 이미지 경로 = {image_path}")
                elif hasattr(item, 'image_3d') and item.image_3d:
                    # storage_key may already include the filename, so check both possibilities
                    image_path = Path(settings.STORAGE_ROOT) / item.image_3d.storage_key
                    if not image_path.exists():
                        # Try appending filename if storage_key is just the directory
                        image_path = Path(settings.STORAGE_ROOT) / item.image_3d.storage_key / item.image_3d.file_name
                    logger.info(f"{dataset_type} Image {idx}: 3D 이미지 경로 = {image_path}")
                else:
                    logger.warning(f"{dataset_type} Image {idx}: image_2d와 image_3d 모두 없음")

                if not image_path:
                    logger.warning(f"{dataset_type} Image {idx}: image_path가 None")
                    continue

                if not image_path.exists():
                    logger.warning(f"{dataset_type} Image {idx}: 파일이 존재하지 않음 - {image_path}")
                    continue

                # Load image
                img = Image.open(image_path).convert('RGB')
                draw = ImageDraw.Draw(img)
                logger.info(f"{dataset_type} Image {idx}: 이미지 로드 성공, 크기 = {img.size}")

                # Draw predictions
                pred_count = 0
                if item.prediction:
                    for pred in item.prediction:
                        bbox = pred.get('bbox', {})
                        if bbox:
                            # Parse bbox (assuming normalized coordinates)
                            x_center = bbox.get('x_center', 0)
                            y_center = bbox.get('y_center', 0)
                            width = bbox.get('width', 0)
                            height = bbox.get('height', 0)

                            # Convert to absolute coordinates
                            img_width, img_height = img.size
                            x1 = int((x_center - width / 2) * img_width)
                            y1 = int((y_center - height / 2) * img_height)
                            x2 = int((x_center + width / 2) * img_width)
                            y2 = int((y_center + height / 2) * img_height)

                            # Draw rectangle
                            draw.rectangle([x1, y1, x2, y2], outline='red', width=2)

                            # Add label
                            class_name = pred.get('class_name', '')
                            conf = pred.get('confidence', 0)
                            label = f"{class_name} {conf:.2f}"
                            draw.text((x1, y1 - 10), label, fill='red')
                            pred_count += 1

                logger.info(f"{dataset_type} Image {idx}: {pred_count}개의 바운딩 박스 그림")

                # Convert to base64
                buffered = BytesIO()
                img.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode()

                # Create HTML
                html_parts.append(f'''
                <div class="inference-item">
                    <img src="data:image/jpeg;base64,{img_str}" alt="Image {idx + 1}">
                    <div class="caption">Image {idx + 1}</div>
                </div>
                ''')
                logger.info(f"{dataset_type} Image {idx}: HTML 생성 완료")

            except Exception as e:
                logger.error(f"{dataset_type} Image {idx} 생성 실패: {e}", exc_info=True)
                continue

        logger.info(f"{dataset_type}: 총 {len(html_parts)}개의 이미지 HTML 생성됨")
        return ''.join(html_parts) if html_parts else '<p>이미지 없음</p>'

    def _fill_template(self, data: Dict[str, Any], template_path: Path = None) -> str:
        """템플릿 치환"""
        if template_path is None:
            template_path = self.adversarial_template_path
        with open(template_path, 'r', encoding='utf-8') as f:
            template = f.read()

        # 플레이스홀더 치환
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                # dict/list는 치환하지 않음 (그래프 데이터용)
                continue
            placeholder = f"{{{{{key}}}}}"
            template = template.replace(placeholder, str(value))

        # 치환되지 않은 플레이스홀더를 빈 값으로 처리
        import re
        remaining_placeholders = re.findall(r'\{\{([A-Z_0-9]+)\}\}', template)
        for ph in remaining_placeholders:
            template = template.replace(f"{{{{{ph}}}}}", "")

        return template

    def _cleanup_temp_files(self):
        """임시 파일 정리"""
        try:
            import shutil
            if self.temp_charts_dir.exists():
                shutil.rmtree(self.temp_charts_dir)
                self.temp_charts_dir.mkdir(exist_ok=True)
                logger.info(f"임시 파일 정리 완료: {self.temp_charts_dir}")
        except Exception as e:
            logger.warning(f"임시 파일 정리 실패: {e}")

    @staticmethod
    def _get_attack_type_display(attack_method: str) -> str:
        """공격 유형 표시명"""
        mapping = {
            'fgsm': 'FGSM (Fast Gradient Sign Method)',
            'pgd': 'PGD (Projected Gradient Descent)',
            'patch': 'Adversarial Patch',
            'dpatch': 'D-Patch',
            'robust_dpatch': 'Robust D-Patch',
            'naturalistic': 'NAP (Naturalistic Adversarial Patch)',
            'universal_noise': 'Universal Noise Attack',
            'noise_osfd': 'Noise OSFD',
        }
        return mapping.get(attack_method.lower(), attack_method)

    @staticmethod
    def _get_attack_description(attack_method: str) -> str:
        """공격 설명"""
        descriptions = {
            'pgd': 'PGD는 다중 스텝 반복 최적화를 통해 적대적 perturbation을 생성하는 강력한 공격 기법입니다.',
            'fgsm': 'FGSM은 단일 스텝 gradient 기반으로 빠르게 적대적 샘플을 생성하는 기법입니다.',
            # ... 다른 공격 설명 추가
        }
        return descriptions.get(attack_method.lower(), f'{attack_method} 공격 기법')

    @staticmethod
    def _format_hyperparameters(attack_method: str, attack_params: Dict) -> str:
        """공격 기법별 하이퍼파라미터 포맷팅"""
        if not attack_params:
            return 'N/A'

        method = attack_method.lower()
        params_list = []

        # Define important parameters for each attack method
        if method in ['pgd', 'fgsm']:
            # Gradient-based attacks
            if 'eps' in attack_params:
                params_list.append(f"Epsilon: {attack_params['eps']}")
            if 'eps_step' in attack_params:
                params_list.append(f"Step Size: {attack_params['eps_step']}")
            if 'max_iter' in attack_params:
                params_list.append(f"Iterations: {attack_params['max_iter']}")
            if 'norm' in attack_params:
                params_list.append(f"Norm: L{attack_params['norm']}")

        elif method in ['patch', 'dpatch', 'robust_dpatch']:
            # Patch-based attacks
            if 'patch_shape' in attack_params:
                shape = attack_params['patch_shape']
                if isinstance(shape, (list, tuple)) and len(shape) >= 2:
                    params_list.append(f"Patch Size: {shape[0]}×{shape[1]}")
            if 'learning_rate' in attack_params:
                params_list.append(f"Learning Rate: {attack_params['learning_rate']}")
            if 'max_iter' in attack_params:
                params_list.append(f"Iterations: {attack_params['max_iter']}")
            if method == 'robust_dpatch':
                if 'crop_range' in attack_params:
                    params_list.append(f"Crop Range: {attack_params['crop_range']}")
                if 'brightness_range' in attack_params:
                    params_list.append(f"Brightness Range: {attack_params['brightness_range']}")

        elif method == 'universal_noise':
            # Universal noise attack
            if 'eps' in attack_params:
                params_list.append(f"Epsilon: {attack_params['eps']}")
            if 'eps_step' in attack_params:
                params_list.append(f"Step Size: {attack_params['eps_step']}")
            if 'max_iter' in attack_params:
                params_list.append(f"Iterations: {attack_params['max_iter']}")

        elif method == 'noise_osfd':
            # Noise OSFD attack
            if 'eps' in attack_params:
                params_list.append(f"Epsilon: {attack_params['eps']}")
            if 'eps_step' in attack_params:
                params_list.append(f"Step Size: {attack_params['eps_step']}")
            if 'max_iter' in attack_params:
                params_list.append(f"Iterations: {attack_params['max_iter']}")
            if 'amplification_factor' in attack_params:
                params_list.append(f"Amplification Factor: {attack_params['amplification_factor']}")

        elif method == 'naturalistic':
            # Naturalistic (NAP) attack
            if 'learning_rate' in attack_params:
                params_list.append(f"Learning Rate: {attack_params['learning_rate']}")
            if 'max_iter' in attack_params:
                params_list.append(f"Iterations: {attack_params['max_iter']}")
            if 'patch_scale' in attack_params:
                params_list.append(f"Patch Scale: {attack_params['patch_scale']}")
            if 'gan_class_id' in attack_params:
                params_list.append(f"GAN Class ID: {attack_params['gan_class_id']}")

        # If no specific params found, show general params (excluding meta params)
        if not params_list:
            exclude_keys = {'attack_method', 'device', 'batch_size', 'target_class',
                          'summary_writer', 'verbose', 'estimator', 'return_perturbation'}
            for key, value in attack_params.items():
                if key not in exclude_keys and not key.startswith('_'):
                    params_list.append(f"{key}: {value}")

        return ', '.join(params_list) if params_list else 'N/A'

    @staticmethod
    def _get_performance_note(attacked_value: float, clean_value: float) -> str:
        """성능 변화 노트"""
        if clean_value == 0:
            return 'N/A'

        delta_percent = ((attacked_value - clean_value) / clean_value) * 100

        if delta_percent < -70:
            return '치명적 저하'
        elif delta_percent < -50:
            return '심각한 저하'
        elif delta_percent < -30:
            return '대폭 감소'
        elif delta_percent < -10:
            return '중간 감소'
        else:
            return '약간 감소'

    @staticmethod
    def _get_severity_evaluation(delta_percent: float) -> str:
        """심각도 평가"""
        if delta_percent > 70:
            return '⚠️ 치명적 성능 저하'
        elif delta_percent > 50:
            return '⚠️ 심각한 성능 저하'
        elif delta_percent > 30:
            return '⚠️ 대폭 성능 저하'
        else:
            return '△ 중간 성능 저하'

    @staticmethod
    def _format_duration(started_at, ended_at) -> str:
        """Format duration between two datetime objects"""
        if not started_at or not ended_at:
            return 'N/A'

        duration = ended_at - started_at
        total_seconds = int(duration.total_seconds())

        if total_seconds < 60:
            return f"{total_seconds}초"
        elif total_seconds < 3600:
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            return f"{minutes}분 {seconds}초"
        else:
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            return f"{hours}시간 {minutes}분"

    @staticmethod
    def _calculate_avg_time_per_image(started_at, ended_at, total_images) -> str:
        """Calculate average time per image"""
        if not started_at or not ended_at or not total_images or total_images == 'N/A':
            return 'N/A'

        try:
            num_images = int(total_images)
            if num_images == 0:
                return 'N/A'

            duration = ended_at - started_at
            total_seconds = duration.total_seconds()
            avg_seconds = total_seconds / num_images

            if avg_seconds < 1:
                return f"{avg_seconds * 1000:.1f}ms"
            else:
                return f"{avg_seconds:.2f}초"
        except (ValueError, TypeError):
            return 'N/A'

    @staticmethod
    def _generate_findings(clean_metrics: Dict, attacked_metrics: Dict, robustness_metrics: Dict = None, is_attack_evaluation: bool = True) -> str:
        """주요 발견사항 생성"""
        clean_map = clean_metrics.get('map', clean_metrics.get('ap', 0))

        if not is_attack_evaluation:
            # Base evaluation findings
            if clean_map >= 0.8:
                performance = "우수한"
            elif clean_map >= 0.6:
                performance = "양호한"
            else:
                performance = "개선이 필요한"

            return f"""1. 모델 성능: 전체 mAP {clean_map:.3f}로 {performance} 수준의 객체 탐지 성능을 보였습니다.

2. 정밀도와 재현율: Precision {clean_metrics.get('precision', 0):.3f}, Recall {clean_metrics.get('recall', 0):.3f}로 균형잡힌 탐지 성능을 나타냅니다.

3. 실전 적용 가능성: 현재 성능 수준에서 {"실전 배포가 가능" if clean_map >= 0.7 else "추가 개선이 필요"}합니다."""

        # Attack evaluation findings
        attacked_map = attacked_metrics.get('map', attacked_metrics.get('ap', 0))

        if clean_map > 0:
            drop_pct = ((clean_map - attacked_map) / clean_map) * 100
            severity = "심각한" if drop_pct > 50 else "중간 수준의" if drop_pct > 30 else "경미한"
        else:
            drop_pct = 0
            severity = "알 수 없는"

        return f"""1. 공격 효과: 적대적 공격이 모델 성능에 {severity} 영향을 미쳤습니다 (mAP 감소율: {drop_pct:.1f}%).

2. 탐지 성능 저하: 원본 데이터 대비 공격 데이터에서 객체 탐지 정확도가 감소했습니다.

3. 취약점 식별: 해당 모델은 적대적 공격에 대한 방어 메커니즘이 부족한 것으로 확인되었습니다."""

    @staticmethod
    def _generate_recommendations(clean_metrics: Dict, attacked_metrics: Dict, robustness_metrics: Dict = None, is_attack_evaluation: bool = True) -> str:
        """권장사항 생성"""
        clean_map = clean_metrics.get('map', clean_metrics.get('ap', 0))

        if not is_attack_evaluation:
            # Base evaluation recommendations
            recommendations = []

            if clean_map < 0.7:
                recommendations.append("1. 모델 개선: mAP 0.7 이상을 목표로 모델 재학습 또는 하이퍼파라미터 튜닝이 필요합니다.")
            else:
                recommendations.append("1. 성능 유지: 현재 우수한 성능을 유지하기 위해 정기적인 모니터링이 필요합니다.")

            recommendations.append("2. 데이터 증강: 다양한 환경과 조건에서의 데이터를 추가하여 일반화 성능을 향상시키세요.")
            recommendations.append("3. 적대적 강건성 평가: 적대적 공격에 대한 취약성을 평가하여 모델의 안정성을 확인하세요.")
            recommendations.append("4. 지속적 개선: 새로운 데이터와 시나리오에 대한 정기적인 평가 및 업데이트가 필요합니다.")

            return "\n\n".join(recommendations)

        # Attack evaluation recommendations
        attacked_map = attacked_metrics.get('map', attacked_metrics.get('ap', 0))

        if clean_map > 0:
            drop_pct = ((clean_map - attacked_map) / clean_map) * 100
        else:
            drop_pct = 0

        recommendations = []
        recommendations.append("1. Adversarial Training: 적대적 샘플을 학습 데이터에 포함시켜 모델 강건성을 향상시켜야 합니다.")

        if drop_pct > 50:
            recommendations.append("2. 모델 재설계: 성능 저하가 심각하므로 모델 아키텍처를 재검토하고 방어 메커니즘을 강화해야 합니다.")
        else:
            recommendations.append("2. Input Preprocessing: 전처리 기법을 적용하여 perturbation을 완화할 수 있습니다.")

        recommendations.append("3. 앙상블 모델: 여러 모델을 결합하여 공격 효과를 감소시킬 수 있습니다.")
        recommendations.append("4. 정기적 평가: 새로운 공격 기법에 대한 정기적인 취약성 평가가 필요합니다.")

        return "\n\n".join(recommendations)


# Global instance
report_generation_service = ReportGenerationService()
