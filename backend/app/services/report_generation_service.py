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
        self.storage_root = Path(settings.STORAGE_ROOT)
        self.reports_dir = self.storage_root / "reports"
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
            self._HTML = HTML
            self._CSS = CSS
            self._weasyprint_available = True
            logger.info("WeasyPrint 사용 가능")
        except ImportError:
            logger.warning("WeasyPrint를 사용할 수 없습니다. PDF 생성이 불가능합니다.")

        try:
            import matplotlib
            matplotlib.use('Agg')  # GUI 없이 사용
            import matplotlib.pyplot as plt
            import numpy as np
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

        if dimension == "3d":
            if eval_run.base_dataset_3d_id:
                from sqlalchemy import select
                from app.models.dataset_3d import Dataset3D
                dataset_row = await db.execute(
                    select(Dataset3D).where(Dataset3D.id == eval_run.base_dataset_3d_id, Dataset3D.deleted_at.is_(None))
                )
                dataset = dataset_row.scalar_one_or_none()
        else:
            if eval_run.base_dataset_id:
                dataset = await crud.dataset_2d.get(db, id=eval_run.base_dataset_id)


        # Determine evaluation type (attack vs base)
        is_attack_evaluation = bool(
            eval_run.attack_dataset_id or getattr(eval_run, "attack_dataset_3d_id", None)
        )

        # 3. 보고서 데이터 구성
        report_data = await self._prepare_report_data(
            db, eval_run, model, dataset, include_charts, dimension, is_attack_evaluation
        )

        # 4. HTML 템플릿 로드 및 치환 (평가 유형에 따라 템플릿 선택)
        template_path = self.adversarial_template_path if is_attack_evaluation else self.base_template_path
        html_content = self._fill_template(report_data, template_path)

        # 5. PDF 생성
        output_filename = f"evaluation_{evaluation_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        output_path = self.reports_dir / output_filename

        self._HTML(string=html_content, base_url=str(self.template_dir)).write_pdf(
            str(output_path),
            stylesheets=[self._CSS(filename=str(self.style_path))]
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
        is_attack_evaluation: bool = True
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

        # Extract attack method from params
        attack_method = 'N/A'
        attack_params = {}
        if eval_run.params:
            attack_method = eval_run.params.get('attack_method', 'N/A')
            attack_params = eval_run.params.get('attack_params', {})

        # Extract model info
        input_shape = model.input_spec.get('shape', [640, 640]) if model and model.input_spec else [640, 640]
        input_resolution = f"{input_shape[0]}×{input_shape[1]}" if len(input_shape) >= 2 else "640×640"

        # Extract dataset info
        dataset_name = dataset.name if dataset else 'N/A'
        total_images = 'N/A'
        if dataset:
            if hasattr(dataset, 'image_count') and dataset.image_count:
                total_images = str(dataset.image_count)
            elif hasattr(dataset, 'metadata_') and dataset.metadata_ and 'image_count' in dataset.metadata_:
                total_images = str(dataset.metadata_['image_count'])

        # 기본 정보
        data = {
            # 표지
            'EXPERIMENT_DATE': eval_run.created_at.strftime('%Y-%m-%d') if eval_run.created_at else datetime.now().strftime('%Y-%m-%d'),
            'EXPERIMENTER_NAME': 'AI Security Team',
            'SYSTEM_NAME': 'Army AI Defense System',
            'REPORT_VERSION': 'v1.0',
            'REPORT_GENERATION_DATE': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),

            # 모델 정보
            'MODEL_NAME': model.name if model else 'Unknown Model',
            'MODEL_ARCHITECTURE': model.architecture if model and hasattr(model, 'architecture') else 'N/A',
            'MODEL_SIZE': f"{model.file_size_mb:.1f}MB" if model and hasattr(model, 'file_size_mb') and model.file_size_mb else 'N/A',
            'INPUT_RESOLUTION': input_resolution,
            'NUM_CLASSES': str(len(model.labelmap)) if model and model.labelmap else 'N/A',
            'CLASS_NAMES': ', '.join(list(model.labelmap.values())[:10]) + '...' if model and model.labelmap and len(model.labelmap) > 0 else 'N/A',
            'INFERENCE_DEVICE': 'GPU (CUDA)' if model and getattr(model, 'device_type', None) == 'cuda' else 'CPU',
            'CONFIDENCE_THRESHOLD': '0.25',  # Default value

            # 데이터셋 정보
            'DATASET_NAME': dataset_name,
            'DATASET_DIMENSION': '3D' if dimension == '3d' else '2D',
            'TOTAL_IMAGES': total_images,
            'TARGET_CLASS': 'All Classes',  # Target class filtering is done at runtime, not stored
            'ANNOTATION_FORMAT': 'YOLO',

            # 공격 정보
            'ATTACK_TYPE': self._get_attack_type_display(attack_method),
            'ATTACK_DESCRIPTION': self._get_attack_description(attack_method),
            'ATTACK_INTENSITY': 'Medium',

            # 실험 환경
            'GPU_INFO': 'N/A',
            'BATCH_SIZE': 'N/A',
            'TOTAL_TIME': self._format_duration(eval_run.started_at, eval_run.ended_at) if eval_run.started_at and eval_run.ended_at else 'N/A',
            'AVG_TIME_PER_IMAGE': 'N/A',
        }

        # 성능 지표
        if is_attack_evaluation:
            self._add_metrics_to_data(data, clean_metrics, attacked_metrics, robustness_metrics)
        else:
            self._add_base_metrics_to_data(data, clean_metrics)

        # 그래프 생성
        if include_charts and self._matplotlib_available:
            await self._generate_charts(db, data, clean_metrics, attacked_metrics, eval_run, is_attack_evaluation)

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

        # Delta 계산
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
                delta_percent = ((attacked_val - clean_val) / clean_val) * 100
                data[f'DELTA_{metric_name.upper()}'] = f"{delta_percent:+.1f}%"
                data[f'DELTA_{metric_name.upper()}_CLASS'] = 'negative' if delta_percent < 0 else 'positive'
            else:
                data[f'DELTA_{metric_name.upper()}'] = 'N/A'
                data[f'DELTA_{metric_name.upper()}_CLASS'] = ''

        # 공격 효과 지표 (Use robustness metrics if available)
        if robustness_metrics:
            delta_map_percent = robustness_metrics.get('drop_percentage', 0)
            robustness_ratio = robustness_metrics.get('robustness_ratio', 1.0)
            delta_map_percent = ((attacked_map - clean_map) / clean_map) * 100
            robustness_ratio = attacked_map / clean_map if clean_map > 0 else 0.0
        else:
            delta_map_percent = 0.0
            robustness_ratio = 0.0

        if clean_map > 0 or robustness_metrics:
            data['DELTA_MAP_PERCENT'] = f"{delta_map_percent:.1f}%"
            data['DELTA_MAP_PERCENT_CLASS'] = 'negative' if delta_map_percent < 0 else 'positive'
            data['DELTA_MAP_PERCENT_EVAL'] = self._get_severity_evaluation(abs(delta_map_percent))

            # ASR (Attack Success Rate) - mAP 50% 이상 감소 시 성공으로 간주
            asr = min(100, abs(delta_map_percent) * 2) if delta_map_percent < 0 else 0
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
        bars1 = ax.bar(x - width/2, clean_values, width, label='Clean', color='#4C4CFF', alpha=0.8)
        bars2 = ax.bar(x + width/2, attacked_values, width, label='Attacked', color='#cc0000', alpha=0.8)

        # 값 표시
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)

        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Clean vs Attacked Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names, fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=11)
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

        label_clean = "Clean" if is_attack_evaluation else "Performance"
        ax.plot(base_pr_data['recalls'], base_pr_data['precisions'],
               color='#4C4CFF', linewidth=2, label=f'{label_clean} (AP={base_pr_data["ap"]:.3f})')
        ax.fill_between(base_pr_data['recalls'], base_pr_data['precisions'], alpha=0.1, color='#4C4CFF')

        if attack_pr_data:
            ax.plot(attack_pr_data['recalls'], attack_pr_data['precisions'],
                   color='#cc0000', linewidth=2, label=f'Attacked (AP={attack_pr_data["ap"]:.3f})')
            ax.fill_between(attack_pr_data['recalls'], attack_pr_data['precisions'], alpha=0.1, color='#cc0000')

        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        title = 'Precision-Recall Curve Comparison (IoU=0.5)' if is_attack_evaluation else 'Precision-Recall Curve (IoU=0.5)'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', fontsize=11)

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
