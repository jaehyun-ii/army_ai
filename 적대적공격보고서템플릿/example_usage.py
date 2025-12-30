#!/usr/bin/env python3
"""
보고서 생성 사용 예시

이 스크립트는 generate_report.py의 사용 방법을 보여줍니다.
"""

from generate_report import AdversarialReportGenerator
import json


def example_basic_usage():
    """기본 사용 예시"""
    print("=" * 60)
    print("예시 1: 기본 사용 방법")
    print("=" * 60)

    # 1. 보고서 생성기 초기화
    generator = AdversarialReportGenerator()

    # 2. 데이터 로드
    with open('example_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 3. 보고서 생성
    output_path = generator.generate_report(
        data=data,
        output_pdf='example_report.pdf',
        generate_charts=True
    )

    print(f"\n생성된 보고서: {output_path}")

    # 4. 임시 파일 정리
    generator.cleanup()


def example_custom_charts():
    """커스텀 그래프 사용 예시"""
    print("\n" + "=" * 60)
    print("예시 2: 커스텀 그래프 사용")
    print("=" * 60)

    generator = AdversarialReportGenerator()

    # 데이터 준비
    data = {
        'EXPERIMENT_DATE': '2025-12-29',
        'EXPERIMENTER_NAME': 'AI Security Team',
        'MODEL_NAME': 'YOLOv11n',
        'ATTACK_TYPE': 'FGSM',
        # ... 기타 필수 필드

        # 커스텀 그래프 경로 직접 지정
        'METRICS_BAR_CHART': '/path/to/custom/bar_chart.png',
        'PR_CURVE_CHART': '/path/to/custom/pr_curve.png',
        'PER_CLASS_AP_CHART': '/path/to/custom/class_chart.png',
        'CONFIDENCE_DISTRIBUTION_CHART': '/path/to/custom/conf_dist.png',
    }

    # 그래프 자동 생성 비활성화
    generator.generate_report(
        data=data,
        output_pdf='custom_charts_report.pdf',
        generate_charts=False
    )

    generator.cleanup()


def example_programmatic_generation():
    """프로그래밍 방식으로 보고서 생성"""
    print("\n" + "=" * 60)
    print("예시 3: 평가 결과로부터 보고서 생성")
    print("=" * 60)

    from datetime import datetime

    # 평가 결과 (실제로는 evaluation_service에서 가져옴)
    evaluation_result = {
        'model_id': 'yolov8n_coco',
        'dataset_id': 'coco_val2017',
        'attack_method': 'pgd',

        'clean_metrics': {
            'map50': 0.684,
            'map75': 0.531,
            'map': 0.452,
            'precision': 0.723,
            'recall': 0.691,
            'f1': 0.707
        },

        'attacked_metrics': {
            'map50': 0.216,
            'map75': 0.152,
            'map': 0.123,
            'precision': 0.346,
            'recall': 0.219,
            'f1': 0.273
        },

        # ... 기타 데이터
    }

    # 보고서 데이터 구성
    data = {
        'EXPERIMENT_DATE': datetime.now().strftime('%Y-%m-%d'),
        'EXPERIMENTER_NAME': 'Automated System',
        'SYSTEM_NAME': 'Army AI Defense System',

        # 모델 정보
        'MODEL_NAME': 'YOLOv8n',
        'MODEL_ARCHITECTURE': 'YOLOv8',
        'INPUT_RESOLUTION': '640×640',

        # 성능 지표
        'CLEAN_MAP50': f"{evaluation_result['clean_metrics']['map50']:.4f}",
        'ATTACKED_MAP50': f"{evaluation_result['attacked_metrics']['map50']:.4f}",

        # 그래프 데이터
        'clean_metrics': evaluation_result['clean_metrics'],
        'attacked_metrics': evaluation_result['attacked_metrics'],

        # ... 나머지 필드
    }

    # 보고서 생성
    generator = AdversarialReportGenerator()
    generator.generate_report(
        data=data,
        output_pdf='automated_report.pdf',
        generate_charts=True
    )
    generator.cleanup()


def example_evaluation_integration():
    """평가 서비스와 통합 예시"""
    print("\n" + "=" * 60)
    print("예시 4: 평가 서비스 통합 (의사 코드)")
    print("=" * 60)

    print("""
    # backend/app/services/evaluation_service.py 에 추가:

    async def generate_evaluation_report(
        self,
        evaluation_id: UUID,
        output_path: str = None
    ) -> str:
        \"\"\"평가 결과로부터 PDF 보고서 생성\"\"\"

        # 1. 평가 결과 가져오기
        evaluation = await crud.evaluation.get(db, id=evaluation_id)

        # 2. 보고서 데이터 구성
        report_data = {
            'EXPERIMENT_DATE': evaluation.created_at.strftime('%Y-%m-%d'),
            'MODEL_NAME': evaluation.model.name,
            'DATASET_NAME': evaluation.dataset.name,
            'ATTACK_TYPE': evaluation.attack_method,

            # 성능 지표
            'clean_metrics': evaluation.clean_metrics,
            'attacked_metrics': evaluation.attacked_metrics,

            # PR Curve 데이터
            'clean_pr_data': evaluation.clean_pr_data,
            'attacked_pr_data': evaluation.attacked_pr_data,

            # ... 기타 필드
        }

        # 3. 보고서 생성
        from generate_report import AdversarialReportGenerator
        generator = AdversarialReportGenerator()

        if output_path is None:
            output_path = f'reports/evaluation_{evaluation_id}.pdf'

        pdf_path = generator.generate_report(
            data=report_data,
            output_pdf=output_path,
            generate_charts=True
        )

        generator.cleanup()
        return pdf_path
    """)


if __name__ == '__main__':
    # 기본 예시 실행
    example_basic_usage()

    # 다른 예시들은 주석 해제하여 실행
    # example_custom_charts()
    # example_programmatic_generation()
    # example_evaluation_integration()
