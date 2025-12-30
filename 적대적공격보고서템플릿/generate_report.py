#!/usr/bin/env python3
"""
적대적 공격 실험 보고서 생성기

WeasyPrint를 사용하여 HTML 템플릿을 PDF 보고서로 변환합니다.
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # GUI 없이 그래프 생성
import numpy as np
from weasyprint import HTML, CSS
import shutil


class AdversarialReportGenerator:
    """적대적 공격 보고서 생성 클래스"""

    def __init__(self, template_dir: str = None):
        """
        Args:
            template_dir: 템플릿 디렉토리 경로 (기본값: 현재 스크립트 디렉토리)
        """
        if template_dir is None:
            self.template_dir = Path(__file__).parent
        else:
            self.template_dir = Path(template_dir)

        self.template_path = self.template_dir / "report_template.html"
        self.style_path = self.template_dir / "style.css"

        # 임시 그래프 저장 디렉토리
        self.temp_charts_dir = self.template_dir / "temp_charts"
        self.temp_charts_dir.mkdir(exist_ok=True)

    def generate_bar_chart(
        self,
        clean_metrics: Dict[str, float],
        attacked_metrics: Dict[str, float],
        output_path: str
    ):
        """
        Clean vs Attacked 성능 비교 막대 그래프 생성

        Args:
            clean_metrics: Clean 데이터셋 메트릭 (f1, ap50, ap75, ap, precision, recall)
            attacked_metrics: Attacked 데이터셋 메트릭
            output_path: 출력 파일 경로
        """
        # 6개 지표
        metrics_names = ['F1 Score', 'AP@50', 'AP@75', 'AP@[50:95]', 'Precision', 'Recall']
        metric_keys = ['f1', 'ap50', 'ap75', 'ap', 'precision', 'recall']

        clean_values = [clean_metrics.get(k, 0.0) for k in metric_keys]
        attacked_values = [attacked_metrics.get(k, 0.0) for k in metric_keys]

        x = np.arange(len(metrics_names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
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

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 막대 그래프 생성: {output_path}")

    def generate_pr_curve(
        self,
        clean_pr_data: Dict[str, List[float]],
        attacked_pr_data: Dict[str, List[float]],
        output_path: str
    ):
        """
        Precision-Recall Curve 생성

        Args:
            clean_pr_data: Clean PR 데이터 {'precision': [...], 'recall': [...], 'auc': float}
            attacked_pr_data: Attacked PR 데이터
            output_path: 출력 파일 경로
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Clean PR Curve
        if 'precision' in clean_pr_data and 'recall' in clean_pr_data:
            ax.plot(clean_pr_data['recall'], clean_pr_data['precision'],
                   label=f"Clean (AUC={clean_pr_data.get('auc', 0):.3f})",
                   color='#4C4CFF', linewidth=2.5)

        # Attacked PR Curve
        if 'precision' in attacked_pr_data and 'recall' in attacked_pr_data:
            ax.plot(attacked_pr_data['recall'], attacked_pr_data['precision'],
                   label=f"Attacked (AUC={attacked_pr_data.get('auc', 0):.3f})",
                   color='#cc0000', linewidth=2.5)

        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curve Comparison', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.0)
        ax.legend(loc='lower left', fontsize=11)
        ax.grid(alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ PR Curve 생성: {output_path}")

    def generate_per_class_chart(
        self,
        class_data: Dict[str, Dict[str, float]],
        output_path: str
    ):
        """
        클래스별 AP 비교 그래프 생성

        Args:
            class_data: 클래스별 데이터 {'person': {'clean': 0.8, 'attacked': 0.3}, ...}
            output_path: 출력 파일 경로
        """
        if not class_data:
            # 빈 그래프 생성
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No class-specific data available',
                   ha='center', va='center', fontsize=14, color='gray')
            ax.axis('off')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            return

        classes = list(class_data.keys())
        clean_aps = [class_data[c].get('clean', 0) for c in classes]
        attacked_aps = [class_data[c].get('attacked', 0) for c in classes]

        x = np.arange(len(classes))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - width/2, clean_aps, width, label='Clean', color='#4C4CFF', alpha=0.8)
        bars2 = ax.bar(x + width/2, attacked_aps, width, label='Attacked', color='#cc0000', alpha=0.8)

        # 감소율 표시
        for i, (c_ap, a_ap) in enumerate(zip(clean_aps, attacked_aps)):
            if c_ap > 0:
                delta = ((a_ap - c_ap) / c_ap) * 100
                ax.text(i, max(c_ap, a_ap) + 0.05, f'{delta:.1f}%',
                       ha='center', va='bottom', fontsize=9, color='red' if delta < 0 else 'green')

        ax.set_ylabel('Average Precision', fontsize=12)
        ax.set_title('Per-Class AP Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 클래스별 AP 그래프 생성: {output_path}")

    def generate_confidence_distribution(
        self,
        clean_confidences: List[float],
        attacked_confidences: List[float],
        output_path: str
    ):
        """
        Confidence Score 분포 히스토그램 생성

        Args:
            clean_confidences: Clean 이미지의 confidence 점수 리스트
            attacked_confidences: Attacked 이미지의 confidence 점수 리스트
            output_path: 출력 파일 경로
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        bins = np.linspace(0, 1, 21)  # 0.05 간격

        if clean_confidences:
            ax.hist(clean_confidences, bins=bins, alpha=0.6, label='Clean',
                   color='#4C4CFF', edgecolor='black', linewidth=0.5)

        if attacked_confidences:
            ax.hist(attacked_confidences, bins=bins, alpha=0.6, label='Attacked',
                   color='#cc0000', edgecolor='black', linewidth=0.5)

        # 평균값 표시
        if clean_confidences:
            clean_mean = np.mean(clean_confidences)
            ax.axvline(clean_mean, color='#1B1760', linestyle='--', linewidth=2,
                      label=f'Clean Mean: {clean_mean:.3f}')

        if attacked_confidences:
            attacked_mean = np.mean(attacked_confidences)
            ax.axvline(attacked_mean, color='#990000', linestyle='--', linewidth=2,
                      label=f'Attacked Mean: {attacked_mean:.3f}')

        ax.set_xlabel('Confidence Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Confidence Score Distribution', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1.0)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Confidence 분포 그래프 생성: {output_path}")

    def fill_template(self, template_content: str, data: Dict[str, Any]) -> str:
        """
        템플릿의 플레이스홀더를 실제 데이터로 치환

        Args:
            template_content: HTML 템플릿 내용
            data: 치환할 데이터 딕셔너리

        Returns:
            치환된 HTML 내용
        """
        result = template_content

        for key, value in data.items():
            placeholder = f"{{{{{key}}}}}"
            result = result.replace(placeholder, str(value))

        return result

    def generate_report(
        self,
        data: Dict[str, Any],
        output_pdf: str,
        generate_charts: bool = True
    ):
        """
        보고서 생성 메인 함수

        Args:
            data: 보고서 데이터 (플레이스홀더 값 포함)
            output_pdf: 출력 PDF 파일 경로
            generate_charts: 그래프 자동 생성 여부
        """
        print("=" * 60)
        print("적대적 공격 보고서 생성 시작")
        print("=" * 60)

        # 1. 그래프 생성 (옵션)
        if generate_charts:
            print("\n[1/4] 그래프 생성 중...")

            # 막대 그래프
            if 'clean_metrics' in data and 'attacked_metrics' in data:
                bar_chart_path = self.temp_charts_dir / "metrics_bar_chart.png"
                self.generate_bar_chart(
                    data['clean_metrics'],
                    data['attacked_metrics'],
                    str(bar_chart_path)
                )
                data['METRICS_BAR_CHART'] = str(bar_chart_path)

            # PR Curve
            if 'clean_pr_data' in data and 'attacked_pr_data' in data:
                pr_curve_path = self.temp_charts_dir / "pr_curve.png"
                self.generate_pr_curve(
                    data['clean_pr_data'],
                    data['attacked_pr_data'],
                    str(pr_curve_path)
                )
                data['PR_CURVE_CHART'] = str(pr_curve_path)

            # 클래스별 AP
            if 'per_class_data' in data:
                class_chart_path = self.temp_charts_dir / "per_class_ap.png"
                self.generate_per_class_chart(
                    data['per_class_data'],
                    str(class_chart_path)
                )
                data['PER_CLASS_AP_CHART'] = str(class_chart_path)

            # Confidence 분포
            if 'clean_confidences' in data and 'attacked_confidences' in data:
                conf_dist_path = self.temp_charts_dir / "confidence_distribution.png"
                self.generate_confidence_distribution(
                    data['clean_confidences'],
                    data['attacked_confidences'],
                    str(conf_dist_path)
                )
                data['CONFIDENCE_DISTRIBUTION_CHART'] = str(conf_dist_path)

        # 2. 템플릿 로드
        print("\n[2/4] 템플릿 로딩 중...")
        with open(self.template_path, 'r', encoding='utf-8') as f:
            template_content = f.read()
        print(f"✓ 템플릿 로드 완료: {self.template_path}")

        # 3. 플레이스홀더 치환
        print("\n[3/4] 데이터 치환 중...")
        html_content = self.fill_template(template_content, data)

        # 치환되지 않은 플레이스홀더를 빈 값으로 처리
        import re
        remaining_placeholders = re.findall(r'\{\{([A-Z_0-9]+)\}\}', html_content)
        if remaining_placeholders:
            print(f"⚠ 치환되지 않은 플레이스홀더 {len(remaining_placeholders)}개:")
            for ph in set(remaining_placeholders[:10]):  # 최대 10개만 표시
                print(f"  - {ph}")
                html_content = html_content.replace(f"{{{{{ph}}}}}", "")

        print("✓ 데이터 치환 완료")

        # 4. PDF 생성
        print("\n[4/4] PDF 생성 중...")
        output_path = Path(output_pdf)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # HTML을 PDF로 변환
        HTML(string=html_content, base_url=str(self.template_dir)).write_pdf(
            str(output_path),
            stylesheets=[CSS(filename=str(self.style_path))]
        )

        print(f"✓ PDF 생성 완료: {output_path}")
        print(f"  파일 크기: {output_path.stat().st_size / 1024:.1f} KB")

        print("\n" + "=" * 60)
        print("보고서 생성 완료!")
        print("=" * 60)

        return str(output_path)

    def cleanup(self):
        """임시 파일 정리"""
        if self.temp_charts_dir.exists():
            shutil.rmtree(self.temp_charts_dir)
            print(f"임시 파일 정리 완료: {self.temp_charts_dir}")


def load_evaluation_result(json_path: str) -> Dict[str, Any]:
    """
    평가 결과 JSON 파일 로드

    Args:
        json_path: 평가 결과 JSON 파일 경로

    Returns:
        보고서 데이터 딕셔너리
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        result = json.load(f)

    # JSON 데이터를 보고서 형식으로 변환
    data = {
        # 기본 정보
        'EXPERIMENT_DATE': result.get('experiment_date', datetime.now().strftime('%Y-%m-%d')),
        'EXPERIMENTER_NAME': result.get('experimenter', 'AI 보안팀'),
        'SYSTEM_NAME': 'Army AI Defense System',
        'REPORT_VERSION': 'v1.0',
        'REPORT_GENERATION_DATE': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),

        # 그래프 데이터 (generate_report에서 자동 생성)
        'clean_metrics': result.get('clean_metrics', {}),
        'attacked_metrics': result.get('attacked_metrics', {}),
        'clean_pr_data': result.get('clean_pr_data', {}),
        'attacked_pr_data': result.get('attacked_pr_data', {}),
        'per_class_data': result.get('per_class_data', {}),
        'clean_confidences': result.get('clean_confidences', []),
        'attacked_confidences': result.get('attacked_confidences', []),
    }

    # 나머지 모든 키를 복사
    for key, value in result.items():
        if key not in data:
            data[key] = value

    return data


def main():
    parser = argparse.ArgumentParser(description='적대적 공격 보고서 생성기')
    parser.add_argument('--data', '-d', type=str, required=True,
                       help='보고서 데이터 JSON 파일 경로')
    parser.add_argument('--output', '-o', type=str, default='adversarial_attack_report.pdf',
                       help='출력 PDF 파일 경로')
    parser.add_argument('--template-dir', '-t', type=str, default=None,
                       help='템플릿 디렉토리 경로 (기본값: 현재 디렉토리)')
    parser.add_argument('--no-charts', action='store_true',
                       help='그래프 자동 생성 비활성화')
    parser.add_argument('--keep-temp', action='store_true',
                       help='임시 파일 유지 (디버깅용)')

    args = parser.parse_args()

    # 보고서 생성기 초기화
    generator = AdversarialReportGenerator(template_dir=args.template_dir)

    try:
        # 데이터 로드
        print(f"데이터 로딩: {args.data}")
        data = load_evaluation_result(args.data)

        # 보고서 생성
        output_path = generator.generate_report(
            data=data,
            output_pdf=args.output,
            generate_charts=not args.no_charts
        )

        print(f"\n✅ 성공: {output_path}")

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        # 임시 파일 정리
        if not args.keep_temp:
            generator.cleanup()

    return 0


if __name__ == '__main__':
    exit(main())
