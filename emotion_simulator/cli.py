"""
命令行接口模块
提供用户友好的CLI界面
"""

import argparse
import sys
import os
from typing import List, Optional
from .models import EmotionDimension, DefaultDimensions
from .simulator import EmotionSimulator
from .visualizer import ASCIIVisualizer
from .exporter import JSONExporter, BatchSummarizer, CSVExporter


def read_sentences_from_file(filepath: str) -> List[str]:
    """
    从文本文件读取句子
    每行一个句子
    """
    sentences = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                sentences.append(line)
    return sentences


def parse_dimensions(dim_str: Optional[str]) -> List[EmotionDimension]:
    """
    解析自定义情绪维度
    格式: "名称1:低端标签:高端标签,名称2:低端标签:高端标签"
    """
    if not dim_str:
        return DefaultDimensions.get_defaults()

    dimensions = []
    for dim_def in dim_str.split(","):
        parts = dim_def.strip().split(":")
        if len(parts) >= 3:
            name = parts[0].strip()
            low_label = parts[1].strip()
            high_label = parts[2].strip()
            min_val = float(parts[3]) if len(parts) > 3 else -5.0
            max_val = float(parts[4]) if len(parts) > 4 else 5.0
            dimensions.append(EmotionDimension(name, low_label, high_label, min_val, max_val))

    return dimensions if dimensions else DefaultDimensions.get_defaults()


def parse_locked_words(locked_str: Optional[str]) -> List[str]:
    """
    解析锁定的关键词
    格式: "词1,词2,词3"
    """
    if not locked_str:
        return []
    return [w.strip() for w in locked_str.split(",") if w.strip()]


def parse_drift(drift_str: Optional[str]) -> dict:
    """
    解析情绪漂移设置
    格式: "维度名:方向,维度名:方向"
    方向范围: -1.0 ~ 1.0
    """
    drifts = {}
    if not drift_str:
        return drifts

    for drift_def in drift_str.split(","):
        parts = drift_def.strip().split(":")
        if len(parts) >= 2:
            dim_name = parts[0].strip()
            direction = float(parts[1].strip())
            drifts[dim_name] = max(-1.0, min(1.0, direction))

    return drifts


def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="文本情绪演进模拟器 - 模拟句子在传递过程中的情绪变化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法 - 模拟10次传递
  python main.py "今天天气不错" -n 10

  # 锁定关键词不让变化
  python main.py "今天天气不错" -n 10 -l "今天,天气"

  # 使用外部词库配置文件
  python main.py "今天天气不错" -n 10 --lexicon work_lexicon.json

  # 自定义情绪维度（配合词库使用才能有分数）
  python main.py "我喜欢这个" -n 10 -d "满意度:不满意:满意,压力:轻松:繁重"

  # 设置情绪漂移（向积极方向）
  python main.py "今天天气不错" -n 10 --drift "positive_negative:0.8"

  # 批量模拟并按变化幅度排序
  python main.py -f sentences.txt -n 10 --sort-by positive_negative_change_abs

  # 批量模拟导出汇总CSV
  python main.py -f sentences.txt -n 10 --output-summary-csv summary.csv

  # 导出结果为JSON和CSV
  python main.py "今天天气不错" -n 10 -o result.json --output-csv result.csv

  # 设置变化概率和幅度
  python main.py "今天天气不错" -n 10 -p 0.8 -m 0.7

  # 调试：显示评分匹配的词
  python main.py "有点累，今天工作很好" -n 0 --show-matched
        """
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "sentence",
        nargs="?",
        help="初始句子（用于单次模拟）"
    )
    input_group.add_argument(
        "-f", "--file",
        help="包含多个句子的文本文件（用于批量模拟，每行一个句子）"
    )

    parser.add_argument(
        "-n", "--steps",
        type=int,
        default=10,
        help="传递次数（默认：10）"
    )

    parser.add_argument(
        "-l", "--locked",
        help="锁定的关键词，用逗号分隔，这些词不会被替换"
    )

    parser.add_argument(
        "-d", "--dimensions",
        help="自定义情绪维度，格式：名称:低端标签:高端标签[,:...]"
    )

    parser.add_argument(
        "-p", "--probability",
        type=float,
        default=0.6,
        help="每次传递发生变化的概率（0.0-1.0，默认：0.6）"
    )

    parser.add_argument(
        "-m", "--magnitude",
        type=float,
        default=0.5,
        help="变化幅度（0.0-1.0，默认：0.5）"
    )

    parser.add_argument(
        "--drift",
        help="情绪漂移设置，格式：维度名:方向[,:...]，方向范围-1.0到1.0"
    )

    parser.add_argument(
        "-s", "--seed",
        type=int,
        help="随机数种子，用于可重复的模拟"
    )

    parser.add_argument(
        "-o", "--output",
        help="导出结果为JSON文件"
    )

    parser.add_argument(
        "--no-color",
        action="store_true",
        help="禁用彩色输出"
    )

    parser.add_argument(
        "--no-chart",
        action="store_true",
        help="不显示ASCII图表"
    )

    parser.add_argument(
        "--multi-dim",
        action="store_true",
        help="显示多维度叠加图表"
    )

    parser.add_argument(
        "--chart-width",
        type=int,
        default=80,
        help="图表宽度（默认：80）"
    )

    parser.add_argument(
        "--chart-height",
        type=int,
        default=15,
        help="图表高度（默认：15）"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="显示详细的每步信息"
    )

    parser.add_argument(
        "--lexicon",
        help="外部词库配置文件路径（JSON格式）"
    )

    parser.add_argument(
        "--no-merge-default",
        action="store_true",
        help="不合并默认词库（只使用外部词库）"
    )

    parser.add_argument(
        "--sort-by",
        help="批量模拟结果排序字段（如 positive_negative_change_abs, changes_count 等）"
    )

    parser.add_argument(
        "--sort-ascending",
        action="store_true",
        help="按升序排列（默认降序）"
    )

    parser.add_argument(
        "--output-csv",
        help="导出时间序列数据为CSV文件"
    )

    parser.add_argument(
        "--output-summary",
        help="导出批量汇总为简洁的JSON文件"
    )

    parser.add_argument(
        "--output-summary-csv",
        help="导出批量汇总为CSV文件"
    )

    parser.add_argument(
        "--show-matched",
        action="store_true",
        help="显示评分时匹配到的词（用于调试评分逻辑）"
    )

    return parser


def create_simulator(args: argparse.Namespace, cli_dimensions: Optional[List[EmotionDimension]] = None) -> EmotionSimulator:
    """根据参数创建模拟器"""
    if args.lexicon:
        simulator = EmotionSimulator.from_lexicon_file(
            lexicon_filepath=args.lexicon,
            merge_default=not args.no_merge_default,
            change_probability=args.probability,
            magnitude=args.magnitude,
            random_seed=args.seed,
        )

        if cli_dimensions:
            existing_dim_names = {d.name for d in simulator.dimensions}
            for dim in cli_dimensions:
                if dim.name not in existing_dim_names:
                    simulator.add_custom_dimension(dim)
    else:
        dimensions = cli_dimensions or parse_dimensions(args.dimensions)
        simulator = EmotionSimulator(
            dimensions=dimensions,
            change_probability=args.probability,
            magnitude=args.magnitude,
            random_seed=args.seed,
        )

    return simulator


def run_single_simulation(args: argparse.Namespace) -> None:
    """运行单次模拟"""
    print("\n" + "=" * 60)
    print("文本情绪演进模拟器 - 单次模拟")
    print("=" * 60)

    cli_dimensions = parse_dimensions(args.dimensions) if args.dimensions else None
    locked_words = parse_locked_words(args.locked)
    drifts = parse_drift(args.drift)

    print(f"\n初始句子: {args.sentence}")
    print(f"传递次数: {args.steps}")
    print(f"变化概率: {args.probability:.1%}")
    print(f"变化幅度: {args.magnitude:.1%}")
    if args.lexicon:
        print(f"词库配置: {args.lexicon}")
        if args.no_merge_default:
            print(f"不合并默认词库")
    if locked_words:
        print(f"锁定关键词: {', '.join(locked_words)}")
    if drifts:
        print(f"情绪漂移: {drifts}")

    simulator = create_simulator(args, cli_dimensions)

    print(f"情绪维度: {', '.join([d.name for d in simulator.dimensions])}")

    for dim_name, direction in drifts.items():
        simulator.set_drift(dim_name, direction)

    visualizer = ASCIIVisualizer(
        width=args.chart_width,
        height=args.chart_height,
    )

    use_color = not args.no_color

    def step_callback(step):
        if args.verbose:
            print()
            print(visualizer.print_step_info(step, use_color=use_color))

    print("\n开始模拟...")
    if not args.verbose:
        print()

    result = simulator.simulate(
        initial_sentence=args.sentence,
        num_steps=args.steps,
        locked_words=locked_words,
        callback=step_callback if args.verbose else None,
    )

    if not args.verbose:
        print("\n" + "-" * 60)
        print("传递过程摘要:")
        print("-" * 60)
        for step in result.steps:
            print(visualizer.print_step_info(step, use_color=use_color))
            print()

    print("=" * 60)
    print("模拟完成！")
    print("=" * 60)
    print(f"\n初始句子: {result.initial_sentence}")
    print(f"最终句子: {result.get_final_sentence()}")

    changes_count = sum(1 for s in result.steps if s.changed_word)
    print(f"总变化次数: {changes_count}/{args.steps}")

    if args.show_matched:
        print("\n" + "=" * 60)
        print("评分匹配详情（最长匹配）:")
        print("=" * 60)
        details = simulator.scorer.get_score_details(args.sentence)
        print(f"句子: {details['sentence']}")
        print(f"匹配到 {len(details['matched_words'])} 个词:")
        for i, match in enumerate(details['matched_words'], 1):
            print(f"  {i}. 位置{match['position']}: '{match['word']}'")
            print(f"     强度: {match['intensity']}, 分数: {match['emotion_scores']}")
        print(f"最终分数: {details['final_scores']}")
        print()

        if args.steps == 0:
            return

    if result.steps:
        print("\n情绪评分变化:")
        for dim in simulator.dimensions:
            series = result.get_emotion_series(dim.name)
            if series:
                print(f"  {dim.high_label} vs {dim.low_label}: "
                      f"{series[0]:+.2f} → {series[-1]:+.2f} "
                      f"(变化: {series[-1] - series[0]:+.2f})")

    if not args.no_chart and args.steps > 0:
        print("\n" + "=" * 60)
        if args.multi_dim:
            print("多维度情绪演进曲线:")
            print("=" * 60)
            print(visualizer.plot_multi_dimension(result, use_color=use_color))
        else:
            for dim in simulator.dimensions:
                print(f"\n{dim.high_label} vs {dim.low_label} 情绪演进曲线:")
                print("=" * 60)
                print(visualizer.plot_single(result, dim.name, use_color=use_color))
                print()

    if args.output:
        JSONExporter.export(result, args.output)
        print(f"\n结果已导出到: {args.output}")

    if args.output_csv:
        CSVExporter.export_timeseries(result, args.output_csv)
        print(f"时间序列CSV已导出到: {args.output_csv}")


def run_batch_simulation(args: argparse.Namespace) -> None:
    """运行批量模拟"""
    print("\n" + "=" * 60)
    print("文本情绪演进模拟器 - 批量模拟")
    print("=" * 60)

    sentences = read_sentences_from_file(args.file)
    if not sentences:
        print(f"错误: 文件 {args.file} 中没有有效的句子")
        sys.exit(1)

    cli_dimensions = parse_dimensions(args.dimensions) if args.dimensions else None
    locked_words = parse_locked_words(args.locked)
    drifts = parse_drift(args.drift)

    print(f"\n输入文件: {args.file}")
    print(f"句子数量: {len(sentences)}")
    print(f"传递次数: {args.steps}")
    print(f"变化概率: {args.probability:.1%}")
    print(f"变化幅度: {args.magnitude:.1%}")
    if args.lexicon:
        print(f"词库配置: {args.lexicon}")
        if args.no_merge_default:
            print(f"不合并默认词库")
    if locked_words:
        print(f"锁定关键词: {', '.join(locked_words)}")
    if drifts:
        print(f"情绪漂移: {drifts}")
    if args.sort_by:
        print(f"排序字段: {args.sort_by} ({'升序' if args.sort_ascending else '降序'})")

    simulator = create_simulator(args, cli_dimensions)
    print(f"情绪维度: {', '.join([d.name for d in simulator.dimensions])}")

    print("\n待模拟句子:")
    for i, sentence in enumerate(sentences, 1):
        print(f"  {i}. {sentence}")

    for dim_name, direction in drifts.items():
        simulator.set_drift(dim_name, direction)

    visualizer = ASCIIVisualizer(
        width=args.chart_width,
        height=args.chart_height,
    )

    use_color = not args.no_color

    print("\n开始批量模拟...")
    results = simulator.batch_simulate(
        sentences=sentences,
        num_steps=args.steps,
        locked_words=locked_words,
    )

    print("\n" + "=" * 60)
    print("批量模拟完成！")
    print("=" * 60)

    for i, result in enumerate(results, 1):
        print(f"\n[{i}/{len(results)}] {result.initial_sentence}")
        print(f"    → {result.get_final_sentence()}")

        changes_count = sum(1 for s in result.steps if s.changed_word)
        print(f"    变化次数: {changes_count}/{args.steps}")

        if result.steps:
            for dim in simulator.dimensions:
                series = result.get_emotion_series(dim.name)
                if series:
                    print(f"    {dim.name}: {series[0]:+.2f} → {series[-1]:+.2f}")

    print("\n" + BatchSummarizer.print_summary(
        results,
        sort_by=args.sort_by,
        sort_ascending=args.sort_ascending,
        use_color=use_color,
    ))

    if not args.no_chart and args.steps > 0:
        print("\n" + "=" * 60)
        for dim in simulator.dimensions:
            print(f"\n{dim.high_label} vs {dim.low_label} 对比曲线:")
            print("=" * 60)
            labels = [s[:20] + "..." if len(s) > 20 else s for s in sentences]
            print(visualizer.plot_comparison(results, dim.name, labels=labels, use_color=use_color))
            print()

    if args.output:
        JSONExporter.export_batch(results, args.output)
        print(f"\n批量结果已导出到: {args.output}")

    if args.output_csv:
        CSVExporter.export_batch_timeseries(results, args.output_csv)
        print(f"批量时间序列CSV已导出到: {args.output_csv}")

    if args.output_summary:
        BatchSummarizer.export_summary_json(
            results,
            args.output_summary,
            sort_by=args.sort_by,
            sort_ascending=args.sort_ascending,
        )
        print(f"汇总JSON已导出到: {args.output_summary}")

    if args.output_summary_csv:
        BatchSummarizer.export_summary_csv(
            results,
            args.output_summary_csv,
            sort_by=args.sort_by,
            sort_ascending=args.sort_ascending,
        )
        print(f"汇总CSV已导出到: {args.output_summary_csv}")


def main() -> None:
    """主函数"""
    parser = create_parser()
    args = parser.parse_args()

    try:
        if args.file:
            run_batch_simulation(args)
        else:
            run_single_simulation(args)
    except KeyboardInterrupt:
        print("\n\n用户中断，退出程序。")
        sys.exit(1)
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
