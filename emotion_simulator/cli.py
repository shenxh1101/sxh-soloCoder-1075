"""
命令行接口模块
提供用户友好的CLI界面
"""

import argparse
import sys
import os
import json
from typing import List, Optional, Dict

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
from .models import (
    EmotionDimension,
    DefaultDimensions,
    WordConflictStrategy,
    SimulationProfile,
    export_lexicon_template,
    save_profile,
    load_profile,
    list_profiles,
    delete_profile,
    dimension_to_dict,
)
from .simulator import EmotionSimulator
from .visualizer import ASCIIVisualizer
from .exporter import JSONExporter, BatchSummarizer, CSVExporter, YAMLExporter


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


def parse_word_scores(score_str: Optional[str]) -> Dict[str, Dict[str, float]]:
    """
    解析词分覆盖设置
    格式: "词1:维度1=分数,维度2=分数;词2:维度1=分数"
    示例: "有点累:work_pressure=3.0,positive_negative=-2.0;很好:positive_negative=4.0"
    """
    overrides = {}
    if not score_str:
        return overrides

    for word_def in score_str.split(";"):
        word_def = word_def.strip()
        if not word_def or ":" not in word_def:
            continue

        word_part, scores_part = word_def.split(":", 1)
        word = word_part.strip()
        if not word:
            continue

        scores = {}
        for score_def in scores_part.split(","):
            score_def = score_def.strip()
            if "=" in score_def:
                dim_name, score_val = score_def.split("=", 1)
                try:
                    scores[dim_name.strip()] = float(score_val.strip())
                except ValueError:
                    continue

        if scores:
            overrides[word] = scores

    return overrides


def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="文本情绪演进模拟器 - 模拟句子在传递过程中的情绪变化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
【基础用法】
  # 基本用法 - 模拟10次传递
  python main.py "今天天气不错" -n 10

  # 使用外部词库配置文件
  python main.py "今天天气不错" -n 10 --lexicon work_lexicon.json

【自定义维度和词分】
  # 命令行指定词的维度分数（无需改代码）
  python main.py "今天工作有点累" -n 10 -d "压力:轻松:压力大" \\
    --word-scores "有点累:压力=3.5,positive_negative=-2.0;很好:positive_negative=4.0"

【词库管理】
  # 导出词库模板（JSON格式，包含默认词库）
  python main.py --export-template my_lexicon.json

  # 导出词库模板（YAML格式，极简模板）
  python main.py --export-template my_lexicon.yaml --no-default-lexicon

  # 交互式创建词库配置
  python main.py --wizard

【批量分析】
  # 批量模拟并显示对比报告，包含主要命中词和贡献最大维度
  python main.py -f sentences.txt -n 10 --report

  # 只显示变化幅度Top 5的结果，按positive_negative维度筛选
  python main.py -f sentences.txt -n 10 --report --top 5 --filter-dim positive_negative

  # 批量模拟按变化幅度排序并导出
  python main.py -f sentences.txt -n 10 --sort-by positive_negative_change_abs \\
    --output-summary-csv summary.csv --output-summary summary.json

【其他选项】
  # 设置词冲突处理策略（keep_first/keep_last/error/merge）
  python main.py "句子" -n 10 --lexicon work_lexicon.json --conflict-strategy merge

  # 调试：显示评分匹配的词和主要贡献
  python main.py "有点累，今天工作很好" -n 0 --show-matched

  # 锁定关键词不让变化
  python main.py "今天天气不错" -n 10 -l "今天,天气"
        """
    )

    input_group = parser.add_argument_group("输入选项")
    input_source = input_group.add_mutually_exclusive_group(required=False)
    input_source.add_argument(
        "sentence",
        nargs="?",
        help="初始句子（用于单次模拟）"
    )
    input_source.add_argument(
        "-f", "--file",
        help="包含多个句子的文本文件（用于批量模拟，每行一个句子）"
    )

    special_group = parser.add_argument_group("特殊命令（无需输入句子）")
    special = special_group.add_mutually_exclusive_group()
    special.add_argument(
        "--export-template",
        help="导出词库配置模板到指定文件（JSON或YAML）"
    )
    special.add_argument(
        "--wizard",
        action="store_true",
        help="启动交互式配置向导，一步步创建词库"
    )
    special.add_argument(
        "--list-profiles",
        action="store_true",
        help="列出所有可用的场景配置Profile"
    )
    special.add_argument(
        "--delete-profile",
        metavar="NAME",
        help="删除指定的场景配置Profile"
    )
    parser.add_argument(
        "--no-default-lexicon",
        action="store_true",
        help="导出模板时不包含默认词库（只生成极简模板）"
    )

    profile_group = parser.add_argument_group("场景配置Profile")
    profile_group.add_argument(
        "--profile",
        metavar="NAME",
        help="加载指定的场景配置Profile"
    )
    profile_group.add_argument(
        "--save-profile",
        metavar="NAME",
        help="将当前配置保存为场景配置Profile"
    )
    profile_group.add_argument(
        "--profile-description",
        default="",
        help="保存Profile时的描述信息"
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
        help="导出结果为JSON文件（后缀为.yaml/.yml时自动导出YAML格式）"
    )

    parser.add_argument(
        "--output-yaml",
        help="导出结果为YAML文件"
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
        help="外部词库配置文件路径（JSON或YAML格式）"
    )

    parser.add_argument(
        "--no-merge-default",
        action="store_true",
        help="不合并默认词库（只使用外部词库）"
    )

    parser.add_argument(
        "--word-scores",
        help="指定词的维度分数，格式：词1:维度1=分数,维度2=分数;词2:维度1=分数"
    )

    parser.add_argument(
        "--conflict-strategy",
        choices=["keep_first", "keep_last", "error", "merge"],
        default="keep_first",
        help="词冲突处理策略：keep_first(保留第一个)|keep_last(保留最后一个)|error(报错)|merge(合并分数)"
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
        "--output-summary-yaml",
        help="导出批量汇总为YAML文件"
    )

    parser.add_argument(
        "--show-matched",
        action="store_true",
        help="显示评分时匹配到的词（用于调试评分逻辑）"
    )

    parser.add_argument(
        "--trace",
        action="store_true",
        help="显示可追溯分析报告，每步展示命中词、替换词、维度贡献变化"
    )

    parser.add_argument(
        "--trace-top",
        type=int,
        default=0,
        help="批量追踪时只显示变化最大的前N条句子（0表示全部显示）"
    )

    batch_report_group = parser.add_argument_group("批量分析选项")
    batch_report_group.add_argument(
        "--report",
        action="store_true",
        help="显示批量对比报告，包含主要命中词和贡献最大维度"
    )
    batch_report_group.add_argument(
        "--top",
        type=int,
        default=0,
        help="只显示Top N个结果（0表示全部显示）"
    )
    batch_report_group.add_argument(
        "--filter-dim",
        help="按维度筛选，只显示该维度变化不为0的结果"
    )

    multi_scene_group = parser.add_argument_group("多场景批量对比")
    multi_scene_group.add_argument(
        "--multi-scene",
        help="多场景词库路径，用逗号分隔，同一批句子分别用不同词库运行"
    )
    multi_scene_group.add_argument(
        "--scene-names",
        help="场景名称，用逗号分隔，与--multi-scene一一对应"
    )

    return parser


def apply_profile_to_args(profile: SimulationProfile, args: argparse.Namespace) -> argparse.Namespace:
    """将Profile中的配置应用到args"""
    if profile.dimensions and not args.dimensions:
        dim_strs = []
        for d in profile.dimensions:
            dim_strs.append(f"{d['name']}:{d['low_label']}:{d['high_label']}")
        args.dimensions = ",".join(dim_strs)
    
    if profile.lexicon_path and not args.lexicon:
        args.lexicon = profile.lexicon_path
    
    if profile.no_merge_default:
        args.no_merge_default = True
    
    if profile.word_score_overrides and not args.word_scores:
        score_strs = []
        for word, scores in profile.word_score_overrides.items():
            dim_scores = ",".join([f"{k}={v}" for k, v in scores.items()])
            score_strs.append(f"{word}:{dim_scores}")
        args.word_scores = ";".join(score_strs)
    
    if profile.conflict_strategy and args.conflict_strategy == "keep_first":
        args.conflict_strategy = profile.conflict_strategy
    
    if profile.change_probability != 0.6 and args.probability == 0.6:
        args.probability = profile.change_probability
    
    if profile.magnitude != 0.5 and args.magnitude == 0.5:
        args.magnitude = profile.magnitude
    
    if profile.drift and not args.drift:
        drift_strs = []
        for dim, direction in profile.drift.items():
            drift_strs.append(f"{dim}:{direction}")
        args.drift = ",".join(drift_strs)
    
    return args


def list_profiles_cli() -> None:
    """列出所有可用的Profile"""
    profiles = list_profiles()
    
    print("\n" + "=" * 80)
    print("  可用的场景配置Profile")
    print("=" * 80)
    
    if not profiles:
        print("\n  暂无可用的Profile")
        print("  使用 --save-profile <名称> 保存当前配置为Profile")
    else:
        print(f"\n  共找到 {len(profiles)} 个Profile:\n")
        for i, p in enumerate(profiles, 1):
            print(f"  {i}. {p['name']}")
            if p['description']:
                print(f"     描述: {p['description']}")
            if p['updated_at']:
                print(f"     更新时间: {p['updated_at']}")
            print(f"     文件: {p['filepath']}")
            print()
    
    print("=" * 80)


def delete_profile_cli(name: str) -> None:
    """删除指定的Profile"""
    print("\n" + "=" * 60)
    print(f"  删除Profile: {name}")
    print("=" * 60)
    
    if delete_profile(name):
        print(f"\n✓ Profile '{name}' 已删除")
    else:
        print(f"\n✗ Profile '{name}' 不存在")
    
    print("=" * 60)


def save_current_profile(
    args: argparse.Namespace,
    cli_dimensions: Optional[List[EmotionDimension]],
    word_score_overrides: Optional[Dict[str, Dict[str, float]]],
    drifts: Optional[Dict[str, float]],
) -> None:
    """将当前配置保存为Profile"""
    name = args.save_profile
    
    dimensions = []
    if cli_dimensions:
        for d in cli_dimensions:
            dimensions.append(dimension_to_dict(d))
    
    profile = SimulationProfile(
        name=name,
        description=args.profile_description,
        dimensions=dimensions,
        lexicon_path=args.lexicon,
        no_merge_default=args.no_merge_default,
        word_score_overrides=word_score_overrides or {},
        conflict_strategy=args.conflict_strategy,
        change_probability=args.probability,
        magnitude=args.magnitude,
        drift=drifts or {},
    )
    
    filepath = save_profile(profile)
    
    print("\n" + "=" * 60)
    print(f"  保存Profile: {name}")
    print("=" * 60)
    print(f"\n✓ Profile已保存到: {filepath}")
    print(f"\n  下次使用: python main.py ... --profile {name}")
    print("=" * 60)


def create_simulator(
    args: argparse.Namespace,
    cli_dimensions: Optional[List[EmotionDimension]] = None,
    word_score_overrides: Optional[Dict[str, Dict[str, float]]] = None,
) -> EmotionSimulator:
    """根据参数创建模拟器"""
    conflict_strategy = WordConflictStrategy(args.conflict_strategy)

    if args.lexicon:
        simulator = EmotionSimulator.from_lexicon_file(
            lexicon_filepath=args.lexicon,
            merge_default=not args.no_merge_default,
            change_probability=args.probability,
            magnitude=args.magnitude,
            random_seed=args.seed,
            conflict_strategy=conflict_strategy,
            word_score_overrides=word_score_overrides,
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
            conflict_strategy=conflict_strategy,
        )

        if word_score_overrides:
            simulator.apply_word_score_overrides(word_score_overrides)

    return simulator


def run_single_simulation(args: argparse.Namespace) -> None:
    """运行单次模拟"""
    print("\n" + "=" * 60)
    print("文本情绪演进模拟器 - 单次模拟")
    print("=" * 60)

    if args.profile:
        profile = load_profile(args.profile)
        print(f"\n加载Profile: {profile.name}")
        if profile.description:
            print(f"描述: {profile.description}")
        args = apply_profile_to_args(profile, args)

    cli_dimensions = parse_dimensions(args.dimensions) if args.dimensions else None
    locked_words = parse_locked_words(args.locked)
    drifts = parse_drift(args.drift)
    word_score_overrides = parse_word_scores(args.word_scores)

    print(f"\n初始句子: {args.sentence}")
    print(f"传递次数: {args.steps}")
    print(f"变化概率: {args.probability:.1%}")
    print(f"变化幅度: {args.magnitude:.1%}")
    if args.lexicon:
        print(f"词库配置: {args.lexicon}")
        if args.no_merge_default:
            print(f"不合并默认词库")
    if args.profile:
        print(f"使用Profile: {args.profile}")
    if locked_words:
        print(f"锁定关键词: {', '.join(locked_words)}")
    if drifts:
        print(f"情绪漂移: {drifts}")
    if word_score_overrides:
        print(f"词分覆盖: {len(word_score_overrides)} 个词")
        for word, scores in word_score_overrides.items():
            print(f"  - {word}: {scores}")
    print(f"冲突策略: {args.conflict_strategy}")

    simulator = create_simulator(args, cli_dimensions, word_score_overrides)

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
        details = simulator.scorer.get_score_details_dict(args.sentence)
        print(f"句子: {details['sentence']}")
        print(f"匹配到 {len(details['matched_words'])} 个词:")
        for i, match in enumerate(details['matched_words'], 1):
            group_info = f" (词组: {match['group_keyword']})" if match['group_keyword'] else ""
            print(f"  {i}. 位置{match['position']}: '{match['word']}'{group_info}")
            print(f"     强度: {match['intensity']}, 分数: {match['emotion_scores']}")
        print(f"最终分数: {details['final_scores']}")
        
        if details.get('main_contributor'):
            mc = details['main_contributor']
            print(f"\n主要贡献词: '{mc['word']}' (贡献值: {mc['contribution']:.2f})")
        
        if details.get('top_dimension'):
            td = details['top_dimension']
            print(f"最高维度: {td['dimension']} = {td['score']:+.2f} ({td['label']})")
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

    if not args.no_chart:
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
        if args.output.lower().endswith(('.yaml', '.yml')):
            YAMLExporter.export(result, args.output)
            print(f"\n结果已导出到YAML文件: {args.output}")
        else:
            JSONExporter.export(result, args.output)
            print(f"\n结果已导出到JSON文件: {args.output}")

    if args.output_yaml:
        YAMLExporter.export(result, args.output_yaml)
        print(f"\n结果已导出到YAML文件: {args.output_yaml}")

    if args.output_csv:
        CSVExporter.export_timeseries(result, args.output_csv)
        print(f"时间序列CSV已导出到: {args.output_csv}")

    if args.trace:
        print(print_trace_report(result, simulator, use_color=use_color))

    if args.save_profile:
        save_current_profile(args, cli_dimensions, word_score_overrides, drifts)


def run_batch_simulation(args: argparse.Namespace) -> None:
    """运行批量模拟"""
    print("\n" + "=" * 60)
    print("文本情绪演进模拟器 - 批量模拟")
    print("=" * 60)

    if args.profile:
        profile = load_profile(args.profile)
        print(f"\n加载Profile: {profile.name}")
        if profile.description:
            print(f"描述: {profile.description}")
        args = apply_profile_to_args(profile, args)

    sentences = read_sentences_from_file(args.file)
    if not sentences:
        print(f"错误: 文件 {args.file} 中没有有效的句子")
        sys.exit(1)

    cli_dimensions = parse_dimensions(args.dimensions) if args.dimensions else None
    locked_words = parse_locked_words(args.locked)
    drifts = parse_drift(args.drift)
    word_score_overrides = parse_word_scores(args.word_scores)

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
    if word_score_overrides:
        print(f"词分覆盖: {len(word_score_overrides)} 个词")
    if args.sort_by:
        print(f"排序字段: {args.sort_by} ({'升序' if args.sort_ascending else '降序'})")
    if args.filter_dim:
        print(f"维度筛选: {args.filter_dim}")
    if args.top > 0:
        print(f"显示Top: {args.top}")
    print(f"冲突策略: {args.conflict_strategy}")

    simulator = create_simulator(args, cli_dimensions, word_score_overrides)
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

    if args.report:
        print("\n" + print_batch_report(
            results,
            simulator,
            sort_by=args.sort_by,
            sort_ascending=args.sort_ascending,
            filter_dim=args.filter_dim,
            top_n=args.top,
            use_color=use_color,
        ))
    else:
        print("\n" + BatchSummarizer.print_summary(
            results,
            sort_by=args.sort_by,
            sort_ascending=args.sort_ascending,
            use_color=use_color,
        ))

    if not args.no_chart:
        print("\n" + "=" * 60)
        for dim in simulator.dimensions:
            print(f"\n{dim.high_label} vs {dim.low_label} 对比曲线:")
            print("=" * 60)
            labels = [s[:20] + "..." if len(s) > 20 else s for s in sentences]
            print(visualizer.plot_comparison(results, dim.name, labels=labels, use_color=use_color))
            print()

    if args.output:
        if args.output.lower().endswith(('.yaml', '.yml')):
            YAMLExporter.export_batch(results, args.output)
            print(f"\n批量结果已导出到YAML文件: {args.output}")
        else:
            JSONExporter.export_batch(results, args.output)
            print(f"\n批量结果已导出到JSON文件: {args.output}")

    if args.output_yaml:
        YAMLExporter.export_batch(results, args.output_yaml)
        print(f"\n批量结果已导出到YAML文件: {args.output_yaml}")

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

    if args.output_summary_yaml:
        YAMLExporter.export_summary(
            results,
            args.output_summary_yaml,
            sort_by=args.sort_by,
            sort_ascending=args.sort_ascending,
        )
        print(f"汇总YAML已导出到: {args.output_summary_yaml}")

    if args.trace:
        print(print_batch_trace_report(results, simulator, top_n=args.trace_top, use_color=use_color))

    if args.save_profile:
        save_current_profile(args, cli_dimensions, word_score_overrides, drifts)


def print_trace_report(result, simulator: EmotionSimulator, use_color: bool = True) -> str:
    """
    生成并打印可追溯分析报告
    每步展示命中词、替换词、维度贡献变化
    """
    lines = []
    lines.append("\n" + "=" * 90)
    lines.append("  可追溯分析报告")
    lines.append("=" * 90)
    
    for step in result.steps:
        lines.append(f"\n[第 {step.step} 步]")
        lines.append(f"  句子: {step.modified_sentence}")
        lines.append(f"  分数: {step.emotion_scores}")
        
        if step.score_contribution:
            sc = step.score_contribution
            
            matched_words = [mw.word for mw in sc.matched_words]
            if matched_words:
                lines.append(f"  命中词: {', '.join(matched_words)}")
                
                for mw in sc.matched_words:
                    contrib_str = ", ".join([f"{k}={v:+.2f}" for k, v in mw.contribution.items()])
                    lines.append(f"    - '{mw.word}' (词组: {mw.group_keyword}): {contrib_str}")
            
            if sc.main_contributor:
                mc = sc.main_contributor
                total_contrib = sum(abs(v) for v in mc.contribution.values())
                lines.append(f"  主要贡献: '{mc.word}' (总贡献: {total_contrib:.2f})")
            
            if sc.top_dimension:
                td = sc.top_dimension
                lines.append(f"  最高维度: {td['dimension']} = {td['score']:+.2f} ({td['label']})")
        
        if step.replacement_detail:
            rd = step.replacement_detail
            lines.append(f"\n  替换发生: '{rd.original_word}' → '{rd.new_word}'")
            lines.append(f"    位置: {rd.position}, 目标维度: {rd.target_dimension}")
            lines.append(f"    替换前分数: {rd.before_scores}")
            lines.append(f"    替换后分数: {rd.after_scores}")
            
            changed_dims = []
            for dim in rd.before_scores.keys():
                before = rd.before_scores.get(dim, 0)
                after = rd.after_scores.get(dim, 0)
                if abs(after - before) > 0.001:
                    changed_dims.append(f"{dim}: {before:+.2f} → {after:+.2f} (Δ{after-before:+.2f})")
            if changed_dims:
                lines.append(f"    维度变化: {', '.join(changed_dims)}")
        
        lines.append("  " + "-" * 60)
    
    return "\n".join(lines)


def print_batch_trace_report(results, simulator: EmotionSimulator, top_n: int = 0, use_color: bool = True) -> str:
    """
    生成并打印批量可追溯分析报告
    支持按变化幅度排序和Top N筛选
    """
    from .exporter import get_result_metrics
    
    result_with_metrics = []
    for result in results:
        metrics = get_result_metrics(result)
        total_change = 0
        for dim in simulator.dimensions:
            change_key = f"{dim.name}_change_abs"
            total_change += metrics.get(change_key, 0)
        result_with_metrics.append((result, metrics, total_change))
    
    result_with_metrics.sort(key=lambda x: x[2], reverse=True)
    
    if top_n > 0:
        result_with_metrics = result_with_metrics[:top_n]
    
    lines = []
    lines.append("=" * 80)
    lines.append("批量可追溯分析报告")
    if top_n > 0:
        lines.append(f"(显示变化最大的前 {top_n} 条)")
    lines.append("=" * 80)
    
    for idx, (result, metrics, total_change) in enumerate(result_with_metrics, 1):
        lines.append(f"\n[{idx}/{len(result_with_metrics)}] 总变化幅度: {total_change:+.2f}")
        lines.append(f"初始句子: {result.initial_sentence}")
        lines.append(f"最终句子: {result.get_final_sentence()}")
        lines.append(f"变化次数: {metrics['changes_count']}/{metrics['total_steps']}")
        
        dim_changes = []
        for dim in simulator.dimensions:
            change_key = f"{dim.name}_change"
            change = metrics.get(change_key, 0)
            if abs(change) > 0.001:
                dim_changes.append(f"{dim.name}: {change:+.2f}")
        if dim_changes:
            lines.append(f"维度变化: {', '.join(dim_changes)}")
        
        lines.append(f"\n  步骤详情:")
        lines.append(f"  {'-' * 76}")
        
        for step in result.steps:
            if step.step == 0:
                line = f"  [Step 0] 初始: {step.modified_sentence}"
                if step.score_contribution and step.score_contribution.main_contributor:
                    mc = step.score_contribution.main_contributor
                    first_dim = list(mc.contribution.keys())[0] if mc.contribution else ""
                    contrib_val = mc.contribution.get(first_dim, 0)
                    line += f" [主要贡献: '{mc.word}'={contrib_val:.2f}]"
                lines.append(line)
            else:
                if step.changed_word:
                    line = f"  [Step {step.step}] '{step.changed_word[0]}' → '{step.changed_word[1]}'"
                    if step.replacement_detail:
                        rd = step.replacement_detail
                        line += f" [目标维度: {rd.target_dimension}]"
                        max_change_dim = None
                        max_change_val = 0
                        for dim in rd.before_scores.keys():
                            before = rd.before_scores.get(dim, 0)
                            after = rd.after_scores.get(dim, 0)
                            change = abs(after - before)
                            if change > max_change_val:
                                max_change_val = change
                                max_change_dim = dim
                        if max_change_dim:
                            before = rd.before_scores.get(max_change_dim, 0)
                            after = rd.after_scores.get(max_change_dim, 0)
                            line += f" [{max_change_dim}: {before:+.2f}→{after:+.2f}]"
                    lines.append(line)
                else:
                    lines.append(f"  [Step {step.step}] 无变化")
            
            if step.score_contribution and step.score_contribution.matched_words:
                matched_words = [mw.word for mw in step.score_contribution.matched_words]
                lines.append(f"            命中词: {', '.join(matched_words)}")
        
        lines.append(f"  {'-' * 76}")
    
    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


def print_batch_report(
    results,
    simulator: EmotionSimulator,
    sort_by: Optional[str] = None,
    sort_ascending: bool = False,
    filter_dim: Optional[str] = None,
    top_n: int = 0,
    use_color: bool = True,
) -> str:
    """
    生成并打印批量对比报告
    包含主要命中词和贡献最大维度
    """
    from .exporter import get_result_metrics
    
    metrics_list = []
    for result in results:
        metrics = get_result_metrics(result)
        initial_details = simulator.scorer.get_score_details_dict(result.initial_sentence)
        final_details = simulator.scorer.get_score_details_dict(result.get_final_sentence())
        metrics['_initial_details'] = initial_details
        metrics['_final_details'] = final_details
        metrics_list.append(metrics)
    
    if sort_by:
        reverse = not sort_ascending
        metrics_list.sort(key=lambda m: m.get(sort_by, 0), reverse=reverse)
    
    if filter_dim:
        change_key = f"{filter_dim}_change_abs"
        metrics_list = [m for m in metrics_list if m.get(change_key, 0) != 0]
    
    if top_n > 0:
        metrics_list = metrics_list[:top_n]
    
    lines = []
    lines.append("=" * 90)
    title = "批量对比报告"
    if filter_dim:
        title += f" (筛选: {filter_dim})"
    if top_n > 0:
        title += f" (Top {top_n})"
    lines.append(f"  {title}")
    lines.append("=" * 90)
    
    header = f"{'#':<3} {'初始句子':<20} {'变化':<6} {'主要命中词':<12} {'贡献最大维度':<18} {'分数变化':<18}"
    lines.append(header)
    lines.append("-" * 90)
    
    for i, metrics in enumerate(metrics_list, 1):
        initial = metrics['_initial_details']
        final = metrics['_final_details']
        
        main_word = initial.get('main_contributor', {}).get('word', '-') if initial.get('main_contributor') else '-'
        top_dim_info = final.get('top_dimension', {})
        top_dim = f"{top_dim_info.get('dimension', '-')}: {top_dim_info.get('score', 0):+.2f}" if top_dim_info else '-'
        
        changes = []
        for dim in simulator.dimensions[:2]:
            change_key = f"{dim.name}_change"
            change = metrics.get(change_key, 0)
            changes.append(f"{dim.name[:8]}: {change:+.2f}")
        change_str = ", ".join(changes)
        
        initial_sent = metrics['initial_sentence']
        if len(initial_sent) > 18:
            initial_sent = initial_sent[:17] + "…"
        
        lines.append(
            f"{i:<3} {initial_sent:<20} {metrics['changes_count']:<6} "
            f"{main_word:<12} {top_dim:<18} {change_str:<18}"
        )
    
    lines.append("=" * 90)
    
    if metrics_list:
        lines.append("\n详细信息:")
        for i, metrics in enumerate(metrics_list, 1):
            initial = metrics['_initial_details']
            final = metrics['_final_details']
            
            main_contrib = initial.get('main_contributor', {})
            top_dim = final.get('top_dimension', {})
            
            matched_words = [m['word'] for m in initial.get('matched_words', [])]
            
            lines.append(f"\n{i}. {metrics['initial_sentence']}")
            lines.append(f"   → {metrics['final_sentence']}")
            lines.append(f"   变化次数: {metrics['changes_count']}/{metrics['total_steps']}")
            lines.append(f"   主要命中词: {main_contrib.get('word', '-')} (贡献: {main_contrib.get('contribution', 0):.2f})")
            lines.append(f"   所有匹配词: {', '.join(matched_words) if matched_words else '-'}")
            if top_dim:
                lines.append(f"   贡献最大维度: {top_dim.get('dimension')} = {top_dim.get('score', 0):+.2f} ({top_dim.get('label', '')})")
            
            for dim in simulator.dimensions:
                change_key = f"{dim.name}_change"
                change = metrics.get(change_key, 0)
                min_key = f"{dim.name}_min"
                max_key = f"{dim.name}_max"
                min_val = metrics.get(min_key, 0)
                max_val = metrics.get(max_key, 0)
                lines.append(f"   {dim.name}: {min_val:+.2f} ~ {max_val:+.2f} (变化: {change:+.2f})")
    
    return "\n".join(lines)


def run_multi_scene_batch(args: argparse.Namespace) -> None:
    """
    多场景批量对比
    同一批句子分别用不同词库（场景）运行，生成对比报告
    """
    scenes = [s.strip() for s in args.multi_scene.split(",")]
    
    if args.scene_names:
        scene_names = [n.strip() for n in args.scene_names.split(",")]
        if len(scene_names) != len(scenes):
            print(f"警告: 场景名称数量 ({len(scene_names)}) 与场景数量 ({len(scenes)}) 不匹配，使用默认名称")
            scene_names = [f"场景{i+1}" for i in range(len(scenes))]
    else:
        scene_names = [f"场景{i+1}" for i in range(len(scenes))]
    
    print("\n" + "=" * 100)
    print("  多场景批量对比")
    print("=" * 100)
    
    sentences = read_sentences_from_file(args.file)
    if not sentences:
        print(f"错误: 文件 {args.file} 中没有有效的句子")
        sys.exit(1)
    
    cli_dimensions = parse_dimensions(args.dimensions) if args.dimensions else None
    locked_words = parse_locked_words(args.locked)
    drifts = parse_drift(args.drift)
    word_score_overrides = parse_word_scores(args.word_scores)
    conflict_strategy = WordConflictStrategy(args.conflict_strategy)
    
    print(f"\n输入文件: {args.file}")
    print(f"句子数量: {len(sentences)}")
    print(f"传递次数: {args.steps}")
    print(f"场景数量: {len(scenes)}")
    for i, (name, path) in enumerate(zip(scene_names, scenes), 1):
        print(f"  {i}. {name}: {path}")
    print()
    
    all_results = {}
    
    for scene_name, lexicon_path in zip(scene_names, scenes):
        print(f"[{scene_name}] 正在模拟...")
        
        scene_args = argparse.Namespace(**vars(args))
        scene_args.lexicon = lexicon_path
        scene_args.no_merge_default = args.no_merge_default
        
        simulator = EmotionSimulator.from_lexicon_file(
            lexicon_filepath=lexicon_path,
            merge_default=not args.no_merge_default,
            change_probability=args.probability,
            magnitude=args.magnitude,
            random_seed=args.seed,
            conflict_strategy=conflict_strategy,
            word_score_overrides=word_score_overrides,
        )
        
        if cli_dimensions:
            existing_dim_names = {d.name for d in simulator.dimensions}
            for dim in cli_dimensions:
                if dim.name not in existing_dim_names:
                    simulator.add_custom_dimension(dim)
        
        for dim_name, direction in drifts.items():
            simulator.set_drift(dim_name, direction)
        
        results = simulator.batch_simulate(
            sentences=sentences,
            num_steps=args.steps,
            locked_words=locked_words,
        )
        
        all_results[scene_name] = (simulator, results)
    
    max_scene_info = []
    for i, sentence in enumerate(sentences):
        max_change = -1
        max_scene = None
        scene_details = {}
        for scene_name in scene_names:
            simulator, results = all_results[scene_name]
            result = results[i]
            total_change = 0
            max_dim = None
            max_dim_change = -1
            for dim in simulator.dimensions:
                series = result.get_emotion_series(dim.name)
                dim_change = abs(series[-1] - series[0])
                total_change += dim_change
                if dim_change > max_dim_change:
                    max_dim_change = dim_change
                    max_dim = dim.name
            scene_details[scene_name] = {
                'total_change': total_change,
                'max_dimension': max_dim,
                'max_dimension_change': max_dim_change,
            }
            if total_change > max_change:
                max_change = total_change
                max_scene = scene_name
        max_scene_info.append({
            'max_scene': max_scene,
            'max_change': max_change,
            'scene_details': scene_details,
        })
    
    print("\n" + "=" * 100)
    print("  多场景对比报告")
    print("=" * 100)
    print(f"每句子汇总（按场景分组，★表示影响最大的场景）")
    print("-" * 100)
    
    header = "{:<30}".format("句子")
    for name in scene_names:
        header += "{:<25}".format(name)
    print(header)
    print("-" * 100)
    
    for i, sentence in enumerate(sentences):
        row_text = sentence[:28] + "…" if len(sentence) > 28 else sentence
        row = "{:<30}".format(row_text)
        info = max_scene_info[i]
        
        for scene_name in scene_names:
            simulator, results = all_results[scene_name]
            result = results[i]
            series = result.get_emotion_series(simulator.dimensions[0].name)
            change = series[-1] - series[0]
            final_score = series[-1]
            
            dim = simulator.dimensions[0]
            label = dim.high_label if change > 0 else dim.low_label
            
            cell = "{:+.2f} (Δ{:+.2f})".format(final_score, change)
            if scene_name == info['max_scene']:
                cell = "★ " + cell
            else:
                cell = "  " + cell
            row += "{:<25}".format(cell)
        
        print(row)
        
        detail = info['scene_details'][info['max_scene']]
        print(f"  影响最大: {info['max_scene']} (总变化: {info['max_change']:+.2f}, 关键维度: {detail['max_dimension']} Δ{detail['max_dimension_change']:+.2f})")
    
    print("-" * 100)
    
    if args.output:
        data = {
            "scenes": scene_names,
            "scene_paths": scenes,
            "sentences": sentences,
            "results": {},
            "scene_analysis": [],
        }
        for i, sentence in enumerate(sentences):
            info = max_scene_info[i]
            analysis = {
                "sentence": sentence,
                "most_impactful_scene": info['max_scene'],
                "total_change": info['max_change'],
                "scene_details": {},
            }
            for scene_name in scene_names:
                detail = info['scene_details'][scene_name]
                analysis["scene_details"][scene_name] = {
                    "total_change": detail['total_change'],
                    "max_dimension": detail['max_dimension'],
                    "max_dimension_change": detail['max_dimension_change'],
                }
            data["scene_analysis"].append(analysis)
        
        for scene_name in scene_names:
            _, results = all_results[scene_name]
            data["results"][scene_name] = [JSONExporter._result_to_dict(r) for r in results]
        
        if args.output.lower().endswith(('.yaml', '.yml')):
            if not HAS_YAML:
                print("\n警告: PyYAML未安装，无法导出YAML格式。将使用JSON格式导出。")
                args.output = args.output.rsplit('.', 1)[0] + '.json'
            else:
                with open(args.output, "w", encoding="utf-8") as f:
                    yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
                print(f"\n多场景对比结果已导出到YAML文件: {args.output}")
                return
        
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\n多场景对比结果已导出到JSON文件: {args.output}")


def main() -> None:
    """主函数"""
    parser = create_parser()
    args = parser.parse_args()

    try:
        if args.export_template:
            print("\n" + "=" * 60)
            print("  导出词库配置模板")
            print("=" * 60)
            include_default = not args.no_default_lexicon
            export_lexicon_template(args.export_template, include_default=include_default)
            print(f"\n✓ 模板已导出到: {args.export_template}")
            print(f"\n提示: 编辑该文件配置您的词库，然后使用 --lexicon 参数加载")
            return
        
        if args.wizard:
            from .wizard import run_wizard
            run_wizard()
            return
        
        if args.list_profiles:
            list_profiles_cli()
            return
        
        if args.delete_profile:
            delete_profile_cli(args.delete_profile)
            return
        
        if not args.sentence and not args.file:
            parser.print_help()
            print("\n错误: 请指定输入句子（位置参数）或输入文件（-f/--file），或使用特殊命令（--export-template/--wizard/--list-profiles/--delete-profile）")
            sys.exit(1)
        
        if args.multi_scene:
            if not args.file:
                print("\n错误: 多场景对比需要指定输入文件（-f/--file）")
                sys.exit(1)
            run_multi_scene_batch(args)
            return
        
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
