#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本 - 验证情绪演进模拟器的所有功能
"""

import os
import sys
import tempfile
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from emotion_simulator.models import (
    EmotionDimension,
    SynonymGroup,
    SynonymEntry,
    DefaultDimensions,
    create_default_synonym_groups,
    load_lexicon_from_file,
    load_config_from_file,
)
from emotion_simulator.scorer import EmotionScorer
from emotion_simulator.simulator import EmotionSimulator
from emotion_simulator.visualizer import ASCIIVisualizer
from emotion_simulator.exporter import JSONExporter, BatchSummarizer, CSVExporter


def test_models():
    """测试数据模型"""
    print("=" * 60)
    print("测试1: 数据模型")
    print("=" * 60)

    dim = EmotionDimension("test", "低", "高", -5.0, 5.0)
    assert dim.name == "test"
    assert dim.low_label == "低"
    assert dim.high_label == "高"
    assert dim.normalize(0.0) == 0.5
    assert dim.normalize(5.0) == 1.0
    assert dim.normalize(-5.0) == 0.0
    print("✓ EmotionDimension 测试通过")

    entry = SynonymEntry("test_word", {"positive_negative": 2.5}, 1.0)
    assert entry.word == "test_word"
    assert entry.get_score("positive_negative") == 2.5
    assert entry.get_score("unknown") == 0.0
    print("✓ SynonymEntry 测试通过")

    group = SynonymGroup(
        keyword="test",
        entries=[
            SynonymEntry("bad", {"positive_negative": -3.0}),
            SynonymEntry("ok", {"positive_negative": 0.0}),
            SynonymEntry("good", {"positive_negative": 3.0}),
        ]
    )
    assert group.get_entry("bad") is not None
    assert group.get_entry("nonexistent") is None
    sorted_entries = group.get_sorted_by_dimension("positive_negative")
    assert sorted_entries[0].word == "bad"
    assert sorted_entries[-1].word == "good"
    print("✓ SynonymGroup 测试通过")

    default_groups = create_default_synonym_groups()
    assert len(default_groups) == 10
    print(f"✓ 默认同义词库加载成功，共 {len(default_groups)} 组")

    default_dims = DefaultDimensions.get_defaults()
    assert len(default_dims) == 3
    print(f"✓ 默认情绪维度加载成功，共 {len(default_dims)} 个维度")

    print("\n数据模型测试全部通过！\n")


def test_scorer():
    """测试情绪评分器"""
    print("=" * 60)
    print("测试2: 情绪评分器")
    print("=" * 60)

    dimensions = DefaultDimensions.get_defaults()
    scorer = EmotionScorer(dimensions)

    scores = scorer.score_sentence("今天天气不错")
    assert "positive_negative" in scores
    assert "anger_calm" in scores
    assert "excitement_calm" in scores
    print(f"✓ 句子评分成功: {scores}")

    replacable = scorer.find_replacable_words("今天天气不错", [])
    assert len(replacable) > 0
    print(f"✓ 找到可替换词: {[(pos, word) for pos, word, _ in replacable]}")

    replacable_locked = scorer.find_replacable_words("今天天气不错", ["不错"])
    assert all(word != "不错" for _, word, _ in replacable_locked)
    print("✓ 关键词锁定功能正常")

    word_entry = scorer.get_word_entry("不错")
    assert word_entry is not None
    assert word_entry.word == "不错"
    print(f"✓ 词条目查询成功: {word_entry.word}")

    group = scorer.get_synonym_group("还行")
    assert group is not None
    assert group.keyword == "不错"
    print(f"✓ 同义词组查询成功: {group.keyword}")

    custom_dim = EmotionDimension("custom", "差", "优")
    scorer.add_custom_dimension(custom_dim)
    assert custom_dim in scorer.dimensions
    print("✓ 自定义维度添加成功")

    custom_group = SynonymGroup(
        keyword="测试",
        entries=[
            SynonymEntry("测试词1", {"positive_negative": 1.0}),
            SynonymEntry("测试词2", {"positive_negative": -1.0}),
        ]
    )
    scorer.add_synonym_group(custom_group)
    assert scorer.get_synonym_group("测试词1") is not None
    print("✓ 自定义同义词组添加成功")

    print("\n情绪评分器测试全部通过！\n")


def test_simulator():
    """测试情绪演进模拟器"""
    print("=" * 60)
    print("测试3: 情绪演进模拟器")
    print("=" * 60)

    simulator = EmotionSimulator(
        change_probability=1.0,
        magnitude=0.5,
        random_seed=42,
    )

    new_sentence, changed, scores = simulator.transmit("今天天气不错")
    print(f"✓ 单次传递完成")
    print(f"  原句: 今天天气不错")
    print(f"  新句: {new_sentence}")
    print(f"  变化: {changed}")
    print(f"  分数: {scores}")

    result = simulator.simulate(
        initial_sentence="今天天气不错",
        num_steps=5,
        locked_words=[],
    )
    assert len(result.steps) == 5
    assert result.initial_sentence == "今天天气不错"
    print(f"\n✓ 5步模拟完成")
    print(f"  初始句子: {result.initial_sentence}")
    print(f"  最终句子: {result.get_final_sentence()}")
    for step in result.steps:
        if step.changed_word:
            print(f"  步骤 {step.step}: {step.changed_word[0]} → {step.changed_word[1]}")

    locked_result = simulator.simulate(
        initial_sentence="今天天气不错",
        num_steps=10,
        locked_words=["不错"],
    )
    for step in locked_result.steps:
        if step.changed_word:
            assert step.changed_word[0] != "不错"
    print("\n✓ 锁定关键词模拟正常，'不错'未被替换")

    simulator.set_drift("positive_negative", 0.9)
    drift_result = simulator.simulate(
        initial_sentence="今天天气不错",
        num_steps=10,
    )
    series = drift_result.get_emotion_series("positive_negative")
    if series:
        trend = series[-1] - series[0]
        print(f"\n✓ 情绪漂移模拟完成，积极趋势: {trend:+.2f}")

    sentences = ["今天天气不错", "我喜欢这个", "他很生气"]
    batch_results = simulator.batch_simulate(sentences, num_steps=5)
    assert len(batch_results) == 3
    print(f"\n✓ 批量模拟完成，共 {len(batch_results)} 个结果")

    print("\n情绪演进模拟器测试全部通过！\n")


def test_visualizer():
    """测试ASCII可视化器"""
    print("=" * 60)
    print("测试4: ASCII可视化器")
    print("=" * 60)

    simulator = EmotionSimulator(random_seed=42)
    result = simulator.simulate("今天天气不错", num_steps=10)

    visualizer = ASCIIVisualizer(width=60, height=10)

    single_chart = visualizer.plot_single(result, "positive_negative", use_color=False)
    assert "情绪演进曲线" in single_chart
    print("✓ 单维度图表生成成功")

    comparison_chart = visualizer.plot_comparison(
        [result, result],
        "positive_negative",
        labels=["测试1", "测试2"],
        use_color=False,
    )
    assert "情绪演进对比" in comparison_chart
    print("✓ 对比图表生成成功")

    multi_chart = visualizer.plot_multi_dimension(result, use_color=False)
    assert "多维度情绪演进曲线" in multi_chart
    print("✓ 多维度图表生成成功")

    step_info = visualizer.print_step_info(result.steps[0], use_color=False)
    assert "第 1 次传递" in step_info
    print("✓ 单步信息格式化成功")

    print("\nASCII可视化器测试全部通过！\n")


def test_exporter():
    """测试JSON导出器"""
    print("=" * 60)
    print("测试5: JSON导出器")
    print("=" * 60)

    simulator = EmotionSimulator(random_seed=42)
    result = simulator.simulate("今天天气不错", num_steps=5)

    json_str = JSONExporter.to_string(result, indent=2)
    assert "initial_sentence" in json_str
    assert "final_sentence" in json_str
    assert "steps" in json_str
    data = json.loads(json_str)
    assert data["initial_sentence"] == "今天天气不错"
    assert len(data["steps"]) == 5
    print("✓ JSON字符串导出成功")

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        temp_path = f.name

    try:
        JSONExporter.export(result, temp_path)
        assert os.path.exists(temp_path)
        with open(temp_path, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        assert loaded_data["initial_sentence"] == "今天天气不错"
        print("✓ JSON文件导出成功")

        result2 = simulator.simulate("我喜欢这个", num_steps=5)
        JSONExporter.export_batch([result, result2], temp_path)
        with open(temp_path, 'r', encoding='utf-8') as f:
            batch_data = json.load(f)
        assert batch_data["count"] == 2
        assert len(batch_data["results"]) == 2
        print("✓ 批量JSON导出成功")

        batch_str = JSONExporter.batch_to_string([result, result2])
        batch_data2 = json.loads(batch_str)
        assert batch_data2["count"] == 2
        print("✓ 批量JSON字符串导出成功")
    finally:
        os.unlink(temp_path)

    print("\nJSON导出器测试全部通过！\n")


def test_integration():
    """集成测试"""
    print("=" * 60)
    print("测试6: 集成测试 - 完整流程")
    print("=" * 60)

    custom_dims = [
        EmotionDimension("满意度", "不满意", "满意"),
        EmotionDimension("情绪强度", "平静", "强烈"),
    ]

    simulator = EmotionSimulator(
        dimensions=custom_dims,
        change_probability=0.7,
        magnitude=0.6,
        random_seed=123,
    )

    simulator.set_drift("满意度", 0.5)

    result = simulator.simulate(
        initial_sentence="今天天气不错，我很高兴",
        num_steps=15,
        locked_words=["今天"],
    )

    print(f"初始句子: {result.initial_sentence}")
    print(f"最终句子: {result.get_final_sentence()}")
    print(f"总步数: {len(result.steps)}")

    changes = sum(1 for s in result.steps if s.changed_word)
    print(f"变化次数: {changes}")

    for dim in custom_dims:
        series = result.get_emotion_series(dim.name)
        if series:
            print(f"{dim.name}: {series[0]:+.2f} → {series[-1]:+.2f} (Δ{series[-1]-series[0]:+.2f})")

    visualizer = ASCIIVisualizer(width=70, height=12)
    chart = visualizer.plot_single(result, "满意度", use_color=False)
    print("\n满意度情绪演进曲线:")
    print(chart)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        temp_path = f.name

    try:
        JSONExporter.export(result, temp_path)
        print(f"\n结果已导出到临时文件: {temp_path}")
    finally:
        os.unlink(temp_path)

    print("\n集成测试通过！\n")


def test_longest_matching():
    """测试最长匹配评分算法"""
    print("=" * 60)
    print("测试7: 最长匹配评分算法")
    print("=" * 60)

    dimensions = DefaultDimensions.get_defaults()
    scorer = EmotionScorer(dimensions)

    test_sentence = "有点累，今天工作很好"
    details = scorer.get_score_details(test_sentence)

    print(f"测试句子: {test_sentence}")
    print(f"匹配到 {len(details['matched_words'])} 个词:")

    matched_words = [m['word'] for m in details['matched_words']]
    print(f"  匹配词: {matched_words}")

    assert '有点累' in matched_words, "应该匹配'有点累'而不是'累'"
    assert '累' not in matched_words, "不应该匹配'累'，因为已经匹配了更长的'有点累'"
    assert '很好' in matched_words, "应该匹配'很好'"
    assert '好' not in matched_words, "不应该匹配'好'，因为已经匹配了更长的'很好'"

    print("✓ 最长匹配算法正确：'有点累' > '累', '很好' > '好'")

    old_score = scorer.score_sentence(test_sentence)
    print(f"✓ 改进后评分: {old_score}")

    print("\n最长匹配评分算法测试通过！\n")


def test_lexicon_file():
    """测试外部词库配置文件加载"""
    print("=" * 60)
    print("测试8: 外部词库配置文件")
    print("=" * 60)

    lexicon_path = os.path.join(os.path.dirname(__file__), "work_lexicon.json")

    assert os.path.exists(lexicon_path), f"词库文件不存在: {lexicon_path}"

    config = load_config_from_file(lexicon_path)
    assert 'dimensions' in config
    assert 'synonym_groups' in config
    print(f"✓ 配置文件加载成功，包含 {len(config['dimensions'])} 个维度和 {len(config['synonym_groups'])} 组词")

    dimensions, groups, neutral_words = load_lexicon_from_file(lexicon_path, merge_default=False)
    assert len(dimensions) == 3
    assert len(groups) == 4
    dim_names = [d.name for d in dimensions]
    assert 'work_pressure' in dim_names
    assert 'satisfaction' in dim_names
    assert 'tension' in dim_names
    print(f"✓ 词库加载成功（不合并默认）: {len(dimensions)} 维度, {len(groups)} 同义词组")

    dimensions_all, groups_all, neutral_all = load_lexicon_from_file(lexicon_path, merge_default=True)
    assert len(dimensions_all) >= 6
    assert len(groups_all) >= 4
    print(f"✓ 词库加载成功（合并默认）: {len(dimensions_all)} 维度, {len(groups_all)} 同义词组")

    simulator = EmotionSimulator.from_lexicon_file(
        lexicon_filepath=lexicon_path,
        merge_default=True,
        random_seed=42,
    )

    dim_names = [d.name for d in simulator.dimensions]
    assert 'work_pressure' in dim_names
    assert 'satisfaction' in dim_names
    assert 'tension' in dim_names
    assert 'positive_negative' in dim_names
    print(f"✓ 模拟器从词库创建成功，维度: {dim_names}")

    test_sentence = "今天工作有点累，任务很困难"
    scores = simulator.scorer.score_sentence(test_sentence)
    print(f"\n测试句子: {test_sentence}")
    print(f"各维度分数:")
    for dim_name, score in scores.items():
        print(f"  {dim_name}: {score:+.2f}")

    assert scores['work_pressure'] > 0, "工作压力维度应该有分数"
    assert scores['satisfaction'] < 0, "满意度维度应该有分数"
    assert scores['tension'] > 0, "紧张度维度应该有分数"

    print("\n外部词库配置文件测试通过！\n")


def test_batch_summary():
    """测试批量汇总和CSV导出"""
    print("=" * 60)
    print("测试9: 批量汇总和CSV导出")
    print("=" * 60)

    simulator = EmotionSimulator(random_seed=42)
    sentences = [
        "今天天气不错",
        "我很生气",
        "工作有点累",
        "这个很好",
        "他很高兴",
    ]
    results = simulator.batch_simulate(sentences, num_steps=8)

    metrics = BatchSummarizer.get_result_metrics(results[0])
    assert 'initial_sentence' in metrics
    assert 'changes_count' in metrics
    assert 'positive_negative_change' in metrics
    print(f"✓ 单个结果指标获取成功: {list(metrics.keys())[:10]}...")

    summaries = BatchSummarizer.summarize(results)
    assert len(summaries) == len(results)
    print(f"✓ 汇总成功，共 {len(summaries)} 条记录")

    sorted_by_change = BatchSummarizer.summarize(
        results,
        sort_by='positive_negative_change_abs',
        sort_ascending=False
    )
    changes = [s['positive_negative_change_abs'] for s in sorted_by_change]
    assert changes == sorted(changes, reverse=True), "应该按变化幅度降序排列"
    print(f"✓ 按变化幅度降序排列成功")

    sorted_by_count = BatchSummarizer.summarize(
        results,
        sort_by='changes_count',
        sort_ascending=True
    )
    counts = [s['changes_count'] for s in sorted_by_count]
    assert counts == sorted(counts), "应该按变化次数升序排列"
    print(f"✓ 按变化次数升序排列成功")

    summary_output = BatchSummarizer.print_summary(results, use_color=False)
    assert '批量模拟结果汇总' in summary_output
    assert '初始句子' in summary_output
    assert '最终句子' in summary_output
    print("✓ 终端汇总输出成功")

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        temp_json = f.name
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
        temp_csv = f.name
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
        temp_ts_csv = f.name

    try:
        BatchSummarizer.export_summary_json(results, temp_json, sort_by='changes_count')
        assert os.path.exists(temp_json)
        with open(temp_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        assert data['count'] == len(results)
        assert data['sort_by'] == 'changes_count'
        print("✓ 汇总JSON导出成功")

        BatchSummarizer.export_summary_csv(results, temp_csv)
        assert os.path.exists(temp_csv)
        print("✓ 汇总CSV导出成功")

        CSVExporter.export_timeseries(results[0], temp_ts_csv)
        assert os.path.exists(temp_ts_csv)
        print("✓ 时间序列CSV导出成功")

    finally:
        for f in [temp_json, temp_csv, temp_ts_csv]:
            if os.path.exists(f):
                os.unlink(f)

    print("\n批量汇总和CSV导出测试通过！\n")


def test_custom_dimension_with_lexicon():
    """测试自定义维度配合词库使用"""
    print("=" * 60)
    print("测试10: 自定义维度配合词库")
    print("=" * 60)

    lexicon_path = os.path.join(os.path.dirname(__file__), "work_lexicon.json")

    custom_dims = [
        EmotionDimension("work_pressure", "轻松", "压力大"),
        EmotionDimension("satisfaction", "不满", "满意"),
        EmotionDimension("tension", "放松", "紧张"),
    ]

    custom_groups = [
        SynonymGroup(
            keyword="工作状态",
            entries=[
                SynonymEntry("摸鱼", {
                    "work_pressure": -4.0,
                    "satisfaction": 3.0,
                    "tension": -3.5,
                    "positive_negative": 2.5,
                }, 1.2),
                SynonymEntry("划水", {
                    "work_pressure": -3.0,
                    "satisfaction": 2.5,
                    "tension": -2.5,
                    "positive_negative": 2.0,
                }, 1.1),
                SynonymEntry("正常工作", {
                    "work_pressure": 0.0,
                    "satisfaction": 0.0,
                    "tension": 0.0,
                    "positive_negative": 0.0,
                }, 1.0),
                SynonymEntry("加班", {
                    "work_pressure": 3.0,
                    "satisfaction": -2.5,
                    "tension": 2.5,
                    "positive_negative": -2.0,
                }, 1.2),
                SynonymEntry("996", {
                    "work_pressure": 4.5,
                    "satisfaction": -4.0,
                    "tension": 4.0,
                    "positive_negative": -3.5,
                }, 1.5),
            ]
        )
    ]

    simulator = EmotionSimulator(
        dimensions=custom_dims,
        synonym_groups=custom_groups,
        change_probability=0.7,
        magnitude=0.6,
        random_seed=123,
    )

    test_sentence = "今天又要加班，感觉很累"
    scores = simulator.scorer.score_sentence(test_sentence)
    print(f"测试句子: {test_sentence}")
    print(f"各维度分数:")
    for dim_name, score in scores.items():
        print(f"  {dim_name}: {score:+.2f}")

    assert scores['work_pressure'] > 2.0, "工作压力分数应该很高"
    assert scores['satisfaction'] < -2.0, "满意度分数应该很低"
    assert scores['tension'] > 2.0, "紧张度分数应该很高"

    print("\n✓ 自定义维度配合词库评分正确")

    result = simulator.simulate(
        initial_sentence="今天工作很轻松，我很满意",
        num_steps=10,
    )

    print(f"\n初始句子: {result.initial_sentence}")
    print(f"最终句子: {result.get_final_sentence()}")

    if result.steps:
        for dim in custom_dims:
            series = result.get_emotion_series(dim.name)
            if series:
                print(f"  {dim.name}: {series[0]:+.2f} → {series[-1]:+.2f} (Δ{series[-1]-series[0]:+.2f})")

    print("\n自定义维度配合词库测试通过！\n")


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("文本情绪演进模拟器 - 功能测试")
    print("=" * 60)

    try:
        test_models()
        test_scorer()
        test_simulator()
        test_visualizer()
        test_exporter()
        test_integration()
        test_longest_matching()
        test_lexicon_file()
        test_batch_summary()
        test_custom_dimension_with_lexicon()

        print("=" * 60)
        print("✓ 所有测试通过！")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"\n✗ 断言失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n✗ 测试出错: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
