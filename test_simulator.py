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
)
from emotion_simulator.scorer import EmotionScorer
from emotion_simulator.simulator import EmotionSimulator
from emotion_simulator.visualizer import ASCIIVisualizer
from emotion_simulator.exporter import JSONExporter


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
