"""
交互式配置向导模块
引导用户一步步创建词库配置文件
"""

import sys
import os
from typing import Dict, List, Tuple, Optional
from .models import (
    EmotionDimension,
    SynonymGroup,
    SynonymEntry,
    export_lexicon_to_file,
    create_default_neutral_words,
    DefaultDimensions,
)


def _input(prompt: str, default: Optional[str] = None) -> str:
    """安全的输入函数，支持默认值"""
    if default is not None:
        prompt = f"{prompt} [{default}]: "
    else:
        prompt = f"{prompt}: "
    
    try:
        value = input(prompt).strip()
        if not value and default is not None:
            return default
        return value
    except (EOFError, KeyboardInterrupt):
        print("\n\n操作已取消")
        sys.exit(0)


def _input_float(prompt: str, default: Optional[float] = None) -> float:
    """输入浮点数"""
    while True:
        value = _input(prompt, str(default) if default is not None else None)
        if not value and default is not None:
            return default
        try:
            return float(value)
        except ValueError:
            print("请输入有效的数字")


def _input_yes_no(prompt: str, default: bool = True) -> bool:
    """输入 yes/no"""
    default_str = "Y/n" if default else "y/N"
    while True:
        value = _input(f"{prompt} ({default_str})", None).lower()
        if not value:
            return default
        if value in ["y", "yes"]:
            return True
        if value in ["n", "no"]:
            return False
        print("请输入 y 或 n")


class LexiconWizard:
    """词库配置交互式向导"""

    def __init__(self):
        self.dimensions: List[EmotionDimension] = []
        self.synonym_groups: List[SynonymGroup] = []
        self.neutral_words: List[str] = []
        self.include_default_neutral: bool = True

    def run(self) -> Tuple[List[EmotionDimension], List[SynonymGroup], List[str]]:
        """运行向导"""
        self._print_header()
        self._configure_dimensions()
        self._configure_synonym_groups()
        self._configure_neutral_words()
        self._summary()
        return self.dimensions, self.synonym_groups, self.neutral_words

    def _print_header(self) -> None:
        """打印向导头部"""
        print("\n" + "=" * 60)
        print("  词库配置交互式向导")
        print("=" * 60)
        print("\n请按提示一步步配置您的词库。")
        print("按 Enter 可跳过使用默认值，按 Ctrl+C 可随时取消。\n")

    def _configure_dimensions(self) -> None:
        """配置情绪维度"""
        print("\n--- 步骤 1: 配置情绪维度 ---")
        print("先设置默认的3个维度，然后添加您的自定义维度。\n")
        
        if _input_yes_no("是否包含默认维度（积极-消极、愤怒-平静、激动-平静）", True):
            self.dimensions = DefaultDimensions.get_defaults()
            print(f"已添加 {len(self.dimensions)} 个默认维度")
        
        add_more = True
        while add_more:
            if _input_yes_no("是否添加自定义维度", len(self.dimensions) == 0):
                name = _input("维度英文名（如 work_pressure）", None)
                if not name:
                    continue
                low_label = _input("低分值标签（如 轻松）", "低")
                high_label = _input("高分值标签（如 压力大）", "高")
                min_val = _input_float("最小值", -5.0)
                max_val = _input_float("最大值", 5.0)
                
                dim = EmotionDimension(name, low_label, high_label, min_val, max_val)
                self.dimensions.append(dim)
                print(f"已添加维度: {name} ({low_label} <-> {high_label})")
            else:
                add_more = False
        
        if not self.dimensions:
            print("\n错误: 至少需要一个情绪维度！")
            self._configure_dimensions()
        else:
            print(f"\n当前维度列表 ({len(self.dimensions)} 个):")
            for i, dim in enumerate(self.dimensions, 1):
                print(f"  {i}. {dim.name}: {dim.low_label} <-> {dim.high_label}")

    def _configure_synonym_groups(self) -> None:
        """配置同义词组"""
        print("\n--- 步骤 2: 配置同义词组 ---")
        print("每组同义词包含一组语义相近但情绪强度不同的词。\n")
        
        add_more = True
        while add_more:
            if _input_yes_no("是否添加同义词组", True):
                keyword = _input("同义词组关键词（如 工作压力）", None)
                if not keyword:
                    continue
                
                entries = self._configure_entries(keyword)
                if entries:
                    group = SynonymGroup(keyword=keyword, entries=entries)
                    self.synonym_groups.append(group)
                    print(f"已添加同义词组: {keyword}（{len(entries)} 个词）")
            else:
                add_more = False
        
        if not self.synonym_groups:
            print("\n警告: 未添加任何同义词组，模拟器将无法进行情绪变化！")
        
        print(f"\n当前同义词组列表 ({len(self.synonym_groups)} 组):")
        for i, group in enumerate(self.synonym_groups, 1):
            words = ", ".join(e.word for e in group.entries)
            print(f"  {i}. {group.keyword}: {words}")

    def _configure_entries(self, group_keyword: str) -> List[SynonymEntry]:
        """配置同义词组的条目"""
        print(f"\n  配置同义词组 '{group_keyword}' 的词:")
        entries = []
        
        add_more = True
        while add_more:
            word = _input("  词（输入空结束）", None)
            if not word:
                break
            
            scores = {}
            for dim in self.dimensions:
                score = _input_float(f"    {dim.name} 分数 ({dim.low_label}:{dim.min_val} <-> {dim.high_label}:{dim.max_val})", 0.0)
                scores[dim.name] = score
            
            intensity = _input_float("    强度权重", 1.0)
            
            entry = SynonymEntry(word=word, emotion_scores=scores, intensity=intensity)
            entries.append(entry)
            print(f"    已添加: {word}")
            
            add_more = _input_yes_no("  继续添加词", True)
        
        if len(entries) < 2:
            print(f"  警告: 同义词组 '{group_keyword}' 只有 {len(entries)} 个词，建议至少2个词才能进行替换。")
        
        return entries

    def _configure_neutral_words(self) -> None:
        """配置中性词"""
        print("\n--- 步骤 3: 配置中性词 ---")
        print("中性词不参与情绪变化，也不影响评分。\n")
        
        self.include_default_neutral = _input_yes_no("是否包含默认中性词（今天、我、的、了 等）", True)
        
        if self.include_default_neutral:
            self.neutral_words = create_default_neutral_words()
            print(f"已包含 {len(self.neutral_words)} 个默认中性词")
        
        custom_neutral = _input("输入自定义中性词，用逗号分隔（可选）", "").strip()
        if custom_neutral:
            words = [w.strip() for w in custom_neutral.split(",") if w.strip()]
            self.neutral_words = list(set(self.neutral_words + words))
            print(f"已添加 {len(words)} 个自定义中性词")
        
        print(f"当前共有 {len(self.neutral_words)} 个中性词")

    def _summary(self) -> None:
        """显示配置摘要"""
        print("\n" + "=" * 60)
        print("  配置摘要")
        print("=" * 60)
        print(f"\n情绪维度: {len(self.dimensions)} 个")
        for dim in self.dimensions:
            print(f"  - {dim.name}: {dim.low_label} <-> {dim.high_label}")
        
        print(f"\n同义词组: {len(self.synonym_groups)} 组")
        for group in self.synonym_groups:
            words = ", ".join(e.word for e in group.entries)
            print(f"  - {group.keyword}: {words}")
        
        print(f"\n中性词: {len(self.neutral_words)} 个")

    def save(self, filepath: Optional[str] = None) -> str:
        """保存配置到文件"""
        if filepath is None:
            filepath = _input("\n请输入保存路径（如 work_lexicon.yaml 或 custom_lexicon.json）", "my_lexicon.json")
        
        if not os.path.splitext(filepath)[1]:
            filepath += ".json"
        
        try:
            export_lexicon_to_file(
                filepath,
                self.dimensions,
                self.synonym_groups,
                self.neutral_words,
            )
            print(f"\n✓ 词库配置已保存到: {filepath}")
            print(f"\n下次使用: python main.py \"句子\" --lexicon {filepath}")
            return filepath
        except Exception as e:
            print(f"\n✗ 保存失败: {e}")
            return self.save(None)


def run_wizard(output_file: Optional[str] = None) -> str:
    """
    运行交互式向导并保存配置
    
    Args:
        output_file: 输出文件路径，None则提示用户输入
    
    Returns:
        保存的文件路径
    """
    wizard = LexiconWizard()
    wizard.run()
    return wizard.save(output_file)
