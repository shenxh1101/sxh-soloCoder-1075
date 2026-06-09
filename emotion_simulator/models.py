"""
数据模型模块
定义情绪维度、同义词库、传递记录等核心数据结构
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum
import json
import os

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


class WordConflictStrategy(Enum):
    """词冲突处理策略"""
    KEEP_FIRST = "keep_first"
    KEEP_LAST = "keep_last"
    ERROR = "error"
    MERGE = "merge"


class EmotionDimension:
    """
    情绪维度类
    定义一个情绪维度，如积极-消极、愤怒-平静
    """

    def __init__(self, name: str, low_label: str, high_label: str, min_val: float = -5.0, max_val: float = 5.0):
        self.name = name
        self.low_label = low_label
        self.high_label = high_label
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, value: float) -> float:
        """将值归一化到[0, 1]区间"""
        return (value - self.min_val) / (self.max_val - self.min_val)

    def __repr__(self) -> str:
        return f"EmotionDimension({self.name}: {self.low_label} <-> {self.high_label})"


@dataclass
class SynonymEntry:
    """
    同义词条目
    包含一个词及其在各个情绪维度上的强度值
    """
    word: str
    emotion_scores: Dict[str, float]
    intensity: float = 1.0

    def get_score(self, dimension: str) -> float:
        """获取指定情绪维度的分数"""
        return self.emotion_scores.get(dimension, 0.0)


@dataclass
class SynonymGroup:
    """
    同义词组
    包含一组语义相近但情绪强度不同的词
    """
    keyword: str
    entries: List[SynonymEntry]

    def get_entry(self, word: str) -> Optional[SynonymEntry]:
        """根据词获取对应的条目"""
        for entry in self.entries:
            if entry.word == word:
                return entry
        return None

    def get_sorted_by_dimension(self, dimension: str) -> List[SynonymEntry]:
        """根据指定情绪维度排序"""
        return sorted(self.entries, key=lambda e: e.get_score(dimension))

    def get_random_entry(self, exclude_word: Optional[str] = None) -> SynonymEntry:
        """随机获取一个条目，可排除指定词"""
        import random
        candidates = [e for e in self.entries if e.word != exclude_word]
        return random.choice(candidates) if candidates else self.entries[0]


@dataclass
class TransmissionStep:
    """
    单次传递记录
    """
    step: int
    original_sentence: str
    modified_sentence: str
    emotion_scores: Dict[str, float]
    changed_word: Optional[Tuple[str, str]] = None
    locked_words: List[str] = field(default_factory=list)


@dataclass
class SimulationResult:
    """
    模拟结果
    包含完整的演进过程
    """
    initial_sentence: str
    steps: List[TransmissionStep] = field(default_factory=list)
    locked_words: List[str] = field(default_factory=list)
    dimensions: List[EmotionDimension] = field(default_factory=list)

    def get_emotion_series(self, dimension: str) -> List[float]:
        """获取指定情绪维度的时间序列"""
        return [step.emotion_scores.get(dimension, 0.0) for step in self.steps]

    def get_final_sentence(self) -> str:
        """获取最终句子"""
        return self.steps[-1].modified_sentence if self.steps else self.initial_sentence


class DefaultDimensions:
    """
    默认情绪维度定义
    """
    POSITIVE_NEGATIVE = EmotionDimension(
        name="positive_negative",
        low_label="消极",
        high_label="积极",
        min_val=-5.0,
        max_val=5.0
    )

    ANGER_CALM = EmotionDimension(
        name="anger_calm",
        low_label="愤怒",
        high_label="平静",
        min_val=-5.0,
        max_val=5.0
    )

    EXCITEMENT_CALM = EmotionDimension(
        name="excitement_calm",
        low_label="平静",
        high_label="激动",
        min_val=-5.0,
        max_val=5.0
    )

    @classmethod
    def get_defaults(cls) -> List[EmotionDimension]:
        """获取默认的情绪维度列表"""
        return [cls.POSITIVE_NEGATIVE, cls.ANGER_CALM, cls.EXCITEMENT_CALM]


def create_default_synonym_groups() -> List[SynonymGroup]:
    """
    创建默认的中文同义词库
    包含常见词汇及其情绪强度
    """
    groups = []

    groups.append(SynonymGroup(
        keyword="不错",
        entries=[
            SynonymEntry("糟糕", {"positive_negative": -4.0, "anger_calm": -2.0, "excitement_calm": -1.0}, 1.2),
            SynonymEntry("不好", {"positive_negative": -2.5, "anger_calm": -1.0, "excitement_calm": -0.5}, 1.0),
            SynonymEntry("一般", {"positive_negative": -0.5, "anger_calm": 0.0, "excitement_calm": -0.5}, 0.8),
            SynonymEntry("还行", {"positive_negative": 1.0, "anger_calm": 1.0, "excitement_calm": 0.0}, 0.9),
            SynonymEntry("不错", {"positive_negative": 2.5, "anger_calm": 1.5, "excitement_calm": 1.0}, 1.0),
            SynonymEntry("很好", {"positive_negative": 3.5, "anger_calm": 2.0, "excitement_calm": 2.0}, 1.1),
            SynonymEntry("非常好", {"positive_negative": 4.5, "anger_calm": 2.5, "excitement_calm": 3.0}, 1.3),
            SynonymEntry("完美", {"positive_negative": 5.0, "anger_calm": 3.0, "excitement_calm": 4.0}, 1.5),
        ]
    ))

    groups.append(SynonymGroup(
        keyword="高兴",
        entries=[
            SynonymEntry("悲伤", {"positive_negative": -4.5, "anger_calm": -1.0, "excitement_calm": -3.0}, 1.3),
            SynonymEntry("难过", {"positive_negative": -3.0, "anger_calm": -0.5, "excitement_calm": -2.0}, 1.1),
            SynonymEntry("郁闷", {"positive_negative": -1.5, "anger_calm": -1.0, "excitement_calm": -1.0}, 1.0),
            SynonymEntry("平静", {"positive_negative": 0.0, "anger_calm": 3.0, "excitement_calm": -2.0}, 0.9),
            SynonymEntry("开心", {"positive_negative": 2.0, "anger_calm": 1.0, "excitement_calm": 1.5}, 1.0),
            SynonymEntry("高兴", {"positive_negative": 3.0, "anger_calm": 1.5, "excitement_calm": 2.5}, 1.1),
            SynonymEntry("兴奋", {"positive_negative": 4.0, "anger_calm": 0.5, "excitement_calm": 4.0}, 1.3),
            SynonymEntry("狂喜", {"positive_negative": 5.0, "anger_calm": -0.5, "excitement_calm": 5.0}, 1.5),
        ]
    ))

    groups.append(SynonymGroup(
        keyword="生气",
        entries=[
            SynonymEntry("狂喜", {"positive_negative": 5.0, "anger_calm": -0.5, "excitement_calm": 5.0}, 1.5),
            SynonymEntry("开心", {"positive_negative": 2.0, "anger_calm": 1.0, "excitement_calm": 1.5}, 1.0),
            SynonymEntry("平静", {"positive_negative": 0.0, "anger_calm": 3.0, "excitement_calm": -2.0}, 0.9),
            SynonymEntry("不满", {"positive_negative": -1.0, "anger_calm": -1.5, "excitement_calm": 0.5}, 0.9),
            SynonymEntry("生气", {"positive_negative": -2.0, "anger_calm": -3.0, "excitement_calm": 1.0}, 1.1),
            SynonymEntry("愤怒", {"positive_negative": -3.5, "anger_calm": -4.5, "excitement_calm": 2.5}, 1.3),
            SynonymEntry("暴怒", {"positive_negative": -4.5, "anger_calm": -5.0, "excitement_calm": 4.0}, 1.5),
            SynonymEntry("抓狂", {"positive_negative": -5.0, "anger_calm": -5.0, "excitement_calm": 5.0}, 1.5),
        ]
    ))

    groups.append(SynonymGroup(
        keyword="喜欢",
        entries=[
            SynonymEntry("痛恨", {"positive_negative": -5.0, "anger_calm": -4.0, "excitement_calm": 2.0}, 1.5),
            SynonymEntry("讨厌", {"positive_negative": -3.5, "anger_calm": -2.0, "excitement_calm": 0.5}, 1.2),
            SynonymEntry("反感", {"positive_negative": -2.0, "anger_calm": -1.0, "excitement_calm": 0.0}, 1.0),
            SynonymEntry("无感", {"positive_negative": 0.0, "anger_calm": 1.0, "excitement_calm": -0.5}, 0.8),
            SynonymEntry("喜欢", {"positive_negative": 2.5, "anger_calm": 1.5, "excitement_calm": 1.5}, 1.0),
            SynonymEntry("喜爱", {"positive_negative": 3.5, "anger_calm": 2.0, "excitement_calm": 2.5}, 1.2),
            SynonymEntry("热爱", {"positive_negative": 4.5, "anger_calm": 2.5, "excitement_calm": 3.5}, 1.4),
            SynonymEntry("挚爱", {"positive_negative": 5.0, "anger_calm": 3.0, "excitement_calm": 4.0}, 1.5),
        ]
    ))

    groups.append(SynonymGroup(
        keyword="快",
        entries=[
            SynonymEntry("极慢", {"positive_negative": -2.0, "anger_calm": 1.0, "excitement_calm": -4.0}, 1.3),
            SynonymEntry("很慢", {"positive_negative": -1.5, "anger_calm": 1.5, "excitement_calm": -3.0}, 1.1),
            SynonymEntry("较慢", {"positive_negative": -0.5, "anger_calm": 2.0, "excitement_calm": -1.5}, 0.9),
            SynonymEntry("适中", {"positive_negative": 0.5, "anger_calm": 2.5, "excitement_calm": 0.0}, 0.8),
            SynonymEntry("较快", {"positive_negative": 1.0, "anger_calm": 1.5, "excitement_calm": 1.5}, 0.9),
            SynonymEntry("快", {"positive_negative": 1.5, "anger_calm": 0.5, "excitement_calm": 2.5}, 1.0),
            SynonymEntry("很快", {"positive_negative": 2.0, "anger_calm": -0.5, "excitement_calm": 3.5}, 1.2),
            SynonymEntry("飞快", {"positive_negative": 2.5, "anger_calm": -1.5, "excitement_calm": 4.5}, 1.4),
        ]
    ))

    groups.append(SynonymGroup(
        keyword="热",
        entries=[
            SynonymEntry("严寒", {"positive_negative": -3.0, "anger_calm": 0.0, "excitement_calm": -3.0}, 1.5),
            SynonymEntry("寒冷", {"positive_negative": -2.0, "anger_calm": 0.5, "excitement_calm": -2.0}, 1.3),
            SynonymEntry("凉爽", {"positive_negative": 1.0, "anger_calm": 2.5, "excitement_calm": -1.0}, 1.0),
            SynonymEntry("温暖", {"positive_negative": 2.0, "anger_calm": 2.0, "excitement_calm": 0.0}, 1.0),
            SynonymEntry("热", {"positive_negative": 0.5, "anger_calm": -0.5, "excitement_calm": 1.0}, 1.0),
            SynonymEntry("炎热", {"positive_negative": -1.0, "anger_calm": -2.0, "excitement_calm": 2.0}, 1.2),
            SynonymEntry("酷热", {"positive_negative": -2.5, "anger_calm": -3.5, "excitement_calm": 3.5}, 1.4),
            SynonymEntry("炙热", {"positive_negative": -3.5, "anger_calm": -4.5, "excitement_calm": 4.5}, 1.5),
        ]
    ))

    groups.append(SynonymGroup(
        keyword="大",
        entries=[
            SynonymEntry("极小", {"positive_negative": -1.0, "anger_calm": 1.0, "excitement_calm": -3.0}, 1.3),
            SynonymEntry("很小", {"positive_negative": -0.5, "anger_calm": 1.5, "excitement_calm": -2.0}, 1.1),
            SynonymEntry("较小", {"positive_negative": 0.0, "anger_calm": 2.0, "excitement_calm": -1.0}, 0.9),
            SynonymEntry("适中", {"positive_negative": 0.5, "anger_calm": 2.5, "excitement_calm": 0.0}, 0.8),
            SynonymEntry("较大", {"positive_negative": 1.0, "anger_calm": 1.5, "excitement_calm": 1.0}, 0.9),
            SynonymEntry("大", {"positive_negative": 1.5, "anger_calm": 0.5, "excitement_calm": 2.0}, 1.0),
            SynonymEntry("很大", {"positive_negative": 2.5, "anger_calm": -0.5, "excitement_calm": 3.0}, 1.2),
            SynonymEntry("巨大", {"positive_negative": 3.5, "anger_calm": -1.5, "excitement_calm": 4.0}, 1.4),
        ]
    ))

    groups.append(SynonymGroup(
        keyword="好",
        entries=[
            SynonymEntry("极差", {"positive_negative": -5.0, "anger_calm": -3.0, "excitement_calm": -2.0}, 1.5),
            SynonymEntry("很差", {"positive_negative": -3.5, "anger_calm": -1.5, "excitement_calm": -1.0}, 1.3),
            SynonymEntry("较差", {"positive_negative": -1.5, "anger_calm": 0.0, "excitement_calm": -0.5}, 1.0),
            SynonymEntry("普通", {"positive_negative": 0.0, "anger_calm": 1.5, "excitement_calm": 0.0}, 0.8),
            SynonymEntry("较好", {"positive_negative": 1.5, "anger_calm": 1.5, "excitement_calm": 0.5}, 0.9),
            SynonymEntry("好", {"positive_negative": 2.5, "anger_calm": 1.0, "excitement_calm": 1.0}, 1.0),
            SynonymEntry("很好", {"positive_negative": 4.0, "anger_calm": 0.5, "excitement_calm": 2.0}, 1.2),
            SynonymEntry("极好", {"positive_negative": 5.0, "anger_calm": 0.0, "excitement_calm": 3.0}, 1.4),
        ]
    ))

    groups.append(SynonymGroup(
        keyword="想",
        entries=[
            SynonymEntry("绝不", {"positive_negative": -3.0, "anger_calm": -2.0, "excitement_calm": -1.0}, 1.3),
            SynonymEntry("不想", {"positive_negative": -2.0, "anger_calm": -1.0, "excitement_calm": -0.5}, 1.1),
            SynonymEntry("不太想", {"positive_negative": -0.5, "anger_calm": 0.0, "excitement_calm": 0.0}, 0.9),
            SynonymEntry("随便", {"positive_negative": 0.0, "anger_calm": 2.0, "excitement_calm": -0.5}, 0.8),
            SynonymEntry("有点想", {"positive_negative": 1.0, "anger_calm": 1.5, "excitement_calm": 0.5}, 0.9),
            SynonymEntry("想", {"positive_negative": 2.0, "anger_calm": 1.0, "excitement_calm": 1.5}, 1.0),
            SynonymEntry("很想", {"positive_negative": 3.0, "anger_calm": 0.0, "excitement_calm": 2.5}, 1.2),
            SynonymEntry("非常想", {"positive_negative": 4.0, "anger_calm": -1.0, "excitement_calm": 3.5}, 1.4),
        ]
    ))

    groups.append(SynonymGroup(
        keyword="累",
        entries=[
            SynonymEntry("精力充沛", {"positive_negative": 3.5, "anger_calm": 2.0, "excitement_calm": 3.0}, 1.4),
            SynonymEntry("精神", {"positive_negative": 2.5, "anger_calm": 2.0, "excitement_calm": 2.0}, 1.2),
            SynonymEntry("轻松", {"positive_negative": 2.0, "anger_calm": 2.5, "excitement_calm": 1.0}, 1.0),
            SynonymEntry("还好", {"positive_negative": 0.5, "anger_calm": 1.5, "excitement_calm": 0.0}, 0.9),
            SynonymEntry("有点累", {"positive_negative": -0.5, "anger_calm": 0.5, "excitement_calm": -1.0}, 0.9),
            SynonymEntry("累", {"positive_negative": -1.5, "anger_calm": 0.0, "excitement_calm": -2.0}, 1.0),
            SynonymEntry("很累", {"positive_negative": -3.0, "anger_calm": -1.0, "excitement_calm": -3.0}, 1.2),
            SynonymEntry("疲惫不堪", {"positive_negative": -4.5, "anger_calm": -2.0, "excitement_calm": -4.5}, 1.5),
        ]
    ))

    return groups


def create_default_neutral_words() -> List[str]:
    """
    创建默认的中性词列表（不参与情绪变化）
    """
    return [
        "今天", "明天", "昨天", "天气", "我", "你", "他", "她", "它",
        "我们", "你们", "他们", "的", "了", "吗", "呢", "啊", "吧",
        "是", "在", "有", "和", "与", "及", "等", "也", "都", "就",
        "要", "会", "可以", "能", "这", "那", "个", "一些", "一点",
        "上", "下", "左", "右", "前", "后", "里", "外", "中",
        "去", "来", "到", "往", "向", "从", "为", "对", "以",
        "把", "被", "给", "让", "叫", "比", "跟", "和", "同",
        "什么", "怎么", "为什么", "哪里", "何时", "谁", "多少",
        "一", "二", "三", "四", "五", "六", "七", "八", "九", "十",
        "个", "只", "条", "件", "本", "张", "篇", "首", "幅", "座",
    ]


def load_dimensions_from_config(config: Dict) -> List[EmotionDimension]:
    """
    从配置字典加载情绪维度
    
    配置格式:
    {
        "dimensions": [
            {
                "name": "work_pressure",
                "low_label": "轻松",
                "high_label": "压力大",
                "min_val": -5.0,
                "max_val": 5.0
            }
        ]
    }
    """
    dimensions = []
    dim_configs = config.get("dimensions", [])
    for dim_config in dim_configs:
        dimension = EmotionDimension(
            name=dim_config["name"],
            low_label=dim_config.get("low_label", "低"),
            high_label=dim_config.get("high_label", "高"),
            min_val=dim_config.get("min_val", -5.0),
            max_val=dim_config.get("max_val", 5.0),
        )
        dimensions.append(dimension)
    return dimensions


def load_synonym_groups_from_config(config: Dict) -> List[SynonymGroup]:
    """
    从配置字典加载同义词组
    
    配置格式:
    {
        "synonym_groups": [
            {
                "keyword": "工作负荷",
                "entries": [
                    {
                        "word": "清闲",
                        "emotion_scores": {
                            "positive_negative": 2.0,
                            "work_pressure": -4.0
                        },
                        "intensity": 1.2
                    }
                ]
            }
        ]
    }
    """
    groups = []
    group_configs = config.get("synonym_groups", [])
    for group_config in group_configs:
        entries = []
        for entry_config in group_config["entries"]:
            entry = SynonymEntry(
                word=entry_config["word"],
                emotion_scores=entry_config.get("emotion_scores", {}),
                intensity=entry_config.get("intensity", 1.0),
            )
            entries.append(entry)
        
        group = SynonymGroup(
            keyword=group_config["keyword"],
            entries=entries,
        )
        groups.append(group)
    return groups


def load_neutral_words_from_config(config: Dict) -> List[str]:
    """
    从配置字典加载中性词
    """
    return config.get("neutral_words", [])


def load_config_from_file(filepath: str) -> Dict:
    """
    从JSON或YAML文件加载配置，自动检测格式
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"配置文件不存在: {filepath}")
    
    ext = os.path.splitext(filepath)[1].lower()
    
    with open(filepath, "r", encoding="utf-8") as f:
        if ext in [".yaml", ".yml"]:
            if not HAS_YAML:
                raise ImportError("请先安装 PyYAML: pip install pyyaml")
            config = yaml.safe_load(f)
        else:
            config = json.load(f)
    
    return config


def detect_word_conflicts(
    groups: List[SynonymGroup]
) -> Dict[str, List[str]]:
    """
    检测同义词组中的词冲突
    
    Returns:
        {词: [所属同义词组keyword列表]}
    """
    word_to_groups: Dict[str, List[str]] = {}
    for group in groups:
        for entry in group.entries:
            if entry.word not in word_to_groups:
                word_to_groups[entry.word] = []
            word_to_groups[entry.word].append(group.keyword)
    
    conflicts = {word: groups for word, groups in word_to_groups.items() if len(groups) > 1}
    return conflicts


def resolve_word_conflicts(
    groups: List[SynonymGroup],
    strategy: WordConflictStrategy = WordConflictStrategy.KEEP_FIRST
) -> Tuple[List[SynonymGroup], Dict[str, str]]:
    """
    解决同义词组中的词冲突
    
    Args:
        groups: 同义词组列表
        strategy: 冲突处理策略
            - KEEP_FIRST: 保留第一个出现的词组中的词
            - KEEP_LAST: 保留最后一个出现的词组中的词
            - ERROR: 遇到冲突抛出错误
            - MERGE: 合并所有词组的分数（取平均值）
    
    Returns:
        (处理后的同义词组, 词到最终所属组的映射)
    """
    conflicts = detect_word_conflicts(groups)
    if not conflicts:
        word_owner = {}
        for group in groups:
            for entry in group.entries:
                word_owner[entry.word] = group.keyword
        return groups, word_owner
    
    if strategy == WordConflictStrategy.ERROR:
        conflict_msgs = [f"'{word}' 出现在词组: {', '.join(groups)}" for word, groups in conflicts.items()]
        raise ValueError(f"检测到词冲突:\n" + "\n".join(conflict_msgs))
    
    word_owner: Dict[str, str] = {}
    word_entries: Dict[str, List[Tuple[str, SynonymEntry]]] = {}
    
    for group in groups:
        for entry in group.entries:
            if entry.word not in word_entries:
                word_entries[entry.word] = []
            word_entries[entry.word].append((group.keyword, entry))
    
    if strategy == WordConflictStrategy.KEEP_FIRST:
        for word, entries in word_entries.items():
            word_owner[word] = entries[0][0]
    elif strategy == WordConflictStrategy.KEEP_LAST:
        for word, entries in word_entries.items():
            word_owner[word] = entries[-1][0]
    elif strategy == WordConflictStrategy.MERGE:
        merged_groups: Dict[str, SynonymGroup] = {}
        for word, entries in word_entries.items():
            if len(entries) == 1:
                word_owner[word] = entries[0][0]
                continue
            
            all_scores: Dict[str, List[float]] = {}
            total_intensity = 0.0
            for _, entry in entries:
                for dim, score in entry.emotion_scores.items():
                    if dim not in all_scores:
                        all_scores[dim] = []
                    all_scores[dim].append(score * entry.intensity)
                total_intensity += entry.intensity
            
            avg_scores = {dim: sum(scores) / len(scores) for dim, scores in all_scores.items()}
            avg_intensity = total_intensity / len(entries)
            
            merged_entry = SynonymEntry(
                word=word,
                emotion_scores=avg_scores,
                intensity=avg_intensity
            )
            
            merged_keyword = entries[0][0]
            word_owner[word] = merged_keyword
            
            if merged_keyword not in merged_groups:
                original_group = next(g for g in groups if g.keyword == merged_keyword)
                merged_groups[merged_keyword] = SynonymGroup(
                    keyword=merged_keyword,
                    entries=[e for e in original_group.entries if e.word != word]
                )
            
            merged_groups[merged_keyword].entries.append(merged_entry)
        
        result_groups = []
        for group in groups:
            if group.keyword in merged_groups:
                result_groups.append(merged_groups[group.keyword])
            else:
                has_conflict_word = any(e.word in conflicts for e in group.entries)
                if not has_conflict_word:
                    result_groups.append(group)
        
        return result_groups, word_owner
    
    result_groups = []
    for group in groups:
        filtered_entries = []
        for entry in group.entries:
            if word_owner.get(entry.word) == group.keyword:
                filtered_entries.append(entry)
        if filtered_entries:
            result_groups.append(SynonymGroup(
                keyword=group.keyword,
                entries=filtered_entries
            ))
    
    return result_groups, word_owner


def apply_word_score_overrides(
    groups: List[SynonymGroup],
    overrides: Dict[str, Dict[str, float]],
    create_if_missing: bool = True
) -> Tuple[List[SynonymGroup], List[str]]:
    """
    应用词分覆盖
    
    Args:
        groups: 同义词组列表
        overrides: {词: {维度名: 分数}}
        create_if_missing: 如果词不存在，是否创建新的词组
    
    Returns:
        (更新后的同义词组, 被修改的词列表)
    """
    modified_words = []
    word_to_group: Dict[str, SynonymGroup] = {}
    
    for group in groups:
        for entry in group.entries:
            word_to_group[entry.word] = group
    
    for word, scores in overrides.items():
        if word in word_to_group:
            group = word_to_group[word]
            entry = group.get_entry(word)
            if entry:
                for dim, score in scores.items():
                    entry.emotion_scores[dim] = score
                modified_words.append(word)
        elif create_if_missing:
            new_group = SynonymGroup(
                keyword=word,
                entries=[
                    SynonymEntry(
                        word=word,
                        emotion_scores=scores,
                        intensity=1.0
                    )
                ]
            )
            groups.append(new_group)
            word_to_group[word] = new_group
            modified_words.append(word)
    
    return groups, modified_words


def export_lexicon_template(
    filepath: str,
    include_default: bool = True,
    format: str = "json"
) -> None:
    """
    导出词库配置模板
    
    Args:
        filepath: 输出文件路径
        include_default: 是否包含默认词库内容
        format: 输出格式 ('json' 或 'yaml')
    """
    if include_default:
        dimensions = DefaultDimensions.get_defaults()
        synonym_groups = create_default_synonym_groups()
        neutral_words = create_default_neutral_words()
    else:
        dimensions = [
            EmotionDimension(name="positive_negative", low_label="消极", high_label="积极"),
            EmotionDimension(name="custom_dimension", low_label="低", high_label="高"),
        ]
        synonym_groups = [
            SynonymGroup(
                keyword="示例词组",
                entries=[
                    SynonymEntry(
                        word="示例词1",
                        emotion_scores={"positive_negative": 2.0, "custom_dimension": -1.0},
                        intensity=1.0
                    ),
                    SynonymEntry(
                        word="示例词2",
                        emotion_scores={"positive_negative": -2.0, "custom_dimension": 1.0},
                        intensity=1.0
                    ),
                ]
            )
        ]
        neutral_words = ["的", "了", "是"]
    
    config = {
        "_comment": "词库配置模板 - 可在此基础上修改为您的场景（工作压力、学习、客服等）",
        "dimensions": [dimension_to_dict(d) for d in dimensions],
        "synonym_groups": [synonym_group_to_dict(g) for g in synonym_groups],
        "neutral_words": neutral_words,
    }
    
    ext = os.path.splitext(filepath)[1].lower()
    if ext in [".yaml", ".yml"]:
        format = "yaml"
    
    with open(filepath, "w", encoding="utf-8") as f:
        if format == "yaml":
            if not HAS_YAML:
                raise ImportError("请先安装 PyYAML: pip install pyyaml")
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        else:
            json.dump(config, f, ensure_ascii=False, indent=2)


def load_lexicon_from_file(
    filepath: str,
    merge_default: bool = True,
    conflict_strategy: WordConflictStrategy = WordConflictStrategy.KEEP_FIRST,
    word_score_overrides: Optional[Dict[str, Dict[str, float]]] = None,
) -> Tuple[List[EmotionDimension], List[SynonymGroup], List[str], Dict[str, str]]:
    """
    从文件加载完整词库配置
    
    Args:
        filepath: JSON或YAML配置文件路径
        merge_default: 是否合并默认词库（默认维度、同义词组、中性词）
        conflict_strategy: 词冲突处理策略
        word_score_overrides: 词分覆盖 {词: {维度名: 分数}}
    
    Returns:
        (dimensions, synonym_groups, neutral_words, word_owner_map)
    """
    config = load_config_from_file(filepath)
    
    dimensions = load_dimensions_from_config(config)
    synonym_groups = load_synonym_groups_from_config(config)
    neutral_words = load_neutral_words_from_config(config)
    
    if merge_default:
        default_dims = DefaultDimensions.get_defaults()
        existing_dim_names = {d.name for d in dimensions}
        for dim in default_dims:
            if dim.name not in existing_dim_names:
                dimensions.append(dim)
        
        default_groups = create_default_synonym_groups()
        existing_keywords = {g.keyword for g in synonym_groups}
        existing_words = set()
        for g in synonym_groups:
            for e in g.entries:
                existing_words.add(e.word)
        
        for group in default_groups:
            if group.keyword not in existing_keywords:
                has_conflict = any(e.word in existing_words for e in group.entries)
                if not has_conflict:
                    synonym_groups.append(group)
                    for e in group.entries:
                        existing_words.add(e.word)
        
        default_neutral = create_default_neutral_words()
        neutral_words = list(set(neutral_words + default_neutral))
    
    if word_score_overrides:
        synonym_groups, _ = apply_word_score_overrides(
            synonym_groups, word_score_overrides, create_if_missing=True
        )
    
    synonym_groups, word_owner_map = resolve_word_conflicts(synonym_groups, conflict_strategy)
    
    return dimensions, synonym_groups, neutral_words, word_owner_map


def dimension_to_dict(dimension: EmotionDimension) -> Dict:
    """将情绪维度转换为字典（用于配置导出）"""
    return {
        "name": dimension.name,
        "low_label": dimension.low_label,
        "high_label": dimension.high_label,
        "min_val": dimension.min_val,
        "max_val": dimension.max_val,
    }


def synonym_entry_to_dict(entry: SynonymEntry) -> Dict:
    """将同义词条目转换为字典"""
    return {
        "word": entry.word,
        "emotion_scores": entry.emotion_scores,
        "intensity": entry.intensity,
    }


def synonym_group_to_dict(group: SynonymGroup) -> Dict:
    """将同义词组转换为字典"""
    return {
        "keyword": group.keyword,
        "entries": [synonym_entry_to_dict(e) for e in group.entries],
    }


def export_lexicon_to_file(
    filepath: str,
    dimensions: List[EmotionDimension],
    synonym_groups: List[SynonymGroup],
    neutral_words: Optional[List[str]] = None,
    format: Optional[str] = None,
) -> None:
    """
    导出词库配置到JSON或YAML文件
    
    Args:
        filepath: 输出文件路径
        dimensions: 情绪维度列表
        synonym_groups: 同义词组列表
        neutral_words: 中性词列表
        format: 输出格式 ('json', 'yaml')，None则根据文件扩展名自动检测
    """
    config = {
        "dimensions": [dimension_to_dict(d) for d in dimensions],
        "synonym_groups": [synonym_group_to_dict(g) for g in synonym_groups],
    }
    if neutral_words is not None:
        config["neutral_words"] = neutral_words
    
    if format is None:
        ext = os.path.splitext(filepath)[1].lower()
        format = "yaml" if ext in [".yaml", ".yml"] else "json"
    
    with open(filepath, "w", encoding="utf-8") as f:
        if format == "yaml":
            if not HAS_YAML:
                raise ImportError("请先安装 PyYAML: pip install pyyaml")
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        else:
            json.dump(config, f, ensure_ascii=False, indent=2)
