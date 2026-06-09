"""
情绪演进模拟器核心模块
实现句子传递、情绪变化的核心逻辑
"""

import random
from typing import Dict, List, Optional, Callable
from .models import (
    EmotionDimension,
    SynonymGroup,
    SynonymEntry,
    TransmissionStep,
    SimulationResult,
    DefaultDimensions,
    create_default_synonym_groups,
    load_lexicon_from_file,
)
from .scorer import EmotionScorer


class EmotionSimulator:
    """
    情绪演进模拟器
    模拟句子在传递过程中的情绪变化
    """

    def __init__(
        self,
        dimensions: Optional[List[EmotionDimension]] = None,
        synonym_groups: Optional[List[SynonymGroup]] = None,
        change_probability: float = 0.6,
        magnitude: float = 0.5,
        drift_strength: float = 0.0,
        random_seed: Optional[int] = None,
    ):
        self.dimensions = dimensions or DefaultDimensions.get_defaults()
        self.scorer = EmotionScorer(self.dimensions, synonym_groups)
        self.change_probability = change_probability
        self.magnitude = magnitude
        self.drift_strength = drift_strength
        self.drift_direction: Dict[str, float] = {dim.name: 0.0 for dim in self.dimensions}

        if random_seed is not None:
            random.seed(random_seed)

    def set_drift(self, dimension_name: str, direction: float) -> None:
        """
        设置情绪漂移方向
        direction: -1.0 ~ 1.0，负值偏向低端，正值偏向高端
        """
        if dimension_name in self.drift_direction:
            self.drift_direction[dimension_name] = max(-1.0, min(1.0, direction))

    def _select_replacement(
        self,
        current_word: str,
        group: SynonymGroup,
        target_dimension: str,
        current_score: float,
    ) -> Optional[SynonymEntry]:
        """
        选择替换词
        根据情绪偏差和变化幅度选择合适的同义词
        """
        current_entry = group.get_entry(current_word)
        if not current_entry:
            return None

        sorted_entries = group.get_sorted_by_dimension(target_dimension)
        current_index = next(
            (i for i, e in enumerate(sorted_entries) if e.word == current_word),
            len(sorted_entries) // 2
        )

        max_step = max(1, int(len(sorted_entries) * self.magnitude))

        drift = self.drift_direction.get(target_dimension, 0.0)
        base_direction = random.choice([-1, 1])

        if drift != 0 and random.random() < abs(drift):
            direction = 1 if drift > 0 else -1
        else:
            direction = base_direction

        step = random.randint(1, max_step)
        new_index = current_index + direction * step

        new_index = max(0, min(len(sorted_entries) - 1, new_index))

        if new_index == current_index:
            new_index = current_index + (1 if direction > 0 else -1)
            new_index = max(0, min(len(sorted_entries) - 1, new_index))

        return sorted_entries[new_index]

    def _select_target_dimension(self) -> str:
        """
        随机选择一个情绪维度作为变化目标
        """
        return random.choice(self.dimensions).name

    def transmit(
        self,
        sentence: str,
        locked_words: Optional[List[str]] = None,
    ) -> tuple[str, Optional[tuple[str, str]], Dict[str, float]]:
        """
        执行一次传递
        返回：(新句子, (原词, 新词) or None, 情绪分数)
        """
        locked_words = locked_words or []

        if random.random() > self.change_probability:
            scores = self.scorer.score_sentence(sentence)
            return sentence, None, scores

        replacable = self.scorer.find_replacable_words(sentence, locked_words)
        if not replacable:
            scores = self.scorer.score_sentence(sentence)
            return sentence, None, scores

        pos, original_word, group = random.choice(replacable)

        target_dimension = self._select_target_dimension()
        current_scores = self.scorer.score_sentence(sentence)
        current_score = current_scores.get(target_dimension, 0.0)

        new_entry = self._select_replacement(
            original_word, group, target_dimension, current_score
        )

        if not new_entry or new_entry.word == original_word:
            return sentence, None, current_scores

        new_sentence = sentence[:pos] + new_entry.word + sentence[pos + len(original_word):]
        new_scores = self.scorer.score_sentence(new_sentence)

        return new_sentence, (original_word, new_entry.word), new_scores

    def simulate(
        self,
        initial_sentence: str,
        num_steps: int,
        locked_words: Optional[List[str]] = None,
        callback: Optional[Callable[[TransmissionStep], None]] = None,
    ) -> SimulationResult:
        """
        执行完整的模拟
        """
        locked_words = locked_words or []
        result = SimulationResult(
            initial_sentence=initial_sentence,
            locked_words=locked_words,
            dimensions=self.dimensions,
        )

        current_sentence = initial_sentence

        for step_num in range(num_steps):
            original = current_sentence
            new_sentence, changed, scores = self.transmit(current_sentence, locked_words)

            step = TransmissionStep(
                step=step_num + 1,
                original_sentence=original,
                modified_sentence=new_sentence,
                emotion_scores=scores,
                changed_word=changed,
                locked_words=locked_words.copy(),
            )

            result.steps.append(step)
            current_sentence = new_sentence

            if callback:
                callback(step)

        return result

    def batch_simulate(
        self,
        sentences: List[str],
        num_steps: int,
        locked_words: Optional[List[str]] = None,
    ) -> List[SimulationResult]:
        """
        批量模拟多个句子
        """
        results = []
        for sentence in sentences:
            result = self.simulate(sentence, num_steps, locked_words)
            results.append(result)
        return results

    def add_custom_synonym_group(self, group: SynonymGroup) -> None:
        """添加自定义同义词组"""
        self.scorer.add_synonym_group(group)

    def add_custom_dimension(self, dimension: EmotionDimension) -> None:
        """添加自定义情绪维度"""
        self.dimensions.append(dimension)
        self.scorer.add_custom_dimension(dimension)
        self.drift_direction[dimension.name] = 0.0

    @classmethod
    def from_lexicon_file(
        cls,
        lexicon_filepath: str,
        merge_default: bool = True,
        change_probability: float = 0.6,
        magnitude: float = 0.5,
        drift_strength: float = 0.0,
        random_seed: Optional[int] = None,
    ) -> "EmotionSimulator":
        """
        从词库配置文件创建模拟器
        
        Args:
            lexicon_filepath: 词库JSON配置文件路径
            merge_default: 是否合并默认词库
            change_probability: 变化概率
            magnitude: 变化幅度
            drift_strength: 漂移强度
            random_seed: 随机种子
        
        Returns:
            EmotionSimulator实例
        """
        dimensions, synonym_groups, neutral_words = load_lexicon_from_file(
            lexicon_filepath, merge_default=merge_default
        )

        simulator = cls(
            dimensions=dimensions,
            synonym_groups=synonym_groups,
            change_probability=change_probability,
            magnitude=magnitude,
            drift_strength=drift_strength,
            random_seed=random_seed,
        )

        simulator.scorer.neutral_words = neutral_words

        return simulator
