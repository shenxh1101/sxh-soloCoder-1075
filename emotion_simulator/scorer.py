"""
情绪评分模块
负责计算句子的情绪分数
"""

from typing import Dict, List, Tuple, Optional
from .models import (
    EmotionDimension,
    SynonymGroup,
    SynonymEntry,
    create_default_synonym_groups,
    create_default_neutral_words,
)


class EmotionScorer:
    """
    情绪评分器
    根据同义词库计算句子的情绪分数
    """

    def __init__(
        self,
        dimensions: List[EmotionDimension],
        synonym_groups: Optional[List[SynonymGroup]] = None,
        neutral_words: Optional[List[str]] = None,
    ):
        self.dimensions = dimensions
        self.synonym_groups = synonym_groups or create_default_synonym_groups()
        self.neutral_words = neutral_words or create_default_neutral_words()
        self._word_to_group: Dict[str, SynonymGroup] = {}
        self._build_word_index()

    def _build_word_index(self) -> None:
        """构建词到同义词组的索引"""
        for group in self.synonym_groups:
            for entry in group.entries:
                self._word_to_group[entry.word] = group

    def find_replacable_words(self, sentence: str, locked_words: List[str]) -> List[Tuple[int, str, SynonymGroup]]:
        """
        找出句子中可替换的词
        返回列表：[(起始位置, 词, 同义词组)]
        """
        results = []
        locked_lower = [w.lower() for w in locked_words]

        sorted_words = sorted(
            self._word_to_group.keys(),
            key=lambda w: len(w),
            reverse=True
        )

        for word in sorted_words:
            if word.lower() in locked_lower:
                continue

            start = 0
            while True:
                pos = sentence.find(word, start)
                if pos == -1:
                    break

                if not self._is_overlapping(pos, len(word), results):
                    group = self._word_to_group[word]
                    results.append((pos, word, group))

                start = pos + 1

        return sorted(results, key=lambda x: x[0])

    def _is_overlapping(
        self,
        pos: int,
        length: int,
        existing: List[Tuple[int, str, SynonymGroup]]
    ) -> bool:
        """检查是否与已有词重叠"""
        for e_pos, e_word, _ in existing:
            e_end = e_pos + len(e_word)
            new_end = pos + length
            if not (new_end <= e_pos or pos >= e_end):
                return True
        return False

    def score_sentence(self, sentence: str) -> Dict[str, float]:
        """
        计算句子的情绪分数
        返回各维度的分数字典
        """
        scores = {dim.name: 0.0 for dim in self.dimensions}
        total_intensity = 0.0

        for group in self.synonym_groups:
            for entry in group.entries:
                if entry.word in sentence:
                    count = sentence.count(entry.word)
                    weight = entry.intensity * count
                    total_intensity += weight
                    for dim_name in scores:
                        scores[dim_name] += entry.get_score(dim_name) * weight

        if total_intensity > 0:
            for dim_name in scores:
                scores[dim_name] /= total_intensity

        return scores

    def get_word_entry(self, word: str) -> Optional[SynonymEntry]:
        """获取词的同义词条目"""
        group = self._word_to_group.get(word)
        if group:
            return group.get_entry(word)
        return None

    def get_synonym_group(self, word: str) -> Optional[SynonymGroup]:
        """获取词所属的同义词组"""
        return self._word_to_group.get(word)

    def add_synonym_group(self, group: SynonymGroup) -> None:
        """添加同义词组"""
        self.synonym_groups.append(group)
        for entry in group.entries:
            self._word_to_group[entry.word] = group

    def add_custom_dimension(self, dimension: EmotionDimension) -> None:
        """添加自定义情绪维度"""
        self.dimensions.append(dimension)
