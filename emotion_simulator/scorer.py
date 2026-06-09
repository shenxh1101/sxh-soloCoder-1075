"""
情绪评分模块
负责计算句子的情绪分数
"""

from typing import Dict, List, Tuple, Optional, Any
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

    def find_matched_words(self, sentence: str) -> List[Tuple[int, str, SynonymEntry]]:
        """
        使用最长匹配算法找出句子中所有匹配的词
        返回：[(起始位置, 匹配词, 同义词条目)]
        保证不重叠，优先匹配更长的词
        """
        all_words = []
        for group in self.synonym_groups:
            for entry in group.entries:
                word = entry.word
                start = 0
                while True:
                    pos = sentence.find(word, start)
                    if pos == -1:
                        break
                    all_words.append((pos, len(word), word, entry))
                    start = pos + 1

        all_words.sort(key=lambda x: (-x[1], x[0]))

        matched = []
        used_ranges = []

        for pos, length, word, entry in all_words:
            word_end = pos + length
            overlaps = False
            for used_start, used_end in used_ranges:
                if not (word_end <= used_start or pos >= used_end):
                    overlaps = True
                    break

            if not overlaps:
                matched.append((pos, word, entry))
                used_ranges.append((pos, word_end))

        matched.sort(key=lambda x: x[0])
        return matched

    def score_sentence(self, sentence: str) -> Dict[str, float]:
        """
        计算句子的情绪分数（使用最长匹配算法，避免重复计分）
        返回各维度的分数字典
        """
        scores = {dim.name: 0.0 for dim in self.dimensions}
        total_intensity = 0.0

        matched_words = self.find_matched_words(sentence)

        for pos, word, entry in matched_words:
            weight = entry.intensity
            total_intensity += weight
            for dim_name in scores:
                scores[dim_name] += entry.get_score(dim_name) * weight

        if total_intensity > 0:
            for dim_name in scores:
                scores[dim_name] /= total_intensity

        return scores

    def get_score_details(self, sentence: str) -> Dict[str, Any]:
        """
        获取详细的评分信息（用于调试）
        返回包含匹配词、各词分数等详细信息
        """
        matched_words = self.find_matched_words(sentence)
        scores = self.score_sentence(sentence)

        details = {
            "sentence": sentence,
            "matched_words": [],
            "final_scores": scores,
        }

        for pos, word, entry in matched_words:
            details["matched_words"].append({
                "position": pos,
                "word": word,
                "intensity": entry.intensity,
                "emotion_scores": entry.emotion_scores,
            })

        return details

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
