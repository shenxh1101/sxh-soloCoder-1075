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
    apply_word_score_overrides,
    WordConflictStrategy,
    resolve_word_conflicts,
)


class EmotionScorer:
    """
    情绪评分器
    根据同义词库计算句子的情绪分数
    评分和替换使用同一组词索引，保证一致性
    """

    def __init__(
        self,
        dimensions: List[EmotionDimension],
        synonym_groups: Optional[List[SynonymGroup]] = None,
        neutral_words: Optional[List[str]] = None,
        word_owner_map: Optional[Dict[str, str]] = None,
        conflict_strategy: WordConflictStrategy = WordConflictStrategy.KEEP_FIRST,
    ):
        self.dimensions = dimensions
        self.synonym_groups = synonym_groups or create_default_synonym_groups()
        self.neutral_words = neutral_words or create_default_neutral_words()
        self._word_to_group: Dict[str, SynonymGroup] = {}
        self._word_to_entry: Dict[str, SynonymEntry] = {}
        self._sorted_words: List[str] = []
        self.conflict_strategy = conflict_strategy
        self._build_word_index(word_owner_map)

    def _build_word_index(self, word_owner_map: Optional[Dict[str, str]] = None) -> None:
        """构建词到同义词组和条目的统一索引
        
        评分和替换都使用这个索引，保证一致性
        """
        if word_owner_map is None:
            self.synonym_groups, word_owner_map = resolve_word_conflicts(
                self.synonym_groups, self.conflict_strategy
            )
        
        for group in self.synonym_groups:
            for entry in group.entries:
                if word_owner_map.get(entry.word) == group.keyword:
                    self._word_to_group[entry.word] = group
                    self._word_to_entry[entry.word] = entry
        
        self._sorted_words = sorted(
            self._word_to_entry.keys(),
            key=lambda w: len(w),
            reverse=True
        )

    def _find_all_matches(
        self,
        sentence: str,
        locked_words: Optional[List[str]] = None,
    ) -> List[Tuple[int, str, SynonymEntry, SynonymGroup]]:
        """
        统一的匹配算法，供评分和替换共同使用
        使用最长匹配，保证不重叠
        
        返回：[(起始位置, 匹配词, 同义词条目, 同义词组)]
        """
        locked_lower = [w.lower() for w in (locked_words or [])]
        all_matches = []
        
        for word in self._sorted_words:
            if word.lower() in locked_lower:
                continue
            
            entry = self._word_to_entry.get(word)
            group = self._word_to_group.get(word)
            if not entry or not group:
                continue
            
            start = 0
            while True:
                pos = sentence.find(word, start)
                if pos == -1:
                    break
                all_matches.append((pos, len(word), word, entry, group))
                start = pos + 1
        
        all_matches.sort(key=lambda x: (-x[1], x[0]))
        
        matched = []
        used_ranges = []
        
        for pos, length, word, entry, group in all_matches:
            word_end = pos + length
            overlaps = False
            for used_start, used_end in used_ranges:
                if not (word_end <= used_start or pos >= used_end):
                    overlaps = True
                    break
            
            if not overlaps:
                matched.append((pos, word, entry, group))
                used_ranges.append((pos, word_end))
        
        matched.sort(key=lambda x: x[0])
        return matched

    def find_replacable_words(self, sentence: str, locked_words: List[str]) -> List[Tuple[int, str, SynonymGroup]]:
        """
        找出句子中可替换的词
        返回列表：[(起始位置, 词, 同义词组)]
        使用统一的匹配算法，保证与评分逻辑一致
        """
        matches = self._find_all_matches(sentence, locked_words)
        return [(pos, word, group) for pos, word, entry, group in matches if len(group.entries) > 1]

    def find_matched_words(self, sentence: str) -> List[Tuple[int, str, SynonymEntry]]:
        """
        使用最长匹配算法找出句子中所有匹配的词
        返回：[(起始位置, 匹配词, 同义词条目)]
        使用统一的匹配算法，保证与替换逻辑一致
        """
        matches = self._find_all_matches(sentence)
        return [(pos, word, entry) for pos, word, entry, group in matches]

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
        返回包含匹配词、各词分数、贡献最大维度等详细信息
        """
        matched_words = self.find_matched_words(sentence)
        scores = self.score_sentence(sentence)

        details = {
            "sentence": sentence,
            "matched_words": [],
            "final_scores": scores,
            "main_contributor": self.get_main_contributor(sentence),
            "top_dimension": self.get_top_dimension(sentence),
        }

        for pos, word, entry in matched_words:
            details["matched_words"].append({
                "position": pos,
                "word": word,
                "intensity": entry.intensity,
                "emotion_scores": entry.emotion_scores,
                "group_keyword": self._word_to_group.get(word, {}).keyword if self._word_to_group.get(word) else None,
            })

        return details

    def get_main_contributor(self, sentence: str) -> Optional[Dict[str, Any]]:
        """
        获取对分数贡献最大的词
        """
        matched_words = self.find_matched_words(sentence)
        if not matched_words:
            return None
        
        max_contribution = 0.0
        main_word = None
        
        for pos, word, entry in matched_words:
            total_abs_score = sum(abs(s) for s in entry.emotion_scores.values()) * entry.intensity
            if total_abs_score > max_contribution:
                max_contribution = total_abs_score
                main_word = word
        
        if main_word:
            entry = self._word_to_entry[main_word]
            group = self._word_to_group[main_word]
            return {
                "word": main_word,
                "intensity": entry.intensity,
                "emotion_scores": entry.emotion_scores,
                "group_keyword": group.keyword,
                "contribution": max_contribution,
            }
        return None

    def get_top_dimension(self, sentence: str) -> Optional[Dict[str, Any]]:
        """
        获取绝对分数最高的维度
        """
        scores = self.score_sentence(sentence)
        if not scores:
            return None
        
        max_abs = 0.0
        top_dim = None
        top_score = 0.0
        
        for dim_name, score in scores.items():
            abs_score = abs(score)
            if abs_score > max_abs:
                max_abs = abs_score
                top_dim = dim_name
                top_score = score
        
        if top_dim:
            dim = next((d for d in self.dimensions if d.name == top_dim), None)
            return {
                "dimension": top_dim,
                "score": top_score,
                "label": f"{dim.high_label if top_score > 0 else dim.low_label}" if dim else "",
            }
        return None

    def apply_word_overrides(self, overrides: Dict[str, Dict[str, float]]) -> List[str]:
        """
        应用词分覆盖
        Args:
            overrides: {词: {维度名: 分数}}
        Returns:
            被修改的词列表
        """
        self.synonym_groups, modified = apply_word_score_overrides(
            self.synonym_groups, overrides, create_if_missing=True
        )
        self._build_word_index()
        return modified

    def get_word_entry(self, word: str) -> Optional[SynonymEntry]:
        """获取词的同义词条目（使用统一索引）"""
        return self._word_to_entry.get(word)

    def get_synonym_group(self, word: str) -> Optional[SynonymGroup]:
        """获取词所属的同义词组（使用统一索引）"""
        return self._word_to_group.get(word)

    def add_synonym_group(self, group: SynonymGroup) -> None:
        """添加同义词组（重建索引）"""
        self.synonym_groups.append(group)
        self._build_word_index()

    def add_custom_dimension(self, dimension: EmotionDimension) -> None:
        """添加自定义情绪维度"""
        self.dimensions.append(dimension)

    def get_all_words(self) -> List[str]:
        """获取所有可识别的词"""
        return sorted(self._word_to_entry.keys())
