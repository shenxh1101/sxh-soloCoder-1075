"""
导出模块
支持将模拟结果导出为JSON格式
"""

import json
from typing import List, Dict, Any
from .models import SimulationResult, TransmissionStep, EmotionDimension


class JSONExporter:
    """
    JSON导出器
    将模拟结果导出为JSON格式
    """

    @staticmethod
    def _dimension_to_dict(dimension: EmotionDimension) -> Dict[str, Any]:
        """将情绪维度转换为字典"""
        return {
            "name": dimension.name,
            "low_label": dimension.low_label,
            "high_label": dimension.high_label,
            "min_val": dimension.min_val,
            "max_val": dimension.max_val,
        }

    @staticmethod
    def _step_to_dict(step: TransmissionStep) -> Dict[str, Any]:
        """将传递步骤转换为字典"""
        data = {
            "step": step.step,
            "original_sentence": step.original_sentence,
            "modified_sentence": step.modified_sentence,
            "emotion_scores": step.emotion_scores,
            "locked_words": step.locked_words,
        }
        if step.changed_word:
            data["changed_word"] = {
                "original": step.changed_word[0],
                "new": step.changed_word[1],
            }
        else:
            data["changed_word"] = None
        return data

    @staticmethod
    def _result_to_dict(result: SimulationResult) -> Dict[str, Any]:
        """将模拟结果转换为字典"""
        return {
            "initial_sentence": result.initial_sentence,
            "final_sentence": result.get_final_sentence(),
            "locked_words": result.locked_words,
            "dimensions": [JSONExporter._dimension_to_dict(d) for d in result.dimensions],
            "steps": [JSONExporter._step_to_dict(s) for s in result.steps],
            "emotion_series": {
                dim.name: result.get_emotion_series(dim.name)
                for dim in result.dimensions
            },
            "summary": JSONExporter._generate_summary(result),
        }

    @staticmethod
    def _generate_summary(result: SimulationResult) -> Dict[str, Any]:
        """生成模拟摘要"""
        summary = {
            "total_steps": len(result.steps),
            "changes_count": sum(1 for s in result.steps if s.changed_word is not None),
            "initial_scores": result.steps[0].emotion_scores if result.steps else {},
            "final_scores": result.steps[-1].emotion_scores if result.steps else {},
        }

        if result.steps:
            changes = []
            for step in result.steps:
                if step.changed_word:
                    changes.append({
                        "step": step.step,
                        "original": step.changed_word[0],
                        "new": step.changed_word[1],
                    })
            summary["all_changes"] = changes

            score_diffs = {}
            for dim in result.dimensions:
                series = result.get_emotion_series(dim.name)
                if series:
                    score_diffs[dim.name] = {
                        "start": series[0],
                        "end": series[-1],
                        "change": series[-1] - series[0],
                        "min": min(series),
                        "max": max(series),
                        "mean": sum(series) / len(series),
                    }
            summary["score_differences"] = score_diffs

        return summary

    @staticmethod
    def export(result: SimulationResult, filepath: str, indent: int = 2) -> None:
        """
        导出单个模拟结果为JSON文件
        """
        data = JSONExporter._result_to_dict(result)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)

    @staticmethod
    def export_batch(results: List[SimulationResult], filepath: str, indent: int = 2) -> None:
        """
        导出批量模拟结果为JSON文件
        """
        data = {
            "count": len(results),
            "results": [JSONExporter._result_to_dict(r) for r in results],
        }

        if results:
            data["common_dimensions"] = [
                JSONExporter._dimension_to_dict(d) for d in results[0].dimensions
            ]

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)

    @staticmethod
    def to_string(result: SimulationResult, indent: int = 2) -> str:
        """
        将单个模拟结果转换为JSON字符串
        """
        data = JSONExporter._result_to_dict(result)
        return json.dumps(data, ensure_ascii=False, indent=indent)

    @staticmethod
    def batch_to_string(results: List[SimulationResult], indent: int = 2) -> str:
        """
        将批量模拟结果转换为JSON字符串
        """
        data = {
            "count": len(results),
            "results": [JSONExporter._result_to_dict(r) for r in results],
        }
        return json.dumps(data, ensure_ascii=False, indent=indent)
