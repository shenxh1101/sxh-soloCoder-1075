"""
导出模块
支持将模拟结果导出为JSON和YAML格式
"""

import json
import csv
import io
from typing import List, Dict, Any, Optional

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

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
    def _matched_word_to_dict(mw) -> Dict[str, Any]:
        """将MatchedWordDetail转换为字典"""
        return {
            "word": mw.word,
            "position": mw.position,
            "group_keyword": mw.group_keyword,
            "emotion_scores": mw.emotion_scores,
            "intensity": mw.intensity,
            "contribution": mw.contribution,
        }

    @staticmethod
    def _score_contribution_to_dict(sc) -> Dict[str, Any]:
        """将ScoreContribution转换为字典"""
        data = {
            "matched_words": [JSONExporter._matched_word_to_dict(mw) for mw in sc.matched_words],
            "main_contributor": JSONExporter._matched_word_to_dict(sc.main_contributor) if sc.main_contributor else None,
            "top_dimension": sc.top_dimension,
        }
        return data

    @staticmethod
    def _replacement_detail_to_dict(rd) -> Dict[str, Any]:
        """将ReplacementDetail转换为字典"""
        if rd is None:
            return None
        data = {
            "original_word": rd.original_word,
            "new_word": rd.new_word,
            "group_keyword": rd.group_keyword,
            "position": rd.position,
            "target_dimension": rd.target_dimension,
            "before_scores": rd.before_scores,
            "after_scores": rd.after_scores,
            "before_contribution": JSONExporter._score_contribution_to_dict(rd.before_contribution) if rd.before_contribution else None,
            "after_contribution": JSONExporter._score_contribution_to_dict(rd.after_contribution) if rd.after_contribution else None,
        }
        return data

    @staticmethod
    def _step_to_dict(step: TransmissionStep) -> Dict[str, Any]:
        """将传递步骤转换为字典（包含完整可追溯信息）"""
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
        
        if step.score_contribution:
            data["score_contribution"] = JSONExporter._score_contribution_to_dict(step.score_contribution)
        
        if step.replacement_detail:
            data["replacement_detail"] = JSONExporter._replacement_detail_to_dict(step.replacement_detail)
        
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


class BatchSummarizer:
    """
    批量模拟结果汇总器
    支持排序、统计、简洁导出
    """

    @staticmethod
    def get_result_metrics(result: SimulationResult) -> Dict[str, Any]:
        """
        获取单个结果的统计指标
        """
        if not result.steps:
            return {}

        metrics = {
            "initial_sentence": result.initial_sentence,
            "final_sentence": result.get_final_sentence(),
            "total_steps": len(result.steps),
            "changes_count": sum(1 for s in result.steps if s.changed_word is not None),
        }

        for dim in result.dimensions:
            series = result.get_emotion_series(dim.name)
            if series:
                change = series[-1] - series[0]
                metrics[f"{dim.name}_start"] = series[0]
                metrics[f"{dim.name}_end"] = series[-1]
                metrics[f"{dim.name}_change"] = change
                metrics[f"{dim.name}_change_abs"] = abs(change)
                metrics[f"{dim.name}_min"] = min(series)
                metrics[f"{dim.name}_max"] = max(series)
                metrics[f"{dim.name}_mean"] = sum(series) / len(series)

        return metrics

    @staticmethod
    def summarize(
        results: List[SimulationResult],
        sort_by: Optional[str] = None,
        sort_ascending: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        汇总批量模拟结果，支持排序
        
        Args:
            results: 模拟结果列表
            sort_by: 排序字段名（如 "positive_negative_change_abs", "changes_count" 等）
            sort_ascending: 是否升序排列
        
        Returns:
            排序后的统计指标列表
        """
        summaries = [BatchSummarizer.get_result_metrics(r) for r in results]

        if sort_by and sort_by in summaries[0] if summaries else False:
            summaries.sort(key=lambda x: x.get(sort_by, 0), reverse=not sort_ascending)

        return summaries

    @staticmethod
    def export_summary_json(
        results: List[SimulationResult],
        filepath: str,
        sort_by: Optional[str] = None,
        sort_ascending: bool = False,
    ) -> None:
        """
        导出批量汇总为简洁的JSON文件
        """
        summary = {
            "count": len(results),
            "sort_by": sort_by,
            "sort_ascending": sort_ascending,
            "results": BatchSummarizer.summarize(results, sort_by, sort_ascending),
        }

        if results:
            summary["dimensions"] = [
                JSONExporter._dimension_to_dict(d) for d in results[0].dimensions
            ]

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    @staticmethod
    def export_summary_csv(
        results: List[SimulationResult],
        filepath: str,
        sort_by: Optional[str] = None,
        sort_ascending: bool = False,
        include_steps: bool = False,
    ) -> None:
        """
        导出批量汇总为CSV文件
        
        Args:
            results: 模拟结果列表
            filepath: 输出CSV文件路径
            sort_by: 排序字段名
            sort_ascending: 是否升序排列
            include_steps: 是否包含每步的情绪分数（会产生多行列）
        """
        summaries = BatchSummarizer.summarize(results, sort_by, sort_ascending)

        if not summaries:
            return

        fieldnames = list(summaries[0].keys())

        with open(filepath, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for summary in summaries:
                writer.writerow(summary)

    @staticmethod
    def print_summary(
        results: List[SimulationResult],
        sort_by: Optional[str] = None,
        sort_ascending: bool = False,
        top_n: Optional[int] = None,
        use_color: bool = True,
    ) -> str:
        """
        打印批量汇总结果到终端
        
        Args:
            results: 模拟结果列表
            sort_by: 排序字段名
            sort_ascending: 是否升序排列
            top_n: 只显示前N条结果
            use_color: 是否使用彩色输出
        
        Returns:
            格式化的汇总字符串
        """
        summaries = BatchSummarizer.summarize(results, sort_by, sort_ascending)

        if top_n is not None:
            summaries = summaries[:top_n]

        lines = []
        lines.append("=" * 80)
        sort_info = f" (按 {sort_by} {'升序' if sort_ascending else '降序'})" if sort_by else ""
        lines.append(f"批量模拟结果汇总{sort_info}")
        lines.append("=" * 80)

        if not summaries:
            lines.append("无数据")
            return "\n".join(lines)

        headers = ["#", "初始句子", "最终句子", "变化次数"]
        dim_names = [d.name for d in results[0].dimensions]
        for dim_name in dim_names:
            headers.append(f"{dim_name}变化")
            headers.append(f"{dim_name}|min|max")

        col_widths = [4, 25, 25, 10] + [12, 18] * len(dim_names)

        header_line = "".join(h.ljust(w) for h, w in zip(headers, col_widths))
        lines.append(header_line)
        lines.append("-" * 80)

        for i, summary in enumerate(summaries, 1):
            row = [
                str(i),
                summary["initial_sentence"][:23] + "..." if len(summary["initial_sentence"]) > 25 else summary["initial_sentence"],
                summary["final_sentence"][:23] + "..." if len(summary["final_sentence"]) > 25 else summary["final_sentence"],
                str(summary["changes_count"]),
            ]

            for dim_name in dim_names:
                change = summary.get(f"{dim_name}_change", 0)
                min_val = summary.get(f"{dim_name}_min", 0)
                max_val = summary.get(f"{dim_name}_max", 0)
                change_str = f"{change:+.2f}"
                range_str = f"{min_val:+.1f}|{max_val:+.1f}"

                if use_color:
                    if change > 0:
                        change_str = f"\033[92m{change_str}\033[0m"
                    elif change < 0:
                        change_str = f"\033[91m{change_str}\033[0m"

                row.append(change_str)
                row.append(range_str)

            row_line = "".join(str(c).ljust(w) for c, w in zip(row, col_widths))
            lines.append(row_line)

        lines.append("=" * 80)
        return "\n".join(lines)


class CSVExporter:
    """
    CSV导出器
    支持将模拟结果导出为各种格式的CSV
    """

    @staticmethod
    def export_timeseries(
        result: SimulationResult,
        filepath: str,
    ) -> None:
        """
        导出单个结果的时间序列数据为CSV（包含完整可追溯信息）
        
        列: step, sentence, changed_word, target_dimension, main_contributor, top_dimension, top_dimension_score, [各维度分数...]
        """
        with open(filepath, "w", encoding="utf-8-sig", newline="") as f:
            fieldnames = [
                "step", 
                "sentence", 
                "changed_word", 
                "target_dimension", 
                "main_contributor", 
                "main_contributor_contribution",
                "top_dimension", 
                "top_dimension_score"
            ]
            for dim in result.dimensions:
                fieldnames.append(dim.name)

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for step in result.steps:
                row = {
                    "step": step.step,
                    "sentence": step.modified_sentence,
                    "changed_word": f"{step.changed_word[0]}→{step.changed_word[1]}" if step.changed_word else "",
                    "target_dimension": step.replacement_detail.target_dimension if step.replacement_detail else "",
                    "main_contributor": "",
                    "main_contributor_contribution": "",
                    "top_dimension": "",
                    "top_dimension_score": "",
                }
                
                if step.score_contribution:
                    sc = step.score_contribution
                    if sc.main_contributor:
                        row["main_contributor"] = sc.main_contributor.word
                        row["main_contributor_contribution"] = sum(abs(v) for v in sc.main_contributor.contribution.values())
                    if sc.top_dimension:
                        row["top_dimension"] = sc.top_dimension.get("dimension", "")
                        row["top_dimension_score"] = sc.top_dimension.get("score", "")
                
                for dim in result.dimensions:
                    row[dim.name] = step.emotion_scores.get(dim.name, 0.0)
                writer.writerow(row)

    @staticmethod
    def export_batch_timeseries(
        results: List[SimulationResult],
        filepath: str,
    ) -> None:
        """
        导出批量结果的时间序列数据为CSV
        
        列: result_id, initial_sentence, step, sentence, changed_word, [各维度分数...]
        """
        if not results:
            return

        with open(filepath, "w", encoding="utf-8-sig", newline="") as f:
            fieldnames = ["result_id", "initial_sentence", "step", "sentence", "changed_word"]
            for dim in results[0].dimensions:
                fieldnames.append(dim.name)

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for result_id, result in enumerate(results, 1):
                for step in result.steps:
                    row = {
                        "result_id": result_id,
                        "initial_sentence": result.initial_sentence,
                        "step": step.step,
                        "sentence": step.modified_sentence,
                        "changed_word": f"{step.changed_word[0]}→{step.changed_word[1]}" if step.changed_word else "",
                    }
                    for dim in results[0].dimensions:
                        row[dim.name] = step.emotion_scores.get(dim.name, 0.0)
                    writer.writerow(row)


def get_result_metrics(result: SimulationResult) -> Dict[str, Any]:
    """
    获取单个结果的统计指标（供外部使用）
    """
    return BatchSummarizer.get_result_metrics(result)


class YAMLExporter:
    """
    YAML导出器
    将模拟结果导出为YAML格式
    """

    @staticmethod
    def _check_yaml() -> None:
        """检查YAML是否可用"""
        if not HAS_YAML:
            raise ImportError("请先安装 PyYAML: pip install pyyaml")

    @staticmethod
    def export(result: SimulationResult, filepath: str) -> None:
        """
        导出单个模拟结果为YAML文件
        """
        YAMLExporter._check_yaml()
        data = JSONExporter._result_to_dict(result)
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    @staticmethod
    def export_batch(results: List[SimulationResult], filepath: str) -> None:
        """
        导出批量模拟结果为YAML文件
        """
        YAMLExporter._check_yaml()
        data = {
            "count": len(results),
            "results": [JSONExporter._result_to_dict(r) for r in results],
        }
        if results:
            data["common_dimensions"] = [
                JSONExporter._dimension_to_dict(d) for d in results[0].dimensions
            ]
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    @staticmethod
    def export_summary(
        results: List[SimulationResult],
        filepath: str,
        sort_by: Optional[str] = None,
        sort_ascending: bool = False,
    ) -> None:
        """
        导出批量汇总为YAML文件
        """
        YAMLExporter._check_yaml()
        summary = {
            "count": len(results),
            "sort_by": sort_by,
            "sort_ascending": sort_ascending,
            "results": BatchSummarizer.summarize(results, sort_by, sort_ascending),
        }
        if results:
            summary["dimensions"] = [
                JSONExporter._dimension_to_dict(d) for d in results[0].dimensions
            ]
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(summary, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    @staticmethod
    def to_string(result: SimulationResult) -> str:
        """
        将单个模拟结果转换为YAML字符串
        """
        YAMLExporter._check_yaml()
        data = JSONExporter._result_to_dict(result)
        return yaml.dump(data, allow_unicode=True, default_flow_style=False, sort_keys=False)

    @staticmethod
    def batch_to_string(results: List[SimulationResult]) -> str:
        """
        将批量模拟结果转换为YAML字符串
        """
        YAMLExporter._check_yaml()
        data = {
            "count": len(results),
            "results": [JSONExporter._result_to_dict(r) for r in results],
        }
        return yaml.dump(data, allow_unicode=True, default_flow_style=False, sort_keys=False)
