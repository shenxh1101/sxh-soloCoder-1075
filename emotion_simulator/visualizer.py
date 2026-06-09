"""
可视化模块
生成ASCII情绪演进曲线图
"""

from typing import List, Dict, Optional
from .models import SimulationResult, EmotionDimension


class ASCIIVisualizer:
    """
    ASCII可视化器
    生成终端可显示的情绪演进曲线图
    """

    def __init__(
        self,
        width: int = 80,
        height: int = 15,
        show_legend: bool = True,
    ):
        self.width = width
        self.height = height
        self.show_legend = show_legend

    def _get_symbols(self) -> List[str]:
        """获取不同维度的绘制符号"""
        return ["●", "■", "▲", "◆", "★", "○", "□", "△", "◇", "☆"]

    def _get_colors(self) -> List[str]:
        """获取ANSI颜色代码（如果终端支持）"""
        return [
            "\033[91m",
            "\033[92m",
            "\033[94m",
            "\033[93m",
            "\033[95m",
            "\033[96m",
            "\033[97m",
            "\033[90m",
        ]

    def _colorize(self, text: str, color_code: str, use_color: bool = True) -> str:
        """为文本添加颜色"""
        if use_color:
            return f"{color_code}{text}\033[0m"
        return text

    def _normalize_series(
        self,
        series: List[float],
        min_val: float,
        max_val: float,
    ) -> List[int]:
        """将数值序列归一化到图表高度"""
        if not series:
            return []

        normalized = []
        for val in series:
            if max_val == min_val:
                norm = 0.5
            else:
                norm = (val - min_val) / (max_val - min_val)
            normalized.append(int(norm * (self.height - 1)))
        return normalized

    def _get_axis_labels(
        self,
        dimension: EmotionDimension,
        min_val: float,
        max_val: float,
    ) -> List[str]:
        """获取Y轴标签"""
        labels = []
        for i in range(self.height):
            ratio = 1 - (i / (self.height - 1))
            val = min_val + ratio * (max_val - min_val)
            if i == 0:
                labels.append(f"{val:5.2f} {dimension.high_label}")
            elif i == self.height - 1:
                labels.append(f"{val:5.2f} {dimension.low_label}")
            elif i == self.height // 2:
                labels.append(f"{val:5.2f} 中性")
            else:
                labels.append(f"{val:5.2f}")
        return labels

    def plot_single(
        self,
        result: SimulationResult,
        dimension_name: Optional[str] = None,
        use_color: bool = True,
    ) -> str:
        """
        绘制单个模拟结果的情绪曲线图
        """
        if not result.steps:
            return "无数据可绘制"

        if dimension_name is None:
            dimension_name = result.dimensions[0].name

        dimension = next((d for d in result.dimensions if d.name == dimension_name), None)
        if not dimension:
            return f"未找到情绪维度: {dimension_name}"

        series = result.get_emotion_series(dimension_name)
        num_points = len(series)

        min_val = min(series + [dimension.min_val])
        max_val = max(series + [dimension.max_val])

        normalized = self._normalize_series(series, min_val, max_val)
        y_labels = self._get_axis_labels(dimension, min_val, max_val)

        if num_points == 1:
            chart_width = 5
            sampled_series = series
            sampled_normalized = normalized
            sampled_indices = [0]
        else:
            chart_width = min(self.width - 15, num_points * 2)
            step_interval = max(1, num_points // chart_width)

            sampled_series = [series[i] for i in range(0, num_points, step_interval)]
            sampled_normalized = [normalized[i] for i in range(0, num_points, step_interval)]
            sampled_indices = [i for i in range(0, num_points, step_interval)]

            if len(sampled_series) < 2:
                sampled_series = series[:2]
                sampled_normalized = normalized[:2]
                sampled_indices = [0, num_points - 1]

        grid = [[" "] * len(sampled_series) for _ in range(self.height)]

        for i, (y, idx) in enumerate(zip(sampled_normalized, sampled_indices)):
            grid_y = self.height - 1 - y
            if 0 <= grid_y < self.height and 0 <= i < len(grid[grid_y]):
                symbol = self._get_symbols()[0]
                color = self._get_colors()[0]
                grid[grid_y][i] = self._colorize(symbol, color, use_color)

        if len(sampled_normalized) > 1:
            for i in range(len(sampled_normalized) - 1):
                y1 = sampled_normalized[i]
                y2 = sampled_normalized[i + 1]
                min_y = min(y1, y2)
                max_y = max(y1, y2)

                for y in range(min_y, max_y + 1):
                    grid_y = self.height - 1 - y
                    if 0 <= grid_y < self.height and 0 <= i < len(grid[grid_y]):
                        if grid[grid_y][i] == " ":
                            grid[grid_y][i] = "│"

        lines = []
        title = f"情绪演进曲线 - {dimension.high_label} vs {dimension.low_label}"
        lines.append(title)
        lines.append("=" * (len(title) + 20))

        for row_idx, row in enumerate(grid):
            label = y_labels[row_idx]
            line = f"{label:15s} │" + "".join(row)
            lines.append(line)

        x_axis = " " * 15 + "└" + "─" * len(sampled_series)
        lines.append(x_axis)

        x_labels = " " * 15 + " "
        if len(sampled_indices) == 1:
            x_labels += "  0"
        else:
            if len(sampled_indices) > 10:
                display_indices = sampled_indices[::len(sampled_indices) // 10]
            else:
                display_indices = sampled_indices

            pos = 0
            for idx in display_indices:
                label = str(idx)
                x_labels += " " * (pos - len(x_labels) + 16) + label
                pos += len(sampled_series) // len(display_indices)

        lines.append(x_labels)
        lines.append(" " * 15 + " " + "传递次数")

        return "\n".join(lines)

    def plot_comparison(
        self,
        results: List[SimulationResult],
        dimension_name: Optional[str] = None,
        labels: Optional[List[str]] = None,
        use_color: bool = True,
    ) -> str:
        """
        绘制多个模拟结果的对比曲线图
        """
        if not results:
            return "无数据可绘制"

        if dimension_name is None:
            dimension_name = results[0].dimensions[0].name

        dimension = next((d for d in results[0].dimensions if d.name == dimension_name), None)
        if not dimension:
            return f"未找到情绪维度: {dimension_name}"

        all_series = []
        for result in results:
            series = result.get_emotion_series(dimension_name)
            all_series.append(series)

        num_points = max(len(s) for s in all_series)
        min_val = min(min(s) for s in all_series + [[dimension.min_val]])
        max_val = max(max(s) for s in all_series + [[dimension.max_val]])

        chart_width = min(self.width - 15, num_points * 2)
        step_interval = max(1, num_points // chart_width)

        grid = [[" "] * (num_points // step_interval + 1) for _ in range(self.height)]
        y_labels = self._get_axis_labels(dimension, min_val, max_val)

        symbols = self._get_symbols()
        colors = self._get_colors()

        for result_idx, series in enumerate(all_series):
            normalized = self._normalize_series(series, min_val, max_val)
            sampled_normalized = [normalized[i] for i in range(0, len(series), step_interval)]

            symbol = symbols[result_idx % len(symbols)]
            color = colors[result_idx % len(colors)]

            for i, y in enumerate(sampled_normalized):
                grid_y = self.height - 1 - y
                if 0 <= grid_y < self.height and 0 <= i < len(grid[grid_y]):
                    grid[grid_y][i] = self._colorize(symbol, color, use_color)

        lines = []
        title = f"情绪演进对比 - {dimension.high_label} vs {dimension.low_label}"
        lines.append(title)
        lines.append("=" * (len(title) + 20))

        for row_idx, row in enumerate(grid):
            label = y_labels[row_idx]
            line = f"{label:15s} │" + "".join(row)
            lines.append(line)

        x_axis = " " * 15 + "└" + "─" * (num_points // step_interval + 1)
        lines.append(x_axis)
        lines.append(" " * 15 + " " + "传递次数")

        if self.show_legend:
            lines.append("")
            lines.append("图例:")
            for i, result in enumerate(results):
                symbol = symbols[i % len(symbols)]
                color = colors[i % len(colors)]
                label = labels[i] if labels and i < len(labels) else result.initial_sentence[:20]
                colored_symbol = self._colorize(symbol, color, use_color)
                lines.append(f"  {colored_symbol} - {label}")

        return "\n".join(lines)

    def plot_multi_dimension(
        self,
        result: SimulationResult,
        use_color: bool = True,
    ) -> str:
        """
        绘制单个模拟结果的多维度情绪曲线图
        """
        if not result.steps:
            return "无数据可绘制"

        series_dict = {}
        for dim in result.dimensions:
            series_dict[dim.name] = result.get_emotion_series(dim.name)

        num_points = len(result.steps)
        all_values = [v for series in series_dict.values() for v in series]
        min_val = min(all_values)
        max_val = max(all_values)

        chart_width = min(self.width - 20, num_points * 2)
        step_interval = max(1, num_points // chart_width)

        grid = [[" "] * (num_points // step_interval + 1) for _ in range(self.height)]

        symbols = self._get_symbols()
        colors = self._get_colors()

        for dim_idx, (dim_name, series) in enumerate(series_dict.items()):
            normalized = self._normalize_series(series, min_val, max_val)
            sampled_normalized = [normalized[i] for i in range(0, len(series), step_interval)]

            symbol = symbols[dim_idx % len(symbols)]
            color = colors[dim_idx % len(colors)]

            for i, y in enumerate(sampled_normalized):
                grid_y = self.height - 1 - y
                if 0 <= grid_y < self.height and 0 <= i < len(grid[grid_y]):
                    if grid[grid_y][i] == " ":
                        grid[grid_y][i] = self._colorize(symbol, color, use_color)
                    else:
                        grid[grid_y][i] = self._colorize("*", "\033[97m", use_color)

        lines = []
        title = "多维度情绪演进曲线"
        lines.append(title)
        lines.append("=" * (len(title) + 20))

        y_labels = []
        for i in range(self.height):
            ratio = 1 - (i / (self.height - 1))
            val = min_val + ratio * (max_val - min_val)
            y_labels.append(f"{val:5.2f}")

        for row_idx, row in enumerate(grid):
            line = f"{y_labels[row_idx]:8s} │" + "".join(row)
            lines.append(line)

        x_axis = " " * 8 + "└" + "─" * (num_points // step_interval + 1)
        lines.append(x_axis)
        lines.append(" " * 8 + " " + "传递次数")

        if self.show_legend:
            lines.append("")
            lines.append("图例:")
            for i, dim in enumerate(result.dimensions):
                symbol = symbols[i % len(symbols)]
                color = colors[i % len(colors)]
                colored_symbol = self._colorize(symbol, color, use_color)
                lines.append(f"  {colored_symbol} - {dim.high_label} vs {dim.low_label}")

        return "\n".join(lines)

    def print_step_info(
        self,
        step,
        show_scores: bool = True,
        use_color: bool = True,
    ) -> str:
        """
        格式化单步传递信息
        """
        lines = []

        if step.step == 0:
            header = "[初始句子 (Step 0)]"
        else:
            header = f"[第 {step.step} 次传递]"
        lines.append(header)

        if step.changed_word:
            original, new = step.changed_word
            change_line = f"  变化: {original} → {new}"
            if use_color:
                change_line = f"  变化: \033[91m{original}\033[0m → \033[92m{new}\033[0m"
            lines.append(change_line)

        lines.append(f"  句子: {step.modified_sentence}")

        if show_scores:
            score_line = "  情绪评分: "
            scores = []
            colors = self._get_colors()
            for i, (dim_name, score) in enumerate(step.emotion_scores.items()):
                color = colors[i % len(colors)]
                score_str = f"{dim_name}: {score:+.2f}"
                if use_color:
                    score_str = f"{color}{dim_name}: {score:+.2f}\033[0m"
                scores.append(score_str)
            score_line += ", ".join(scores)
            lines.append(score_line)

        return "\n".join(lines)
