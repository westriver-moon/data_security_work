"""基于共同参考样本的分组中位数估计。

用途：同类记录尺度不同，且多个使用方必须采用同一估计依据时，先拟合共同统计再应用。
输入：参考数值与分组 Series、合法范围、最小组样本数和共同固定回退值；应用时传入
待修复的数值与分组以及拟合结果。输出始终保留待修复值的索引、名称和合法值。
步骤：排除参考样本的非法值，计算组中位数；组内有效样本不足时用共同全局中位数，
全局也没有有效样本时使用给定固定回退值。分组缺失或新组同样走全局回退。
限制：统计参考必须先按业务单位去重，且不能用未来标签分组；中位数估计不是真实观测。
本例使用独立的书籍类别与阅读时长；算法不读取项目数据，也不替调用方选择业务口径。
"""

from numbers import Integral
import numpy as np
import pandas as pd

if __package__:
    from .fixed_value_repair import numeric_copy, validate_bounds
else:
    from fixed_value_repair import numeric_copy, validate_bounds

DEMO_REFERENCE = pd.Series([10, 20, 30, 50, -1], index=[0, 1, 2, 3, 4], name="minutes")
DEMO_GROUPS = pd.Series(["story", "story", "story", "science", "science"], index=DEMO_REFERENCE.index)


def validate_groups(values, groups):
    """检查分组与数值的索引严格对应，并允许缺失组回退到共同全局统计。"""
    if not isinstance(groups, pd.Series) or not groups.index.equals(values.index):
        raise ValueError("groups 必须为与 values 索引及顺序相同的 Series")
    for group in groups:
        if group is None or (isinstance(group, float) and np.isnan(group)) or group is pd.NA:
            continue
        if not isinstance(group, str) or not group.strip():
            raise ValueError("分组必须为非空字符串或缺失值")


def finite_median(values):
    """计算有限数值的中位数，偶数样本采用分半相加以避免大数相加溢出。"""
    ordered = sorted(float(value) for value in values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else ordered[middle - 1] / 2 + ordered[middle] / 2


def build_group_reference(values, groups, lower=0, upper=None, fallback=0, min_samples=3):
    """从共同参考样本拟合有效组中位数和全局回退统计，不使用待修复表的本地统计。"""
    lower, upper, fallback = validate_bounds(lower, upper, fallback)
    if isinstance(min_samples, (bool, np.bool_)) or not isinstance(min_samples, Integral) or min_samples < 1:
        raise ValueError("min_samples 必须为正整数")
    numeric = numeric_copy(values)
    validate_groups(numeric, groups)
    valid = np.isfinite(numeric) & numeric.ge(lower)
    if upper is not None:
        valid &= numeric.le(upper)
    valid_values = numeric.loc[valid]
    global_median = finite_median(valid_values) if len(valid_values) else fallback
    samples = {}
    for value, group, usable in zip(numeric, groups, valid):
        if usable and isinstance(group, str):
            samples.setdefault(group, []).append(float(value))
    return {
        "lower": lower, "upper": upper, "fallback": fallback,
        "global_median": global_median, "valid_count": int(valid.sum()),
        "min_samples": int(min_samples),
        "group_counts": {group: len(items) for group, items in samples.items()},
        "group_medians": {group: finite_median(items) for group, items in samples.items() if len(items) >= min_samples},
    }


def repair_group_median(values, groups, reference):
    """用已拟合的共同组统计修复非法值，新组、缺失组或小样本组使用同一全局回退。"""
    numeric = numeric_copy(values)
    validate_groups(numeric, groups)
    required = {"lower", "upper", "global_median", "group_medians"}
    if not isinstance(reference, dict) or not required <= reference.keys():
        raise ValueError("reference 必须包含 build_group_reference 生成的范围与中位数")
    lower, upper, global_median = validate_bounds(reference["lower"], reference["upper"], reference["global_median"])
    medians = reference["group_medians"]
    if not isinstance(medians, dict):
        raise ValueError("group_medians 必须为字典")
    for group, value in medians.items():
        if not isinstance(group, str) or not group.strip():
            raise ValueError("参考组名必须为非空字符串")
        validate_bounds(lower, upper, value)
    invalid = ~np.isfinite(numeric) | numeric.lt(lower)
    if upper is not None:
        invalid |= numeric.gt(upper)
    for position, (bad, group) in enumerate(zip(invalid, groups)):
        if bad:
            numeric.iloc[position] = medians.get(group, global_median) if isinstance(group, str) else global_median
    return numeric


if __name__ == "__main__":
    reference = build_group_reference(DEMO_REFERENCE, DEMO_GROUPS, upper=120, fallback=25, min_samples=2)
    values = pd.Series([-4, 18, np.inf], index=[8, 9, 10], name="minutes")
    groups = pd.Series(["story", "science", "new"], index=values.index)
    print(repair_group_median(values, groups, reference))
