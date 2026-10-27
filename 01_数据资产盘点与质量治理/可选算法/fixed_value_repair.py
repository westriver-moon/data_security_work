"""固定替代与边界修正：只修复不满足调用方范围的数值。

用途：在已有业务范围和共同回退值时，给缺失、非有限和越界记录提供可解释的估计。
输入：带索引的 pandas Series、有限下界、可选有限上界、范围内的固定替代值。
输出：保留索引、名称与合法值的数值 Series；不修改输入。
步骤：识别非法值，按统一常数替代；可选择把有限的超上界值截到上界。
限制：替代值不是实际观测；不能用统计贡献边界代替数据的真实有效范围。
本例是独立的图书馆阅读时长数据，常数和字段均不构成项目答案。
"""

from numbers import Real
import numpy as np
import pandas as pd

DEMO_VALUES = pd.Series([15.0, -2.0, np.nan, 90.0, np.inf], index=[2, 4, 7, 9, 12], name="minutes")


def validate_bounds(lower, upper, replacement):
    """核验有限下界、可选有限上界及范围内的共同替代值，返回浮点配置。"""
    for label, value in (("lower", lower), ("replacement", replacement)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
            raise ValueError(f"{label} 必须为有限实数")
    if upper is not None and (
        isinstance(upper, (bool, np.bool_)) or not isinstance(upper, Real)
        or not np.isfinite(upper) or upper < lower
    ):
        raise ValueError("upper 必须为不小于 lower 的有限实数，或 None")
    if replacement < lower or (upper is not None and replacement > upper):
        raise ValueError("replacement 必须位于合法范围内")
    return float(lower), None if upper is None else float(upper), float(replacement)


def numeric_copy(values):
    """将数值序列转换为保留索引的浮点副本，无法解析的值作为待修复缺失值。"""
    if not isinstance(values, pd.Series):
        raise TypeError("values 必须为 pandas Series，以保留来源索引")
    converted = pd.to_numeric(values, errors="coerce")
    if pd.api.types.is_complex_dtype(converted.dtype):
        raise ValueError("values 不支持复数")
    return pd.Series(converted.to_numpy(dtype=float, na_value=np.nan, copy=True), index=values.index, name=values.name)


def repair_fixed(values, replacement, lower=0, upper=None, clip_above=False):
    """只替代缺失、非有限或越界值；可选将有限的超上界值截到上界，合法值保持不变。"""
    lower, upper, replacement = validate_bounds(lower, upper, replacement)
    if not isinstance(clip_above, bool):
        raise TypeError("clip_above 必须为 bool")
    if clip_above and upper is None:
        raise ValueError("clip_above=True 时必须给出 upper")
    result = numeric_copy(values)
    finite = np.isfinite(result)
    invalid = ~finite | result.lt(lower)
    if upper is not None:
        invalid |= result.gt(upper)
    result.loc[invalid] = replacement
    if clip_above:
        result.loc[finite & numeric_copy(values).gt(upper)] = upper
    return result


if __name__ == "__main__":
    print(repair_fixed(DEMO_VALUES, replacement=20, lower=0, upper=60, clip_above=True))
