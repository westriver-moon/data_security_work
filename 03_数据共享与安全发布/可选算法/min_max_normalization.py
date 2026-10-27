"""Min-Max数值归一化。

算法背景：不同特征可能具有完全不同的量纲和取值范围。Min-Max归一化把训练数据中的最小值
映射到目标区间下界，把最大值映射到上界，其余数值按线性比例转换。默认目标区间为[0,1]，
公式为(x-min)/(max-min)。

核心步骤：①从训练数据计算每列最小值和最大值；②保存这些统计量；③用同一组统计量转换
训练数据和以后到来的数据。为了避免数据泄漏，测试集不能参与最小值和最大值的拟合。

输入与输出：输入是二维数值矩阵和目标区间，输出是归一化矩阵以及每列的最小值、最大值。
新样本可能超出训练范围，因此转换结果也可能低于0或高于1；是否裁剪应由具体模型决定。

作用与局限：归一化有助于距离计算和梯度优化，但它是可逆的线性尺度变化。只要知道最小值
和最大值，就可还原原值；即使不知道，也会保留顺序、相对距离和异常点。因此它不提供加密、
匿名化或差分隐私保证。常数列还需要单独处理，避免除以零。
"""

import numpy as np

# 独立演示数据：两列分别具有不同量纲，第二行展示中间位置。
DEMO_VALUES = np.array(
    [
        [10.0, 1_000.0],
        [20.0, 3_000.0],
        [30.0, 5_000.0],
    ]
)
DEMO_FEATURE_RANGE = (0.0, 1.0)


def fit_min_max(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """从二维训练矩阵计算逐列最小值和最大值。"""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or 0 in values.shape or not np.isfinite(values).all():
        raise ValueError("values必须是非空且全部有限的二维数值矩阵")
    return values.min(axis=0), values.max(axis=0)


def transform_min_max(
    values: np.ndarray,
    minimum: np.ndarray,
    maximum: np.ndarray,
    feature_range: tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """使用已经拟合的统计量转换数据；常数列统一映射到目标下界。"""
    values = np.asarray(values, dtype=float)
    minimum, maximum = np.asarray(minimum, dtype=float), np.asarray(maximum, dtype=float)
    bounds = np.asarray(feature_range, dtype=float)
    if values.ndim != 2 or values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError("values必须是具有特征列的有限二维数值矩阵")
    if (minimum.shape != (values.shape[1],) or maximum.shape != minimum.shape
            or not np.isfinite(minimum).all() or not np.isfinite(maximum).all()
            or (minimum > maximum).any()):
        raise ValueError("逐列统计量必须有限、形状匹配且最小值不大于最大值")
    if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
        raise ValueError("目标区间须为两个有限数，且下界小于上界")
    lower, upper = bounds
    # 中间运算也须有限：跨度溢出为inf时，除法可能静默产生看似正常的0。
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            span = maximum - minimum
            varying = span != 0
            unit_values = np.zeros_like(values)
            # 常数列直接保持0，避免对最后会舍弃的差值做无意义的溢出运算。
            unit_values[:, varying] = (values[:, varying] - minimum[varying]) / span[varying]
            scaled = lower + unit_values * (upper - lower)
    except FloatingPointError as exc:
        raise ValueError("归一化中间运算超出有限数值范围") from exc
    if not np.isfinite(scaled).all():
        raise ValueError("归一化结果超出有限数值范围")
    return scaled


if __name__ == "__main__":
    demo_minimum, demo_maximum = fit_min_max(DEMO_VALUES)
    demo_scaled = transform_min_max(DEMO_VALUES, demo_minimum, demo_maximum, DEMO_FEATURE_RANGE)
    print("minimum:", demo_minimum)
    print("maximum:", demo_maximum)
    print("scaled:\n", demo_scaled)
