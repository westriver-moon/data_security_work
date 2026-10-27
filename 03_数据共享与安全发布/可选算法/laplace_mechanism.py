"""纯差分隐私的Laplace机制及重复误差评估。

算法定义：若任意相邻数据集D和D'以及任意输出集合S都满足
P[M(D)∈S] ≤ exp(epsilon)×P[M(D')∈S]，随机机制M满足epsilon-差分隐私。Laplace机制对数值
查询f(D)加入均值为0、尺度b=Δf/epsilon的Laplace噪声，其中Δf是全局敏感度。

核心步骤：
1. 确定保护单位与相邻关系，例如增删一个单位或替换一个单位。
2. 通过裁剪或贡献限制计算查询的全局敏感度Δf。
3. 根据epsilon计算噪声尺度b，并从Laplace(0,b)独立采样。
4. 将噪声加到真实统计值上；重复采样时再统计平均绝对误差和误差标准差。

输入与输出：输入是真实统计值、敏感度、epsilon、试验次数和模拟种子；输出把一次带噪发布值
与多次独立模拟的平均绝对误差和误差标准差分开记录。随机种子只用于复现实验，不应在真实
连续发布中固定复用。

关键参数：epsilon越小，输出分布越接近、隐私越强，但噪声尺度越大。敏感度必须由查询定义
和贡献上界推导，不能依据当前数据的偶然范围直接估计。单次生成t个样本的复杂度为O(t)。

安全边界：差分隐私限制在约定相邻关系下，一个保护单位的改变对输出分布的影响。增删一个
单位与替换一个单位需要分别说明；敏感度须依据该相邻关系、查询定义和贡献范围推导。
它不负责访问控制、传输安全或原始数据加密。多次查询会发生隐私损失组合，需要隐私预算账户；
对带噪结果进行裁剪、取整或重复挑选也要分析偏差和后处理流程。
"""

import numpy as np

# 独立演示数据：直接在文件开头构造。
DEMO_TRUE_VALUE = 40.0
DEMO_SENSITIVITY = 1.0
DEMO_EPSILONS = [0.2, 0.5, 1.0]
DEMO_TRIALS = 200
DEMO_SEED = 510


def _noise_scale(true_value, sensitivity, epsilon):
    """检查有限实数、机制参数与噪声尺度，返回可用的非负尺度。
        布尔值不作为数值预算，超出有限精度范围时明确拒绝。
    """
    import math
    from numbers import Real

    values = [true_value, sensitivity, epsilon]
    for value in values:
        try:
            finite_real = (not isinstance(value, (bool, np.bool_))
                           and isinstance(value, Real) and math.isfinite(value))
        except (OverflowError, TypeError, ValueError):
            finite_real = False
        if not finite_real:
            raise ValueError("真值、敏感度和隐私参数必须为可表示的有限实数")
    if sensitivity < 0 or epsilon <= 0:
        raise ValueError("敏感度须非负且epsilon须为正")
    scale = float(sensitivity) / float(epsilon)
    if not math.isfinite(scale):
        raise ValueError("噪声尺度超出有限数值范围")
    return scale


def laplace_error_summary(
    true_value: float, sensitivity: float, epsilon: float,
    trials: int, seed: int,
) -> dict[str, float | int]:
    """返回一次系统随机发布值，以及由独立模拟随机源得到的误差统计。
    seed仅控制本地模拟，真值、误差和模拟结果不能作为正式对外发布结果。
    """
    from numbers import Integral

    scale = _noise_scale(true_value, sensitivity, epsilon)
    if isinstance(trials, (bool, np.bool_)) or not isinstance(trials, Integral) or trials <= 0:
        raise ValueError("trials必须为正整数")
    released_value = laplace_release(true_value, sensitivity, epsilon)
    evaluation_rng = np.random.default_rng(seed)
    simulated_values = true_value + evaluation_rng.laplace(0, scale, size=trials)
    if not np.isfinite(simulated_values).all():
        raise ValueError("误差模拟产生非有限值")
    with np.errstate(over="ignore", invalid="ignore"):
        simulation_errors = simulated_values - true_value
        mean_absolute_error = float(np.abs(simulation_errors).mean())
        error_std = float(simulation_errors.std())
    absolute_error = float(abs(released_value - true_value))
    if not np.isfinite([mean_absolute_error, error_std, absolute_error]).all():
        raise ValueError("误差统计超出有限数值范围")
    return {
        "true_value": float(true_value),
        "epsilon": float(epsilon),
        "released_value": float(released_value),
        "absolute_error": absolute_error,
        "trial_count": int(trials),
        "mean_absolute_error": mean_absolute_error,
        "error_std": error_std,
    }


def laplace_release(true_value, sensitivity, epsilon):
    """使用系统随机源返回一个有限带噪标量；不返回真值、噪声或种子。
    数值采样是教学实现，有限精度采样不是经过审计的生产DP实现。
    """
    import math
    import secrets

    scale = _noise_scale(true_value, sensitivity, epsilon)
    # 开区间避免log(0)，系统随机源与误差模拟随机源分离。
    u = (secrets.randbelow(2**53 - 1) + 1) / 2**53 - 0.5
    noise = -scale * math.copysign(1, u) * math.log1p(-2 * abs(u))
    released_value = float(true_value + noise)
    if not math.isfinite(released_value):
        raise ValueError("发布结果超出有限数值范围")
    return released_value


if __name__ == "__main__":
    for demo_epsilon in DEMO_EPSILONS:
        print(
            laplace_error_summary(
                DEMO_TRUE_VALUE,
                DEMO_SENSITIVITY,
                demo_epsilon,
                trials=DEMO_TRIALS,
                seed=DEMO_SEED,
            )
        )
