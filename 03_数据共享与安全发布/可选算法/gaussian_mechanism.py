"""近似差分隐私的Gaussian机制及重复误差评估。

算法定义：对任意相邻数据集D、D'和任意输出事件S，(epsilon, delta)-差分隐私要求
P[M(D)∈S] ≤ exp(epsilon)×P[M(D')∈S] + delta。概率来自机制的随机性；
delta是概率不等式中的加性松弛项，不能直接等同于一般意义下的“隐私保证失败概率”。
基础Gaussian机制向数值查询加入N(0, sigma²)噪声；常见充分条件为
sigma = Δf×sqrt(2×ln(1.25/delta))/epsilon，其中Δf通常是L2敏感度。

核心步骤：
1. 明确定义相邻数据集和查询函数，并裁剪单条记录的贡献。
2. 根据取值上下界、样本规模或梯度范数计算L2敏感度。
3. 用epsilon、delta和敏感度计算噪声标准差sigma。
4. 独立采样正态噪声加入真实值，并通过重复试验汇总误差分布。

输入与输出：输入是真实值、敏感度、epsilon、delta、试验次数和随机种子；输出把一次带噪
发布值与多次独立模拟的平均绝对误差、误差标准差分开记录，并保留使用的隐私参数。

关键参数：epsilon控制主要隐私强度；delta应是很小的正数，并结合总体规模选择。对固定的
epsilon和敏感度，delta越小通常需要越大的sigma。生成t个噪声样本的时间与空间复杂度均为
O(t)。

安全边界：基础公式不是所有参数范围下最紧的校准方法，多轮机制需要RDP、隐私损失分布等
会计工具处理组合。若输入未裁剪或敏感度计算错误，声明的保证不成立。固定随机种子只适合
复现实验，真实发布必须使用安全随机源并管理整体隐私预算。
"""

import numpy as np

# 独立演示数据：直接在文件开头构造。
DEMO_TRUE_VALUE = 8.0
DEMO_LOWER_BOUND = 0.0
DEMO_UPPER_BOUND = 72.0
DEMO_COUNT = 180
DEMO_EPSILON = 1.0
DEMO_DELTA = 1e-5
DEMO_TRIALS = 200
DEMO_SEED = 510


def _noise_scale(true_value, sensitivity, epsilon, delta):
    """检查有限实数、机制参数与噪声尺度，返回可用的非负尺度。
        布尔值不作为数值预算，超出有限精度范围时明确拒绝。
    """
    import math
    from numbers import Real

    values = [true_value, sensitivity, epsilon, delta]
    for value in values:
        try:
            finite_real = (not isinstance(value, (bool, np.bool_))
                           and isinstance(value, Real) and math.isfinite(value))
        except (OverflowError, TypeError, ValueError):
            finite_real = False
        if not finite_real:
            raise ValueError("真值、敏感度和隐私参数必须为可表示的有限实数")
    if sensitivity < 0 or not 0 < epsilon <= 1 or not 0 < delta < 1:
        raise ValueError("敏感度须非负，Gaussian须0<epsilon<=1且0<delta<1")
    scale = float(sensitivity) * math.sqrt(2 * (math.log(1.25) - math.log(float(delta)))) / float(epsilon)
    if not math.isfinite(scale):
        raise ValueError("噪声尺度超出有限数值范围")
    return scale


def gaussian_error_summary(
    true_value: float, sensitivity: float, epsilon: float,
    delta: float, trials: int, seed: int,
) -> dict[str, float | int]:
    """返回一次系统随机发布值，以及由独立模拟随机源得到的误差统计。
    seed仅控制本地模拟，真值、误差和模拟结果不能作为正式对外发布结果。
    """
    from numbers import Integral

    scale = _noise_scale(true_value, sensitivity, epsilon, delta)
    if isinstance(trials, (bool, np.bool_)) or not isinstance(trials, Integral) or trials <= 0:
        raise ValueError("trials必须为正整数")
    released_value = gaussian_release(true_value, sensitivity, epsilon, delta)
    evaluation_rng = np.random.default_rng(seed)
    simulated_values = true_value + evaluation_rng.normal(0, scale, size=trials)
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
        "delta": float(delta),
        "released_value": float(released_value),
        "absolute_error": absolute_error,
        "trial_count": int(trials),
        "mean_absolute_error": mean_absolute_error,
        "error_std": error_std,
    }


def gaussian_release(true_value, sensitivity, epsilon, delta):
    """使用系统随机源返回一个有限带噪标量；不返回真值、噪声或种子。
    数值采样是教学实现，基础Gaussian公式仅接受0<epsilon<=1。
    """
    import math
    import secrets

    scale = _noise_scale(true_value, sensitivity, epsilon, delta)
    noise = secrets.SystemRandom().gauss(0, scale)
    released_value = float(true_value + noise)
    if not math.isfinite(released_value):
        raise ValueError("发布结果超出有限数值范围")
    return released_value


if __name__ == "__main__":
    # 有界均值敏感度由取值跨度除以样本数得到。
    demo_sensitivity = (DEMO_UPPER_BOUND - DEMO_LOWER_BOUND) / DEMO_COUNT
    print(
        gaussian_error_summary(
            DEMO_TRUE_VALUE,
            demo_sensitivity,
            DEMO_EPSILON,
            DEMO_DELTA,
            trials=DEMO_TRIALS,
            seed=DEMO_SEED,
        )
    )
