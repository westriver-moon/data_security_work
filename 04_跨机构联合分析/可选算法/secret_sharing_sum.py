"""基于一次多项式秘密共享的安全求和演示。

算法定义：在素数有限域中，用一次多项式f(x)=s+a×x mod p隐藏秘密s。任意一个份额(x,f(x))
不能唯一确定s，两份不同横坐标的份额可以通过拉格朗日插值恢复f(0)=s，因此这是(2,3)阈值
共享的教学形式。

核心步骤：
1. 为每个输入整数从整个有限域（包括0）随机选择斜率a，在x=1、2、3处生成三份份额。
2. 各份额持有者分别把来自所有输入的份额相加，得到三个聚合份额。
3. 使用x=1和x=2的两个聚合份额插值；一次多项式下可写为2×y1-y2 mod p。
4. 只返回所有输入的模p总和，不逐项重构单个输入。

输入与输出：输入是两方整数列表和随机种子；输出是一个整数合计。若真实总和可能达到或
超过p，需要设计有符号编码、范围限制或更大的域，否则模运算会产生回绕。

关键参数：p必须为足够大的素数；随机斜率必须来自安全随机源。若共有n个值、生成s份份额，
份额生成复杂度为O(n×s)，重构成本取决于阈值，本示例阈值固定为2。

安全边界：低于阈值的独立份额在信息论意义上不暴露秘密，但该代码在一个进程内模拟各方，
且使用可复现随机数，不构成真实通信协议。它没有认证、恶意份额验证、范围证明、串谋分析、
掉线恢复和隐私输出控制；最终合计本身也可能在参与方过少时泄露单方值。
"""

import numpy as np

# 独立演示数据：直接在文件开头构造。
DEMO_LEFT_VALUES = [100, 200]
DEMO_RIGHT_VALUES = [300, 400]
DEMO_SEED = 510

PRIME = 2_147_483_647


def _add_party_shares(
    values: list[int], share_sums: dict[int, int], rng: np.random.Generator
) -> None:
    """每个值使用一次多项式f(x)=value+slope*x生成3份份额。"""
    for value in values:
        # 接受的NumPy整数先转为Python整数，避免窄整数在有限域运算中溢出。
        value = int(value)
        slope = int(rng.integers(0, PRIME))
        for x in share_sums:
            share_sums[x] = (share_sums[x] + value + slope * x) % PRIME


def two_party_secure_sum(left_values: list[int], right_values: list[int], seed: int) -> int:
    """聚合两方份额，并使用x=1、2两份总份额在x=0处重构。"""
    values = [*left_values, *right_values]
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) for value in values):
        raise TypeError("秘密共享求和只接受整数输入")
    if any(int(value) < 0 for value in values):
        raise ValueError("本教学实现未定义负数编码")
    if sum(int(value) for value in values) >= PRIME:
        raise ValueError("输入合计必须小于有限域素数，避免模回绕")
    rng = np.random.default_rng(seed)
    share_sums = {1: 0, 2: 0, 3: 0}
    _add_party_shares(left_values, share_sums, rng)
    _add_party_shares(right_values, share_sums, rng)
    return int((2 * share_sums[1] - share_sums[2]) % PRIME)


if __name__ == "__main__":
    print("明文合计：", sum(DEMO_LEFT_VALUES) + sum(DEMO_RIGHT_VALUES))
    print(
        "安全合计：", two_party_secure_sum(DEMO_LEFT_VALUES, DEMO_RIGHT_VALUES, seed=DEMO_SEED)
    )
