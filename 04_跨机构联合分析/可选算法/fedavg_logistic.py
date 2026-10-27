"""用于二分类逻辑回归的联邦平均FedAvg。

算法定义：FedAvg让K个客户端从同一全局模型出发，各自在本地数据上执行若干步优化，然后
服务器按样本量加权平均本地模型。第t轮可写为w(t+1)=Σk(nk/N)×wk，其中nk是客户端k的
样本数，N是参与本轮训练的总样本数。

核心步骤：
1. 将全局权重和偏置初始化为0。
2. 每轮把当前全局参数复制给所有客户端。
3. 每个客户端使用二元交叉熵梯度执行指定次数的本地全批量梯度下降。
4. 服务器分别对权重和偏置做样本量加权平均，得到下一轮全局模型。
5. 重复通信轮次，最终返回全局权重与偏置。

输入与输出：输入是多个“特征矩阵、二元标签”客户端数据对，以及通信轮数、本地轮数和
学习率；输出是逻辑回归参数。项目还传入aggregate_models回调执行第11题；独立示例省略回调时使用默认聚合。每轮计算量约为所有客户端本地样本计算量之和，通信量与
客户端数量、轮数和参数维度成正比。

关键参数：学习率过大可能发散，过小会收敛缓慢；增加本地轮数可减少通信频率，但在各方
数据非独立同分布时可能造成客户端漂移。按样本量加权会让大客户端对全局模型影响更大。

安全边界：联邦学习只改变数据所在位置，不自动提供隐私保证。模型更新可能泄露训练样本，
恶意客户端也可能实施投毒或后门攻击。本实现没有客户端抽样、动量、正则化、安全聚合、
差分隐私、认证和鲁棒聚合，适合解释FedAvg核心过程而非生产训练。
"""

import numpy as np

# 独立演示数据：直接在文件开头构造。
DEMO_CLIENTS = [
    (np.array([[0.0], [1.0]]), np.array([0.0, 1.0])),
    (np.array([[2.0], [3.0]]), np.array([1.0, 1.0])),
]
DEMO_ROUNDS = 5
DEMO_LOCAL_EPOCHS = 2
DEMO_LEARNING_RATE = 0.1


class AggregationPending(Exception):
    """项目回调未作答；不生成冒充完成的联邦模型。"""


def fedavg_train(
    clients: list[tuple[np.ndarray, np.ndarray]],
    rounds: int,
    local_epochs: int,
    learning_rate: float,
    aggregate_models=None,
) -> tuple[np.ndarray, float]:
    """使用普通二元交叉熵本地更新；逐轮调用聚合回调，None结果明确表示未完成。

    aggregate_models省略时仅供独立示例使用默认聚合；项目入口必须传第11题函数。
    """
    if not clients:
        raise ValueError("至少需要一个客户端")
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
           or value < 1 for value in (rounds, local_epochs)):
        raise ValueError("通信轮数与本地轮数必须是正整数")
    if (isinstance(learning_rate, (bool, np.bool_))
            or not np.isscalar(learning_rate) or not np.isfinite(learning_rate)
            or learning_rate <= 0):
        raise ValueError("学习率必须是有限正数")
    checked_clients = []
    width = None
    for features, labels in clients:
        features, labels = np.asarray(features, dtype=float), np.asarray(labels, dtype=float)
        if features.ndim != 2 or features.shape[1] == 0 or labels.ndim != 1:
            raise ValueError("特征必须为非零列二维矩阵，标签必须为一维向量")
        if len(features) != len(labels):
            raise ValueError("每个客户端特征与标签的行数必须相同")
        if width is not None and features.shape[1] != width:
            raise ValueError("各客户端特征列数必须一致")
        width = features.shape[1]
        if not np.isfinite(features).all() or not np.isin(labels, [0, 1]).all():
            raise ValueError("特征必须有限，标签只能为0或1")
        if len(labels):
            checked_clients.append((features, labels))
    if not checked_clients:
        raise ValueError("客户端合计至少需要一条训练记录")
    clients = checked_clients  # 空客户端不更新模型，也不占样本权重。
    global_weights = np.zeros(width)
    global_bias = 0.0
    for _ in range(rounds):
        local_models = []
        for features, labels in clients:
            weights, bias = global_weights.copy(), global_bias
            for _ in range(local_epochs):
                probabilities = 1 / (1 + np.exp(-np.clip(features @ weights + bias, -30, 30)))
                error = probabilities - labels
                weights -= learning_rate * (features.T @ error / len(labels))
                bias -= learning_rate * float(error.mean())
            local_models.append((len(labels), weights, bias))
        sizes = np.asarray([row[0] for row in local_models], dtype=float)
        expected_weights = np.average(np.stack([row[1] for row in local_models]), axis=0, weights=sizes)
        expected_bias = float(np.average([row[2] for row in local_models], weights=sizes))
        if aggregate_models is None:
            # 独立算法示例默认提供聚合；项目入口始终显式传入第11题回调。
            global_weights, global_bias = expected_weights, expected_bias
        else:
            actual = aggregate_models([(size, weights.copy(), bias) for size, weights, bias in local_models])
            if actual is None:
                raise AggregationPending("11-A模型加权聚合尚未完成，未生成FedAvg参数")
            if not isinstance(actual, (tuple, list)) or len(actual) != 2:
                raise ValueError("11-A须返回(权重向量, 偏置)")
            global_weights, global_bias = np.asarray(actual[0], dtype=float), actual[1]
            if (global_weights.shape != expected_weights.shape
                    or not np.isfinite(global_weights).all()
                    or not np.isscalar(global_bias) or not np.isfinite(global_bias)
                    or not np.allclose(global_weights, expected_weights, rtol=1e-10, atol=1e-12)
                    or not np.isclose(global_bias, expected_bias, rtol=1e-10, atol=1e-12)):
                raise ValueError("11-A须按本轮训练样本量同时聚合权重和偏置")
            global_bias = float(global_bias)
    if not np.isfinite(global_weights).all() or not np.isfinite(global_bias):
        raise ValueError("训练产生非有限参数，请检查输入尺度与学习率")
    return global_weights, float(global_bias)


if __name__ == "__main__":
    print(
        fedavg_train(
            DEMO_CLIENTS,
            rounds=DEMO_ROUNDS,
            local_epochs=DEMO_LOCAL_EPOCHS,
            learning_rate=DEMO_LEARNING_RATE,
        )
    )
