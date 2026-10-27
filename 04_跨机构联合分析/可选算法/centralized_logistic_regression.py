"""集中式二分类逻辑回归。

算法背景：逻辑回归使用线性组合z=w·x+b描述特征，再通过Sigmoid函数把z转换为0到1之间的
概率。训练通常最小化二元交叉熵，并可加入正则化限制参数规模。它具有计算成本低、输出概率
和参数较容易解释等特点。

集中式训练方式：训练者先取得所有数据批次，将它们合并成同一特征矩阵和标签向量，再拟合
一个模型。示例先用集中数据拟合标准化参数，然后训练逻辑回归；预测阶段复用同一条管道。

输入与输出：输入是若干结构一致的特征矩阵和对应二元标签，以及用于复现实验的随机种子；
输出是包含标准化器和逻辑回归的模型。各批次必须使用一致的特征定义和标签含义。

作用与局限：集中训练能直接利用全部样本，常被用作性能基线，但训练者可以访问合并后的
逐条原始记录。它本身不提供访问控制、加密、匿名化或隐私保证；训练数据中的偏差仍会进入
模型，模型评估还需要独立测试集。逻辑回归只能表示特征经过线性组合后的决策边界。
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 独立演示数据：两个批次具有相同的两列特征和相同的二元标签定义。
DEMO_FEATURE_BATCHES = [
    np.array([[0.2, 1.1], [0.5, 0.9], [1.0, 1.8], [1.2, 2.0], [1.5, 2.4], [1.8, 2.8]]),
    np.array([[0.1, 0.7], [0.7, 1.3], [0.9, 1.5], [1.4, 2.1], [1.7, 2.5], [2.0, 3.0]]),
]
DEMO_LABEL_BATCHES = [
    np.array([0, 0, 0, 1, 1, 1]),
    np.array([0, 0, 1, 1, 1, 1]),
]
DEMO_RANDOM_SEED = 510


def train_centralized_logistic_regression(
    feature_batches: list[np.ndarray],
    label_batches: list[np.ndarray],
    seed: int,
):
    """合并全部批次，并在同一位置拟合标准化器和逻辑回归。"""
    if len(feature_batches) != len(label_batches) or not feature_batches:
        raise ValueError("特征批次和标签批次必须非空且一一对应")
    checked_features, checked_labels = [], []
    width = None
    for features, labels in zip(feature_batches, label_batches):
        features, labels = np.asarray(features, dtype=float), np.asarray(labels, dtype=float)
        if features.ndim != 2 or features.shape[1] == 0 or labels.ndim != 1:
            raise ValueError("每批特征须为非零列二维矩阵，标签须为一维向量")
        if len(features) != len(labels):
            raise ValueError("每批特征和标签必须逐行对应，不能只检查合并后的总行数")
        if width is not None and features.shape[1] != width:
            raise ValueError("各批特征列数必须一致")
        width = features.shape[1]
        if not np.isfinite(features).all() or not np.isin(labels, [0, 1]).all():
            raise ValueError("特征必须有限，标签只能为0或1")
        checked_features.append(features)
        checked_labels.append(labels)
    features = np.vstack(checked_features)
    labels = np.concatenate(checked_labels)
    if set(labels) != {0.0, 1.0}:
        raise ValueError("二分类训练合计必须包含0、1两类标签")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=seed),
    )
    model.fit(features, labels)
    return model


if __name__ == "__main__":
    demo_model = train_centralized_logistic_regression(
        DEMO_FEATURE_BATCHES, DEMO_LABEL_BATCHES, DEMO_RANDOM_SEED
    )
    demo_all_features = np.vstack(DEMO_FEATURE_BATCHES)
    print("class probabilities:\n", demo_model.predict_proba(demo_all_features))
    print(
        "training accuracy:",
        demo_model.score(demo_all_features, np.concatenate(DEMO_LABEL_BATCHES)),
    )
