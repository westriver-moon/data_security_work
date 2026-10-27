"""基于配对样本和固定划分的保护前后效用比较。

背景：数据变换可能提高保护程度，也可能损失分析价值。如果两次评估使用不同样本或不同
拆分，指标变化就无法明确归因于数据变换，因此需要严格控制比较条件。

核心思想：这是一种受控比较程序，不属于隐私保护算法。它让原始数据和处理后数据使用同一
批样本、同一标签、同一训练测试划分和相同模型族，从而尽量把指标差异归因于数据处理。

核心步骤：
1. 按索引求两张表的共同样本，排除只存在于一侧的记录。
2. 项目设备按稳定编号排序后，用共同标签分层拆分并固定种子；不改变来源索引。
3. 数值特征标准化，类别特征独热编码；在两侧分别训练逻辑回归。
4. 在对应的同一测试样本上计算ROC AUC，并返回两侧指标。

输入与输出：输入包括原始表、处理后表、标签名、两侧数值与类别特征列表以及随机种子；
输出是共同样本数、原始AUC和处理后AUC。

关键设计：固定样本和拆分控制比较条件的差异；共同保留样本仍可能受抑制选择影响，
结论只针对该共同样本，应另报全体保留率与抑制损失。预处理管道只在训练集拟合，防止测试信息泄露。
ROC AUC衡量随机正样本排在随机负样本前面的能力；同分按半次正确排序计入。
它对分类阈值不敏感，但不能反映概率校准。
主要计算成本来自两次模型训练，复杂度取决于样本量、特征维度和模型优化过程。

作用与局限：该方法用于比较数据变换前后的预测效用，不提供任何隐私保证。指标下降不必然
意味着变换不可用，指标不变也不证明没有隐私泄露。小数据集与
类别不平衡可能使结果波动，单次拆分和特征替换也有这种可能。完整评估应增加交叉验证、置信区间、
公平性、校准、信息损失和攻击测试，同时单独验证隐私属性。
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 独立演示数据：直接在文件开头构造。
DEMO_RAW_FRAME = pd.DataFrame(
    {
        "study_hours": [2, 8, 3, 9, 4, 10, 3, 11, 5, 12, 4, 10, 6, 13, 5, 14, 7, 15, 6, 16],
        "practice_count": [0, 4, 1, 5, 1, 6, 2, 7, 2, 8, 3, 7, 3, 9, 4, 10, 4, 11, 5, 12],
        "course_type": ["ONLINE", "CLASSROOM"] * 10,
        "passed": [0, 1] * 10,
    }
)
DEMO_PROTECTED_FRAME = pd.DataFrame(
    {
        "study_band": ["LOW", "HIGH"] * 10,
        "practice_band": ["LOW", "HIGH"] * 10,
        "delivery_group": ["REMOTE", "ONSITE"] * 10,
        "passed": [0, 1] * 10,
    }
)
DEMO_LABEL = "passed"
DEMO_RAW_NUMERIC = ["study_hours", "practice_count"]
DEMO_RAW_CATEGORY = ["course_type"]
DEMO_PROTECTED_NUMERIC: list[str] = []
DEMO_PROTECTED_CATEGORY = ["study_band", "practice_band", "delivery_group"]
DEMO_SEED = 510


def _fit_auc(
    train: pd.DataFrame, test: pd.DataFrame, numeric: list[str], category: list[str], label: str
) -> float:
    """在给定训练表上拟合预处理和逻辑回归，再计算测试表的AUC。
    输入数值列、类别列及标签名；返回AUC，不使用测试集拟合预处理。
    """
    preprocessing = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric),
            ("category", OneHotEncoder(handle_unknown="ignore"), category),
        ]
    )
    model = make_pipeline(
        preprocessing, LogisticRegression(max_iter=1000, class_weight="balanced")
    )
    model.fit(train, train[label])
    return float(roc_auc_score(test[label], model.predict_proba(test)[:, 1]))


def compare_same_cases(
    raw_frame: pd.DataFrame,
    protected_frame: pd.DataFrame,
    label: str,
    raw_numeric: list[str],
    raw_category: list[str],
    protected_numeric: list[str],
    protected_category: list[str],
    seed: int,
) -> dict[str, float]:
    """两张表必须保留相同索引；只比较两边共同存在的案例。"""
    if (not raw_frame.index.is_unique or not protected_frame.index.is_unique
            or not raw_frame.columns.is_unique or not protected_frame.columns.is_unique):
        raise ValueError("比较数据的索引和列名必须唯一")
    if label not in raw_frame or label not in protected_frame:
        raise ValueError("两张表都必须含有同名真实标签")
    for frame, numeric, category in [(raw_frame, raw_numeric, raw_category),
                                     (protected_frame, protected_numeric, protected_category)]:
        fields = [*numeric, *category]
        if (not fields or len(fields) != len(set(fields)) or label in fields
                or not set(fields) <= set(frame)):
            raise ValueError("特征须存在且不重复，不能把标签本身作为特征")
    common_index = raw_frame.index.intersection(protected_frame.index, sort=False)
    # 项目数据按设备编号组织评价样本；独立演示没有设备号时沿用双方配对索引。
    if "device_id" in raw_frame:
        common_index = raw_frame.loc[common_index].sort_values("device_id", kind="stable").index
    if not raw_frame.loc[common_index, label].equals(protected_frame.loc[common_index, label]):
        raise ValueError("同一索引的真实标签必须一致")
    counts = raw_frame.loc[common_index, label].value_counts(dropna=False)
    if len(common_index) < 4 or len(counts) != 2 or counts.min() < 2 or counts.index.isna().any():
        raise ValueError("共同评价样本须含两类非空标签且每类至少2条")
    labels = raw_frame.loc[common_index, label]
    train_index, test_index = train_test_split(
        common_index.tolist(), test_size=0.30, random_state=seed, stratify=labels
    )
    raw_auc = _fit_auc(
        raw_frame.loc[train_index], raw_frame.loc[test_index], raw_numeric, raw_category, label
    )
    protected_auc = _fit_auc(
        protected_frame.loc[train_index],
        protected_frame.loc[test_index],
        protected_numeric,
        protected_category,
        label,
    )
    return {
        "same_case_count": len(common_index),
        "raw_auc": raw_auc,
        "protected_auc": protected_auc,
    }


if __name__ == "__main__":
    print(
        compare_same_cases(
            DEMO_RAW_FRAME,
            DEMO_PROTECTED_FRAME,
            DEMO_LABEL,
            DEMO_RAW_NUMERIC,
            DEMO_RAW_CATEGORY,
            DEMO_PROTECTED_NUMERIC,
            DEMO_PROTECTED_CATEGORY,
            seed=DEMO_SEED,
        )
    )
