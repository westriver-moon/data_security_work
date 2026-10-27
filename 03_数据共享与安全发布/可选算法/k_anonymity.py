"""通过等价类抑制实现的k匿名。

算法定义：准标识符是单独看不直接指向个体、但组合后可能重新识别个体的字段。把准标识符
取值完全相同的记录组成等价类；若每个等价类至少包含k条记录，则数据满足k匿名。

核心步骤：
1. 选择准标识符字段，并按这些字段对全部记录分组。
2. 将每条记录映射到所属等价类的大小。
3. 保留组大小大于或等于k的记录，抑制小于k的记录。
4. 返回处理后数据以及输入行数、发布行数、抑制行数和最小等价类大小。

输入与输出：输入是数据表、准标识符字段列表和正整数k；输出是保留原索引的处理后表与
统计报告。k越大，个体隐藏在更大群体中，但通常会导致更多泛化或抑制和更高信息损失。

关键设计：本实现只采用整组抑制，没有搜索最优泛化层级。基于哈希分组时，时间复杂度通常
接近O(n×q)，其中n是记录数、q是准标识符数量；实际开销还与字段类型和基数有关。

安全边界：k匿名主要降低基于准标识符的唯一定位风险，不能抵御敏感属性同质性攻击、背景
知识攻击和发布间链接攻击。它也没有像差分隐私那样提供查询级数学保证。需要更强属性保护时
可进一步考虑l多样性、t接近性，并结合重识别测试和效用评估。
"""

import pandas as pd

# 独立演示数据：直接在文件开头构造。
DEMO_FRAME = pd.DataFrame(
    {
        "age_band": ["20-29", "20-29", "30-39", "30-39", "40-49"],
        "district": ["A", "A", "B", "B", "C"],
        "survey_score": [4, 5, 3, 4, 2],
    }
)
DEMO_QUASI_FIELDS = ["age_band", "district"]
DEMO_K = 2


def suppress_small_groups(
    frame: pd.DataFrame, quasi_fields: list[str], k: int
) -> tuple[pd.DataFrame, dict[str, int]]:
    """保留原索引，便于在同一案例集合上比较处理前后效用。"""
    from numbers import Integral

    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique or not frame.index.is_unique:
        raise ValueError("frame须为列名和索引均唯一的DataFrame")
    if not isinstance(quasi_fields, list) or not quasi_fields or any(
        not isinstance(field, str) for field in quasi_fields
    ) or len(set(quasi_fields)) != len(quasi_fields) or not set(quasi_fields) <= set(frame):
        raise ValueError("准标识字段须为非空、无重复且存在于数据表的字段列表")
    if isinstance(k, bool) or not isinstance(k, Integral) or k < 1:
        raise ValueError("k须为正整数")
    group_size = frame.groupby(quasi_fields, dropna=False, observed=True)[quasi_fields[0]].transform("size")
    kept = frame.loc[group_size >= k].copy()
    report = {
        "k": k,
        "input_rows": len(frame),
        "released_rows": len(kept),
        "suppressed_rows": int((group_size < k).sum()),
        "minimum_group_size": (
            int(kept.groupby(quasi_fields, dropna=False, observed=True).size().min()) if len(kept) else 0
        ),
    }
    return kept, report


if __name__ == "__main__":
    safe, summary = suppress_small_groups(DEMO_FRAME.copy(), DEMO_QUASI_FIELDS, k=DEMO_K)
    print(safe)
    print(summary)
