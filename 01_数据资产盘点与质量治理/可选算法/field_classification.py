"""按优先级执行的字段名称分类器。

背景：字段分类常从数据字典开始。字段名能够提供初步语义，但同义词、缩写和命名习惯会让
单一关键字规则不够可靠，因此通常需要组合“精确匹配、关键字匹配、默认分类”。

核心思想：优先查询精确规则；没有命中时，再按给定顺序检查关键字组；仍未命中则返回默认
分类。优先级是算法的一部分，因为同一字段可能同时包含多个关键字。

输入：字段名、精确规则字典、有序关键字规则列表和默认分类。
输出：一个二元组，包含类别名称与等级名称。算法不限定必须使用L1—L4，也不内置业务字段。

基本步骤：①统一字段名和规则键大小写；②执行精确匹配；③依次测试关键字；④使用默认值兜底。
规则键规范化后若冲突则拒绝；空关键词或不完整的分类配置同样拒绝，避免误匹配全部字段。
精确字典查询平均为常数时间，关键字阶段与规则数量及字段名长度相关。

作用：展示怎样把人工制定的分类表转换为稳定、可测试的程序规则。

局限：字段名不是数据内容，名称相同的字段在不同语境中可能含义不同。规则必须由调用方
提供并负责审批；高风险或未知字段仍需要内容检查和人工复核。
"""

# 演示数据和规则完全在本文件中构造，不包含特定业务字段。
DEMO_FIELDS = ["full_name", "monthly_revenue", "delivery_region", "room_temperature", "notes"]
DEMO_EXACT_RULES = {
    "full_name": ("直接标识符", "L4"),
    "monthly_revenue": ("敏感统计", "L3"),
    "delivery_region": ("准标识信息", "L2"),
}
DEMO_KEYWORD_RULES = [
    (("temperature", "pressure"), ("测量数据", "L2")),
    (("comment", "notes"), ("自由文本", "L2")),
]
DEMO_DEFAULT = ("一般元数据", "L1")


def classify_field(
    field_name: str,
    exact_rules: dict[str, tuple[str, str]],
    keyword_rules: list[tuple[tuple[str, ...], tuple[str, str]]],
    default: tuple[str, str],
) -> tuple[str, str]:
    """按照精确规则、关键字规则、默认值的优先级返回分类结果。"""
    if not isinstance(field_name, str) or not field_name.strip():
        raise ValueError("field_name 必须为非空字符串")
    if not isinstance(exact_rules, dict) or not isinstance(keyword_rules, (list, tuple)):
        raise TypeError("精确规则须为字典，关键字规则须为有序列表或元组")
    validate_classification(default)
    normalized = {}
    for key, result in exact_rules.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("精确规则键必须为非空字符串")
        validate_classification(result)
        key = key.casefold()
        if key in normalized and normalized[key] != result:
            raise ValueError("精确规则键忽略大小写后存在冲突")
        normalized[key] = result
    for rule in keyword_rules:
        if not isinstance(rule, (list, tuple)) or len(rule) != 2:
            raise ValueError("关键字规则须为 (关键词组, 分类二元组)")
        keywords, result = rule
        if not isinstance(keywords, (list, tuple)) or not keywords or any(
            not isinstance(keyword, str) or not keyword.strip() for keyword in keywords
        ):
            raise ValueError("关键词组必须包含非空字符串，不能传入单个字符串")
        validate_classification(result)
    name = field_name.casefold()
    if name in normalized:
        return normalized[name]
    for keywords, result in keyword_rules:
        if any(keyword.casefold() in name for keyword in keywords):
            return result
    return default


def validate_classification(result):
    """检查类别与等级组成的配置二元组，避免把不完整的规则传给后续标签流程。"""
    if not isinstance(result, tuple) or len(result) != 2 or any(
        not isinstance(value, str) or not value.strip() for value in result
    ):
        raise ValueError("分类必须为两个非空字符串组成的元组")



if __name__ == "__main__":
    for demo_field in DEMO_FIELDS:
        print(
            demo_field,
            "->",
            classify_field(demo_field, DEMO_EXACT_RULES, DEMO_KEYWORD_RULES, DEMO_DEFAULT),
        )
