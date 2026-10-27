"""把分类结果转换为机器可读标签。

背景：自然语言类别便于沟通，但程序通常需要有限、稳定的代码值。标签映射把治理词汇转换成
下游系统容易比较、过滤和记录的字符串，同时可以根据等级产生一个初始控制状态。

核心思想：用字典完成“类别到标签”的查找，再判断等级是否属于调用方指定的限制集合。
未知类别使用兜底标签，防止未识别数据被静默当成普通数据。

输入：类别、等级、类别标签映射、需要限制的等级集合，以及两个状态名称。
输出：机器标签与初始控制状态组成的二元组。

基本步骤：①查询类别标签；②未知类别采用复核标签；③测试等级集合成员关系；④返回结果。
字典查询和集合成员判断平均都可以看作常数时间。

作用：展示治理元数据如何从描述性文本转化成程序可执行的标准代码。

局限：标签和初始状态没有自动推导业务合法性，也不等于最终授权。映射内容、等级边界和状态
含义都应由调用方明确配置，并处理版本升级和未知值。
"""

# 演示数据和映射完全在本文件中构造。
DEMO_CATEGORY = "敏感统计"
DEMO_LEVEL = "L3"
DEMO_TAG_BY_CATEGORY = {
    "直接标识符": "IDENTIFIER|DIRECT",
    "敏感统计": "SENSITIVE|STATISTICAL",
    "准标识信息": "IDENTIFIER|QUASI",
    "一般元数据": "GENERAL",
}
DEMO_RESTRICTED_LEVELS = {"L3", "L4"}


def make_tags(
    category: str,
    level: str,
    tag_by_category: dict[str, str],
    restricted_levels: set[str],
    restricted_status: str = "RESTRICTED",
    ordinary_status: str = "CONDITIONAL",
) -> tuple[str, str]:
    """根据调用方提供的映射和等级集合生成标签与初始状态。"""
    for label, value in (("category", category), ("level", level),
                         ("restricted_status", restricted_status), ("ordinary_status", ordinary_status)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} 必须为非空字符串")
    if not isinstance(tag_by_category, dict) or any(
        not isinstance(key, str) or not key.strip() or not isinstance(value, str) or not value.strip()
        for key, value in tag_by_category.items()
    ):
        raise ValueError("tag_by_category 必须为非空字符串到非空字符串的映射")
    if not isinstance(restricted_levels, (set, frozenset)) or any(
        not isinstance(value, str) or not value.strip() for value in restricted_levels
    ):
        raise ValueError("restricted_levels 必须为非空等级字符串的集合")
    tags = tag_by_category.get(category, "REVIEW_REQUIRED")
    status = restricted_status if level in restricted_levels else ordinary_status
    return tags, status


if __name__ == "__main__":
    print(make_tags(DEMO_CATEGORY, DEMO_LEVEL, DEMO_TAG_BY_CATEGORY, DEMO_RESTRICTED_LEVELS))
