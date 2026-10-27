"""基于正则表达式的文本模式检测。

背景：非结构化文本没有固定列结构，常用的第一步是用模式匹配寻找格式相对稳定的内容，
例如邮箱、IP地址、订单号或带标签的姓名。正则表达式把字符规则写成可重复执行的模式。

核心思想：依次使用一组“标签—正则表达式”扫描同一文本。只要某个表达式至少匹配一次，
就把对应标签加入结果。算法不理解语义，只判断字符串是否符合给定形式。

输入：可转成字符串的值（None视作空文本），以及由非空标签映射到非空正则表达式的字典。
输出：按规则定义顺序排列的命中标签列表；没有命中时返回空列表。

基本步骤：①把输入转换为字符串；②逐条执行正则搜索；③记录命中标签；④返回标签列表。
若文本长度为n、规则数量为p，可粗略理解为执行p次文本扫描，实际开销还取决于表达式结构。

作用：适合做内容初筛、格式校验和待复核线索发现。它不负责给数据分类，也不决定保护策略。

局限：规则只能识别已经定义的格式，容易受到拼写变化、编码差异和上下文歧义影响；复杂或
设计不当的表达式还可能产生性能问题。命中结果应由后续规则或人工复核确认。
"""

import re

# 演示数据完全在本文件中构造，不读取外部文件。
DEMO_TEXT = "customer=Alex order=ORD-2048 email=alex@example.invalid ip=198.51.100.24"
DEMO_PATTERNS = {
    "EMAIL": r"[\w.+-]+@[\w.-]+\.invalid",
    "IPV4": r"\b(?:192\.0\.2|198\.51\.100|203\.0\.113)\.\d{1,3}\b",
    "ORDER_ID": r"\bORD-\d{4}\b",
    "LABELED_NAME": r"\bcustomer[=: ]+[A-Za-z]+\b",
}


def detect_tokens(value: object, patterns: dict[str, str]) -> list[str]:
    """使用调用方提供的模式扫描文本，并返回至少命中一次的标签。"""
    if not isinstance(patterns, dict):
        raise TypeError("patterns 必须为标签到正则表达式的字典")
    compiled = []
    for label, pattern in patterns.items():
        if not isinstance(label, str) or not label.strip() or "|" in label:
            raise ValueError("规则标签必须为非空字符串，且不得含线索分隔符 |")
        if not isinstance(pattern, str) or not pattern.strip():
            raise ValueError(f"{label} 的正则表达式必须为非空字符串")
        try:
            compiled.append((label, re.compile(pattern, flags=re.I)))
        except re.error as exc:
            raise ValueError(f"{label} 的正则表达式无效：{exc}") from exc
    text = "" if value is None else str(value)
    return [label for label, pattern in compiled if pattern.search(text)]


if __name__ == "__main__":
    print(detect_tokens(DEMO_TEXT, DEMO_PATTERNS))
