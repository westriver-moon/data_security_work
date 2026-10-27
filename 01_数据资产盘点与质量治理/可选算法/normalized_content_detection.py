"""在规范化副本上执行规则识别，不修改原文。

用途：处理全角字符、兼容字符和连续空白导致的格式差异，供已有模式规则做内容初筛。
输入：文本、标签到正则表达式的字典，以及是否使用 Unicode NFKC 与折叠空白的选项。
输出：按规则顺序排列的命中标签列表，与 detect_tokens 的输出相同。
步骤：复制成字符串，按选项做 Unicode 兼容规范化，将每段空白折为一个空格，再扫描。
局限：NFKC可能合并原本有意义的字符差异；不会删除全部空白、连字符或标点，不把隔开的
数字直接拼接。规则仍需适配空格和上下文，未命中不能证明安全。证据必须定位回原始文本。
示例只使用独立的档案编号和保留域名，算法不预置项目识别规则。
"""

import re
import unicodedata

if __package__:
    from .content_detection import detect_tokens
else:
    from content_detection import detect_tokens

DEMO_TEXT = "档案：ＡＲＣ－２０４８\t 邮箱：ｒｅａｄｅｒ＠ｅｘａｍｐｌｅ．ｉｎｖａｌｉｄ"
DEMO_PATTERNS = {"ARCHIVE": r"\bARC-\d{4}\b", "EMAIL": r"[\w.+-]+@[\w.-]+\.invalid"}


def normalize_scan_copy(value, use_nfkc=True, collapse_whitespace=True):
    """按显式选项生成供扫描的文本副本，保留不同片段之间的分隔而不拼接数字。"""
    if not isinstance(use_nfkc, bool) or not isinstance(collapse_whitespace, bool):
        raise TypeError("规范化选项必须为 bool")
    text = "" if value is None else str(value)
    if use_nfkc:
        text = unicodedata.normalize("NFKC", text)
    if collapse_whitespace:
        text = re.sub(r"\s+", " ", text)
    return text


def detect_normalized_tokens(value, patterns, use_nfkc=True, collapse_whitespace=True):
    """规范化副本后执行调用方的识别规则，返回命中标签并保留原文供外部证据定位。"""
    return detect_tokens(normalize_scan_copy(value, use_nfkc, collapse_whitespace), patterns)


if __name__ == "__main__":
    print("原文：", DEMO_TEXT)
    print("命中：", detect_normalized_tokens(DEMO_TEXT, DEMO_PATTERNS))
