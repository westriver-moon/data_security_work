"""使用SHA-256对标识符进行确定性散列。

算法背景：SHA-256是密码学散列函数，可把任意长度的字节序列映射为固定长度的256位摘要。
散列是单向变换，不需要加密密钥；相同输入总会产生相同摘要，输入发生细小变化时摘要通常
也会显著变化。它常用于完整性校验、内容寻址和数字签名中的消息摘要。

核心步骤：①把输入按约定编码为字节；②计算SHA-256；③把32字节摘要表示为64个十六进制字符
字符串。输入规范化规则是算法之外的重要约定，例如大小写和空格不同会产生不同摘要。

输入与输出：输入是字符串标识符，输出是确定性的十六进制摘要。摘要只能用于比较散列后的
值是否一致，不能自动恢复原始输入。

作用与局限：散列没有秘密密钥。原始标识符若格式固定、取值空间较小或可预测，观察者可
预先枚举候选值并比较摘要，这称为字典攻击。确定性摘要还会泄露相等关系和出现频率。添加
公开盐可以改变预计算表，却不能阻止针对单个数据集重新枚举；碰撞概率虽很低，也并非为零。
因此，散列结果是否足以保护数据取决于输入熵、攻击者能力和具体使用环境。
"""

import hashlib

# 独立演示数据：相同值重复出现，用于观察确定性散列的相等关系。
DEMO_IDENTIFIERS = ["ACCOUNT-0042", "ACCOUNT-0197", "ACCOUNT-0042"]


def sha256_identifier(identifier: str) -> str:
    """按UTF-8编码输入并返回完整的SHA-256十六进制摘要。"""
    encoded_identifier = identifier.encode("utf-8")
    return hashlib.sha256(encoded_identifier).hexdigest()


if __name__ == "__main__":
    for demo_identifier in DEMO_IDENTIFIERS:
        print(demo_identifier, "->", sha256_identifier(demo_identifier))
