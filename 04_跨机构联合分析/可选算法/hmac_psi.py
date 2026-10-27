"""基于共享HMAC令牌的集合求交演示。

算法定义：双方使用相同密钥和域把集合元素确定性映射为HMAC令牌，然后在令牌空间计算普通
集合交集。它展示了PSI的输入、令牌化、比较和输出流程，但不等价于密码学安全的PSI协议。

核心步骤：
1. 对每个元素计算HMAC(key, domain || value)，得到固定长度令牌。
2. 双方分别把令牌列表转换为集合，自动消除重复元素。
3. 计算两个令牌集合的交集，并排序后返回稳定输出。

输入与输出：输入是双方字符串集合、共享密钥和域；输出是共同元素对应的HMAC令牌，而不是
原始元素。设两边集合大小为n和m，令牌计算与哈希集合求交平均复杂度约为O(n+m)，空间成本
也是O(n+m)；为稳定输出排序交集还需O(k log k)，k为交集大小。

安全性质：没有密钥的观察者难以直接从令牌恢复原值；域分离可以减少不同任务间的令牌复用。
共享密钥方却可以对候选值自行计算令牌，从而执行字典攻击或成员测试，双方还会获知交集
令牌和交集大小。

能力边界：该方法没有盲化、OT、OPRF、恶意安全证明、输入一致性检查和通信认证，因此只能
用于理解算法接口。低熵标识尤其容易被枚举。需要实际安全保证时，应使用经过分析的PSI协议，
并明确输出接收方、集合大小泄露、重复元素语义和恶意输入模型。
"""

import hashlib
import hmac

# 独立演示数据：直接在文件开头构造。
DEMO_LEFT_IDS = ["ID-1001", "ID-1002"]
DEMO_RIGHT_IDS = ["ID-1002", "ID-1003"]
DEMO_KEY = b"demo-key-not-for-production"
DEMO_DOMAIN = "membership-overlap"


def tokenize(value: str, key: bytes, domain: str) -> str:
    """用给定密钥与用途域将单个标识转换为HMAC-SHA256令牌。
    返回完整的小写十六进制摘要；双方只有统一协议才能比较令牌。
    """
    if not isinstance(value, str):
        raise TypeError("标识必须是字符串，避免隐式转换合并不同类型的值")
    if not isinstance(key, bytes) or not key:
        raise ValueError("共享密钥必须是非空bytes")
    if not isinstance(domain, str) or not domain:
        raise ValueError("用途域必须是非空字符串")
    return hmac.new(key, f"{domain}:{value}".encode(), hashlib.sha256).hexdigest()


def hmac_intersection(left: list[str], right: list[str], key: bytes, domain: str) -> list[str]:
    """双方使用同一演示密钥和域生成令牌，再返回共同令牌。"""
    left_tokens = {tokenize(value, key, domain) for value in left}
    right_tokens = {tokenize(value, key, domain) for value in right}
    return sorted(left_tokens & right_tokens)


if __name__ == "__main__":
    print(hmac_intersection(DEMO_LEFT_IDS, DEMO_RIGHT_IDS, DEMO_KEY, DEMO_DOMAIN))
    print("注意：共享密钥方仍可能测试候选ID；真实PSI需要成熟协议。")
