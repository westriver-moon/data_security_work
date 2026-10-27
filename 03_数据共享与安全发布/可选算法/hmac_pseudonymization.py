"""基于HMAC和域分离的确定性假名化。

算法定义：HMAC是一种带密钥的消息认证码。它把秘密密钥K和消息m组合后输入密码哈希函数，
可概括为H((K⊕opad) || H((K⊕ipad) || m))。假名化把原标识作为消息，并截取HMAC输出作为
替代标识。

核心步骤：
1. 把场景域和原始值连接为“domain:value”，形成有上下文的消息。
2. 使用秘密密钥和SHA-256计算HMAC摘要。
3. 截取指定长度的十六进制字符串作为假名令牌。

输入与输出：输入是任意原始标识、字节密钥、场景域和可选输出长度；输出是确定性字符串。
相同密钥、域和值始终产生相同结果，因此可以做等值连接；改变场景域会得到不同令牌，这种
做法称为域分离，可减少跨用途关联。

关键参数：密钥必须随机、足够长并妥善轮换；输出截断越短，碰撞概率越高。计算成本主要是
一次SHA-256 HMAC，时间复杂度与输入长度线性相关。

安全边界：HMAC比无密钥哈希更能抵抗离线字典匹配，但持有密钥的一方仍能枚举候选值。
确定性令牌会泄露相等关系和频率分布，截断还会引入碰撞。假名化属于可关联的受保护数据，
不等于匿名化；密钥泄露后应视为全部令牌的保护失效。
"""

import hashlib
import hmac

# 独立演示数据：直接在文件开头构造。
DEMO_VALUE = "USER-0042"
DEMO_KEY = b"demo-key-not-for-production"
DEMO_DOMAINS = ["analytics", "customer-support"]


def pseudonymize(value: object, key: bytes, domain: str, length: int = 16) -> str:
    """使用密钥和场景域生成假名；假名仍是受保护数据，不等于匿名。"""
    from numbers import Integral

    if not isinstance(key, (bytes, bytearray)) or not key:
        raise ValueError("key须为非空字节密钥")
    if not isinstance(domain, str) or not domain.strip():
        raise ValueError("domain须为非空用途域字符串")
    if isinstance(length, bool) or not isinstance(length, Integral) or not 1 <= length <= 64:
        raise ValueError("length须为1到64之间的整数")
    message = f"{domain}:{value}".encode("utf-8")
    return hmac.new(key, message, hashlib.sha256).hexdigest()[:length]


if __name__ == "__main__":
    for demo_domain in DEMO_DOMAINS:
        print(pseudonymize(DEMO_VALUE, DEMO_KEY, demo_domain))
