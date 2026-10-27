"""通过排序归并计算共享HMAC令牌的集合交集。

背景：同一集合交集可以用哈希集合，也可以先排序，再用双指针逐项比较。
本示例沿用hmac_psi.py的HMAC-SHA256协议，只改变交集计算过程。

输入：双方字符串标识列表、同一字节密钥和用途域。
输出：去重、按字典序排列的共同令牌列表；没有交集时返回[]。
步骤：分别生成并排序令牌，比较两个当前位置，相等时输出并同时前进，
不相等时只前进较小值一侧；重复相等值不重复输出。求交不调用集合交集。
令牌化为O(n+m)，排序为O(n log n+m log m)，归并为O(n+m)。
当前返回稳定顺序的哈希方案也需要对最终交集排序；两者不存在一概更快的结论。

边界：排序不会增强HMAC协议本身。共享密钥方仍能测试候选标识，
本地进程能看到双方输入；这不是具有恶意安全保证的完整PSI协议。
单独执行本文件可比较两种实现；演示材料与课程业务数据无关。
"""

if __package__:
    from .hmac_psi import hmac_intersection, tokenize
else:
    from hmac_psi import hmac_intersection, tokenize

DEMO_LEFT_IDS = ["item-C", "item-A", "item-A", "item-D"]
DEMO_RIGHT_IDS = ["item-B", "item-D", "item-A", "item-D"]
DEMO_KEY = b"sorting-demo-key"
DEMO_DOMAIN = "inventory-overlap"


def sorted_token_intersection(
    left: list[str], right: list[str], key: bytes, domain: str
) -> list[str]:
    """统一生成HMAC令牌，再通过排序和双指针归并返回唯一的交集令牌。"""
    left_tokens = sorted(tokenize(value, key, domain) for value in left)
    right_tokens = sorted(tokenize(value, key, domain) for value in right)
    common = []
    left_pos = right_pos = 0
    while left_pos < len(left_tokens) and right_pos < len(right_tokens):
        left_token, right_token = left_tokens[left_pos], right_tokens[right_pos]
        if left_token == right_token:
            if not common or common[-1] != left_token:
                common.append(left_token)
            left_pos += 1
            right_pos += 1
        elif left_token < right_token:
            left_pos += 1
        else:
            right_pos += 1
    return common


if __name__ == "__main__":
    result = sorted_token_intersection(
        DEMO_LEFT_IDS, DEMO_RIGHT_IDS, DEMO_KEY, DEMO_DOMAIN
    )
    print("排序归并交集：", result)
    print("与哈希集合结果一致：", result == hmac_intersection(
        DEMO_LEFT_IDS, DEMO_RIGHT_IDS, DEMO_KEY, DEMO_DOMAIN
    ))
