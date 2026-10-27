"""基于有序属性规则的用途访问决策。

背景：传统角色访问控制主要判断主体是否拥有某项权限；用途控制进一步考虑“为什么访问”。
在属性访问控制思想中，角色、用途、动作、资源状态、时间和审批状态都可以成为判断条件。

核心思想：把每条访问规则表示为“规则名称、条件函数、结果三元组”，再按优先级逐条检查。
第一条条件为真的规则返回“决定、控制方式、原因”；全部未命中时使用默认结果。

输入：包含任意属性的申请字典、有序访问规则列表、默认结果。
输出：三元组“访问决定、控制方式、解释原因”。算法本身不预设角色名称、用途名称或资源字段。

基本步骤：①收集决策所需属性；②按顺序执行条件函数；③命中后立即返回结果；④无命中时
返回默认值。设规则数量为n，最坏情况下执行n次条件判断；具体开销取决于每个条件函数。

作用：展示如何把用途限制和其他属性组织为明确、可测试的策略决策点。拒绝规则通常排在
允许规则之前，但具体优先级必须由调用方显式定义。

局限：算法只产生决策，不负责身份认证、数据库强制执行、凭证签发、到期撤权或审计。条件
函数可能冲突或遗漏，默认允许还可能带来较大风险，因此规则需要测试、审批、版本管理和监控。
"""

from collections.abc import Callable, Mapping

AccessResult = tuple[str, str, str]
AccessRule = tuple[str, Callable[[Mapping[str, object]], bool], AccessResult]

# 演示申请和规则完全在本文件中构造，不包含特定组织的权限答案。
DEMO_REQUEST = {
    "role": "report_viewer",
    "purpose": "monthly_reporting",
    "operation": "VIEW",
    "approval_active": True,
}
DEMO_RULES: list[AccessRule] = [
    (
        "inactive_approval",
        lambda request: not bool(request["approval_active"]),
        ("DENY", "NONE", "审批未生效"),
    ),
    (
        "approved_report_view",
        lambda request: (
            request["role"] == "report_viewer"
            and request["purpose"] == "monthly_reporting"
            and request["operation"] == "VIEW"
        ),
        ("ALLOW", "READ_ONLY", "角色、用途和操作均匹配"),
    ),
]
DEMO_DEFAULT_RESULT: AccessResult = ("DENY", "NONE", "没有匹配的允许规则")


def decide_access(
    request: Mapping[str, object],
    rules: list[AccessRule],
    default_result: AccessResult,
) -> AccessResult:
    """按顺序执行访问规则，并返回第一条命中规则或默认结果。"""
    results = [default_result, *(result for _name, _condition, result in rules)]
    if any(not isinstance(result, tuple) or len(result) != 3
           or any(not isinstance(value, str) or not value.strip() for value in result)
           for result in results):
        raise ValueError("每个决策结果必须是三个非空字符串组成的元组")
    for _rule_name, condition, result in rules:
        matched = condition(request)
        if not isinstance(matched, bool):
            raise TypeError("规则条件必须返回布尔值，不能把字符串False当作命中")
        if matched:
            return result
    return default_result


if __name__ == "__main__":
    print(decide_access(DEMO_REQUEST, DEMO_RULES, DEMO_DEFAULT_RESULT))
