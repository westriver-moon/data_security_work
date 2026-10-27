"""可选方法：根据用途要求组合措施，并形成可核查的决策理由。

背景：一家画廊需要将未公开作品目录临时交给修复人员。修复人员需要看原始
图片细节，随机扰动图片会影响其判断；要求主要是限制谁能查看、传输时保护
内容、约定到期后停止访问。这些要求分别对应接收者限制、加密传输和访问期限。

输入：是否需要原始细节、是否传给外部人员、约定访问天数。
输出：措施代号列表及理由。列表只是待执行的措施计划，不代表技术已接入。
假设：用途和可接收对象已经明确；演示不代替实际访问系统、密码协议或风险判断。
边界：加密传输只保护传输过程，不阻止获准接收者读到原文，也不自动防止另存副本。
到期设置须由访问系统执行；仅写出代号不能证明结果受到了保护。

运行：直接执行本文件查看画廊演示；不读取课程数据，不列课程六类用途的答案。
迁移时分别说明需要保留的业务能力、计算过程的要求及最终结果的暴露范围，
再选择实际可用方法，并到后续产物中寻找措施已经执行的证据。
"""

DEMO_REQUIREMENTS = {
    "need_original_detail": True,
    "external_recipient": True,
    "access_days": 7,
}


def choose_access_measures(need_original_detail, external_recipient, access_days):
    """将原始细节需求、接收范围与期限整理为访问措施及具体理由。"""
    if not isinstance(access_days, int) or isinstance(access_days, bool) or access_days < 1:
        raise ValueError("访问期限必须是正整数天数")
    if not isinstance(need_original_detail, bool) or not isinstance(external_recipient, bool):
        raise TypeError("用途与接收范围开关必须为布尔值，不能用字符串False代替False")
    measures = ["NAMED_RECIPIENTS", "EXPIRING_ACCESS"]
    reasons = [f"限定接收人员并设置{access_days}天访问期限"]
    if external_recipient:
        measures.append("ENCRYPTED_TRANSFER")
        reasons.append("跨组织传输时保护传输内容")
    if need_original_detail:
        reasons.append("修复判断依赖原始细节，不改变作品图片内容")
    else:
        measures.append("PREVIEW_ONLY")
        reasons.append("用途只需预览，限制所提供的细节")
    return measures, "；".join(reasons)


def main():
    """打印独立画廊用途的措施计划，并解释计划与实际执行的区别。"""
    measures, reason = choose_access_measures(**DEMO_REQUIREMENTS)
    print("画廊临时协作的措施计划：", measures)
    print("选择依据：", reason)
    print("仍须在访问系统中落实限制、期限与传输保护，不能将此计划当作执行证据。")


if __name__ == "__main__":
    main()
