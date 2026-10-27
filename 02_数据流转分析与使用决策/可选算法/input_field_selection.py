"""可选方法：按照数据在流程中的角色整理必要输入。

背景：图书馆希望用借阅发生时已知的信息预测逾期，借阅编号用于划分样本，
预约天数和过去借阅次数是预测时已知的特征，实际是否逾期是历史训练标签。
读者的备注虽在表中存在，却不是这个演示用途所需的输入。标签参与监督训练，
不应当进入预测特征；用于拆分的编号也不因此成为训练特征。

输入：按角色列出的字段、可用列名和已完成的字段分级。
输出：按首次出现顺序去重的必要输入字段。缺少字段或有效分级会报错。
边界：只检查存在性和分级，不会自动判定用途必要性、授权状态或保护方法。
必要性须先根据具体计算解释；等级较高不意味着一定采用某一种技术。

运行：直接执行本文件，查看各角色及合并后的字段列表。
示例数据在下方完整定义，不读取课程数据，也不提供课程各用途的字段答案。
迁移时先列出实际操作所需角色，再核对真实字段，最后检查后续代码是否只取这些列。
"""

DEMO_ROLES = {
    "样本划分依据": ["loan_key"],
    "借阅时已知特征": ["planned_days", "past_loan_count"],
    "历史训练标签": ["returned_late"],
}
DEMO_COLUMNS = [
    "loan_key", "planned_days", "past_loan_count", "returned_late", "reader_note"
]
DEMO_LEVELS = {
    "loan_key": "L4", "planned_days": "L2", "past_loan_count": "L3",
    "returned_late": "L3", "reader_note": "L4",
}


def select_input_fields(roles, available_columns, level_by_field):
    """按角色顺序合并字段并检查存在性、合法分级，返回必要输入列表。"""
    if any(isinstance(fields, str) for fields in roles.values()):
        raise TypeError("每个角色应提供字段列表，不能将单个字段字符串当作列表")
    selected = list(dict.fromkeys(field for fields in roles.values() for field in fields))
    if any(not isinstance(field, str) or not field.strip() for field in selected):
        raise ValueError("字段名必须是非空字符串")
    if not selected:
        raise ValueError("至少需要一个输入字段")
    for field in selected:
        if field not in available_columns:
            raise ValueError(f"字段不存在：{field}")
        if level_by_field.get(field) not in {"L1", "L2", "L3", "L4"}:
            raise ValueError(f"字段尚未合法分级：{field}")
    return selected


def main():
    """使用独立图书借阅材料演示字段按角色合并，输出必要性解释。"""
    for role, fields in DEMO_ROLES.items():
        print(f"{role}：{fields}")
    print("图书借阅演示的必要输入：")
    print(select_input_fields(DEMO_ROLES, DEMO_COLUMNS, DEMO_LEVELS))
    print("reader_note未参与本用途；合并输入仍须在训练时区分编号、特征和标签。")


if __name__ == "__main__":
    main()
