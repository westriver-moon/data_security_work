"""可选方法：按照保留、转换、移除三个动作设计明细输出列。

背景：物流运营人员只需要按发货日观察包裹重量，不需要查看精确发货时间、
收件地址或运单编号。因此计划保留重量、将时间转换为日期，并移除另两列。
每个动作需要具体理由；同一个输入不能反复声明互相冲突的处理方式。

输入：由输入列名、动作、输出列名、业务理由组成的处理计划。
输出：按计划顺序排列的输出列名列表。
假设：这里只处理逐列计划；合计、均值或模型参数等新结果需要额外定义结果合同。
边界：函数不会执行时间转换或处理真实值，也不证明最终表已经匿名化。
即使删除编号，其他列的组合仍可能定位个体，必须结合实际方法和产物检查。

运行：直接执行本文件查看独立物流计划；不读取课程数据，不填课程题目的字段配置。
迁移时说明每个必要输入如何用于计算，以及哪些原始值可以或不可以进入最终结果。
"""

# 每项依次为：输入字段、动作、输出字段、业务理由。
DEMO_PLAN = [
    ("waybill_no", "REMOVE", None, "日级重量分析不需要关联具体运单"),
    ("dispatch_at", "TRANSFORM", "dispatch_day", "分析按日开展，不需要精确发货时刻"),
    ("parcel_weight_g", "KEEP", "parcel_weight_g", "保留重量用于计算分布"),
    ("recipient_address", "REMOVE", None, "收件地址与本次日级重量用途无关"),
]


def design_output_fields(plan):
    """校验处理计划并提取输出列；只设计结构，不处理实际记录的值。"""
    outputs = []
    seen_inputs = set()
    for source, action, target, reason in plan:
        if (not isinstance(source, str) or not source.strip()
                or not isinstance(reason, str) or not reason.strip()):
            raise ValueError("输入字段及业务理由必须是非空字符串")
        if source in seen_inputs:
            raise ValueError("输入字段不能重复，且每项需要理由")
        seen_inputs.add(source)
        if action == "REMOVE":
            if target is not None:
                raise ValueError("移除字段不能指定输出列")
            continue
        if action not in {"KEEP", "TRANSFORM"} or not isinstance(target, str) or not target.strip():
            raise ValueError("必须指定有效动作和输出列")
        if action == "KEEP" and target != source:
            raise ValueError("保留动作使用原列名")
        if target in outputs:
            raise ValueError(f"输出列重复：{target}")
        outputs.append(target)
    return outputs


def main():
    """打印独立物流计划与输出结构，并提示值转换仍须实际执行。"""
    for source, action, target, reason in DEMO_PLAN:
        print(f"{source} → {target or '不输出'}（{action}）：{reason}")
    print("物流日级重量分析的输出列：")
    print(design_output_fields(DEMO_PLAN))
    print("程序还须真正将时间转换成日期，并检查记录内容与用途是否一致。")


if __name__ == "__main__":
    main()
