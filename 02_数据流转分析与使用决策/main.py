"""任务2：从实际字段与用途形成保护规则，供处理和检查共同使用。"""
# 【待填】说明题目，成对“作答区”边框标出填写范围；先阅读框外提示，再在框内完成作答。

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# 本文件第04、05题：搜索“待填”定位真正空位；内部服务分析、服务统计、共同设备匹配已提供完整示例。
# 字段与方法按用途成对配置，模型输出已给定；其他 None 及读写、登记、比较框架已提供，无需替换。
import json
import pandas as pd
from utils import prepare_task_run, read_clean, task_dir, read_table, write_csv, USES

# 本任务按字段名填写列表，无需为字段选择额外导入函数。


def main():
    """读取三张清洗表和字段分级，设计用途规则，再检查、保存。

    规则配置完整不代表数值已治理或保护已执行；后续可产生基线或等待记录。
    """
    # 直接运行本文件时，先由根入口准备同一次运行及必要前序任务。
    prepare_task_run(2)
    cn_devices = read_clean("cn_devices.csv")
    de_devices = read_clean("de_devices.csv")
    service_cases = read_clean("service_cases.csv")
    catalog = read_table(task_dir(1) / "字段分级.csv")
    data = {
        "cn_devices.csv": cn_devices,
        "de_devices.csv": de_devices,
        "service_cases.csv": service_cases,
    }

    # 04题决定输入和输出；05题决定保护方法及业务理由。
    field_choices = choose_fields()
    method_choices = choose_protection_methods()
    rules = build_rules(data, catalog, field_choices, method_choices)
    write_csv(2, "流转与保护规则.csv", rules)
    print(rules[["result_name", "protection", "ready", "reason"]].to_string(index=False))


def choose_fields():
    """填写六种用途的必要输入与允许输出，只返回配置，不读写数据。"""
    # 以下 statistics 列出统计结果允许包含的字段，供统计用途引用；给出要求不等于数据可向公众公开。
    statistics = [
        "query",
        "released_value",
        "mechanism",
        "epsilon",
        "delta",
        "privacy_unit",
        "adjacency",
        "public_n",
        "contribution_limit",
        "records_per_unit",
    ]
    # TODO04【核心填空与接入】：参照三种已提供用途，补齐另外三种用途的输入与输出字段。
    # 处理对象：任务1清洗数据中的cn_devices.csv、de_devices.csv、service_cases.csv，
    # 并参考任务1字段分级.csv；本题生成用途配置，不修改这些输入表。
    # 每项格式为(必要输入字段列表, 允许输出字段列表)；输入列来自main()读取的三张清洗表。
    # 六类结果允许包含的字段及含义见任务背景；统计输出可引用 statistics，模型输入按字段名列出。
    # 不能直接保留全部原表；完成后检查流转与保护规则.csv及任务3、4的实际结果列。
    field_choices = {
        "内部服务分析": (
            ["device_id", "downtime_hours", "confirmed_failure"],
            ["device_token", "downtime_hours", "confirmed_failure"],
        ),
        # 【待填 04-A】研究数据：(必要输入列表, 允许输出列表)，两个 [] 都要填写。
        # 来源表：任务1清洗后的德维de_devices.csv；配置供任务3生成研究数据.csv。
        # 提示：先按分组、泛化和效果评价用途选择输入；输出保留分组后的字段和标签，排除原设备编号。
        # 输出字段合同：device_model（型号）、region_group（泛化区域）、hours_band（时长区间）、failure_next_90d（标签）。
        # ========== 作答区 04-A 开始 ==========
        "研究数据": (
            ["device_id", "device_model", "site_region", "operating_hours", "failure_next_90d"],
            ["device_model", "region_group", "hours_band", "failure_next_90d"],
        ),
        # ========== 作答区 04-A 结束 ==========

        # 已提供示例：案例编号核对统计单元，故障和停机参与计算，结果只保留约定的统计字段。
        "服务统计": (['case_id', 'confirmed_failure', 'downtime_hours'], statistics),

        # 已提供示例：编号用于匹配，输出只保留共同设备令牌。
        "共同设备匹配": (['device_id'], ['device_token']),

        # 【待填 04-B】联合告警统计：(必要输入列表, 允许输出列表)，两个 [] 都要填写。
        # 来源表：任务1清洗后的华辰cn_devices.csv与德维de_devices.csv；供任务4汇总alarm_count。
        # 提示：同时考虑设备去重依据和告警贡献；输出是聚合统计及其口径，可参考 statistics。
        # ========== 作答区 04-B 开始 ==========
        "联合告警统计": (["device_id", "alarm_count"], statistics),
        # ========== 作答区 04-B 结束 ==========

        # 【待填 04-C】联合故障模型：填写必要输入列表；输出parameter、value两列已提供。
        # 来源表：任务1清洗后的cn_devices.csv与de_devices.csv；供任务4训练并输出模型参数。
        # 提示：区分拆分依据、观察期特征和预测标签；特征为temperature_c、operating_hours、alarm_count。
        # 读取failure_next_90d用于监督，不把它作为预测特征。
        # ========== 作答区 04-C 开始 ==========
        "联合故障模型": (
            ["device_id", "temperature_c", "operating_hours", "alarm_count", "failure_next_90d"],
            ["parameter", "value"],
        ),
        # ========== 作答区 04-C 结束 ==========

    }
    return field_choices


def choose_protection_methods():
    """填写六种用途的保护方法和理由；实际方法由任务3、4接入执行。"""
    # TODO05【核心填空与接入】：为研究数据与联合告警统计选择保护方法，并说明与必要字段、输出的关系。
    # 处理对象：05-A配置德维de_devices.csv的研究用途；05-B配置cn_devices.csv与de_devices.csv的告警汇总用途。
    # 来源均为任务1清洗表；与04的字段配置一起写入流转与保护规则.csv，不在本题执行保护算法。
    # 每项格式为(程序接受的方法代号, 具体理由)；NONE 表示尚未选择，填写代号尚未执行保护。
    # 可用代号：ANONYMIZED=泛化与分组抑制；DP=差分隐私统计；FEDAVG=联邦平均训练；
    # PSEUDONYM=稳定假名；PSI=令牌交集匹配；SECURE_SUM_DP=保护求和后加噪。
    # 这些是程序接口名称；按业务需要自行匹配，并说明保留的能力、保护的位置和输出边界。
    # 框架会将选择交给任务3、4，并由任务5结合实际产物检查；ready不等于保护已经执行。
    # 参考可选算法中的独立演示理解判断过程，再结合本任务用途和后续操作填写。
    method_choices = {
        "内部服务分析": (
            "PSEUDONYM",
            "复核需要故障与停机信息；替代设备标识，排除联系资料",
        ),
        # 【待填 05-A】研究数据：(方法代号, 具体理由)，两个字符串都要替换。
        # 提示：观察任务3如何处理组合线索与小分组，说明保留分析能力和降低识别风险之间的关系。
        # ========== 作答区 05-A 开始 ==========
        "研究数据": (
            "ANONYMIZED",
            "泛化地区与运行时长区间，按准标识组合抑制小分组，同时保留型号和故障标签用于关系研究",
        ),
        # ========== 作答区 05-A 结束 ==========

        # 已提供示例：DP保护案例统计贡献，发布机制在任务3接入。
        "服务统计": ('DP', '只输出两个受保护统计及口径，不输出案例明细和真值'),

        # 已提供示例：PSI用于令牌交集，匹配实现仍需在任务4接入。
        "共同设备匹配": ('PSI', '共同设备通过令牌交集匹配，原设备编号留在各方'),

        # 【待填 05-B】联合告警统计：(方法代号, 具体理由)，两个字符串都要替换。
        # 提示：分别考虑求和过程中的暴露与最终合计的泄露；一种过程保护未必覆盖两者。
        # ========== 作答区 05-B 开始 ==========
        "联合告警统计": (
            "SECURE_SUM_DP",
            "使用秘密共享保护双方求和过程，再对重构合计加入Laplace噪声，最终只输出受保护合计及统计口径",
        ),
        # ========== 作答区 05-B 结束 ==========

        # 已提供：双方各自训练后聚合参数；编号用于拆分，未来故障仅作监督标签。
        "联合故障模型": (
            "FEDAVG",
            "三项观察期特征参与本地训练，按样本量聚合参数；编号用于拆分、故障用于监督，只输出模型参数",
        ),

    }
    return method_choices


def check_input_fields(data, catalog, asset, inputs):
    """核对当前来源表和字段目录；返回问题列表，None表示尚未核对，不修改输入。"""
    # 第04题延伸：已选择字段还必须实际存在，且在当前资产的目录中具有有效分级。
    # data为资产名到DataFrame的映射；catalog含asset、field、level；inputs为本用途输入列表。
    # 只筛当前资产和所选字段；分级须覆盖全部输入且为L1—L4。无需把L4直接禁用。
    # 目录按资产和字段唯一登记；框架负责空配置、重复字段、错误汇总和规则保存。
    # 【待填 04-D】核对来源字段及分级，返回问题列表；没有问题返回[]。
    # ========== 作答区 04-D 开始 ==========
    problems = []
    if asset not in data:
        return [f"来源表不存在：{asset}"]
    missing = [field for field in inputs if field not in data[asset].columns]
    if missing:
        problems.append(f"来源表缺少字段：{'|'.join(missing)}")
    selected = catalog.loc[
        catalog["asset"].eq(asset) & catalog["field"].isin(inputs),
        ["field", "level"],
    ]
    counts = selected["field"].value_counts()
    uncovered = [field for field in inputs if counts.get(field, 0) != 1]
    if uncovered:
        problems.append(f"字段分级未唯一覆盖：{'|'.join(uncovered)}")
    invalid = selected.loc[~selected["level"].isin(["L1", "L2", "L3", "L4"]), "field"].tolist()
    if invalid:
        problems.append(f"字段分级无效：{'|'.join(invalid)}")
    return problems
    # ========== 作答区 04-D 结束 ==========

def build_rules(data, catalog, field_choices, method_choices):
    """核对各用途的来源、字段和分级，返回六行规则；不修改数据或执行保护。

    同构检查只做一次。ready表示配置完整且方法代号有效，不判断方法是否适合该用途；
    实际数值质量与保护结果由任务3—5检查。
    """
    supported_methods = {"PSEUDONYM", "ANONYMIZED", "DP", "PSI", "SECURE_SUM_DP", "FEDAVG"}
    rows = []
    for name, spec in USES.items():
        inputs, outputs = field_choices[name]
        method, reason = method_choices[name]
        problems = []
        selected = {}

        if not isinstance(inputs, list) or not inputs:
            problems.append("必要输入字段尚未填写")
            inputs = []
        elif len(inputs) != len(set(inputs)):
            problems.append("必要输入字段重复")
        for asset in spec["sources"]:
            selected[asset] = list(inputs)
            field_problems = check_input_fields(data, catalog, asset, inputs)
            if field_problems is None:
                problems.append(asset + "的04-D字段与分级核对尚未完成")
            elif not isinstance(field_problems, list) or not all(isinstance(item, str) for item in field_problems):
                raise ValueError("04-D须返回问题字符串列表；无问题返回[]，未完成返回None")
            else:
                problems.extend(field_problems)

        if not isinstance(outputs, list) or not outputs:
            problems.append("允许输出字段尚未填写")
            outputs = []
        elif len(outputs) != len(set(outputs)):
            problems.append("允许输出字段重复")
        if not isinstance(method, str):
            problems.append("保护方法代号须为字符串")
            # 诊断也需要保存为规则表；不把列表、字典等对象交给状态翻译。
            method = "无效方法代号类型：" + type(method).__name__
        elif not method.strip() or method == "NONE":
            problems.append("保护方法尚未选择")
        elif method not in supported_methods:
            problems.append("未知的保护方法代号：" + method)
        if not isinstance(reason, str) or not reason.strip():
            problems.append("保护理由尚未填写")
            reason = ""

        rule_ready = not problems
        if problems:
            reason = "；".join(problems) + "；" + reason
        rows.append({
            "result_name": name,
            "sources": "|".join(spec["sources"]),
            "input_fields": json.dumps(selected, ensure_ascii=False, sort_keys=True),
            "output_fields": "|".join(outputs),
            "protection": method,
            "ready": rule_ready,
            "reason": reason,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
