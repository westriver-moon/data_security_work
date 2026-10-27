"""任务3：按保护规则准备三类实际结果，并比较保护与效用。"""
# 【待填】说明题目，成对“作答区”边框标出填写范围；先阅读框外提示，再在框内完成作答。
# 本文件为第06—08题，真正空位查找【待填】；其余None是已提供框架的状态值，无需替换。

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# 先按题号理解作答的输入、判断与结果；需要调用上下文时再查main()。
import pandas as pd
import json
from release_support import (
    validate_research_config,
    check_research_transform,
    compare_research_plans,
    check_statistics_config,
    compare_statistics_errors,
    failed_statistics_result,
    publish_statistics_result,
)
from 可选算法.laplace_mechanism import laplace_release
from 可选算法.gaussian_mechanism import gaussian_release
from utils import (
    prepare_task_run,
    select_inputs,
    save_and_register_result,
    save_results,
    write_csv,
    task_dir,
    INTERNAL_TOKEN_DOMAIN,
    numeric_quality_issue,
    TABLE_COLUMNS,
)

# 选用方法时在对应作答框内导入；候选来自本任务的可选算法目录，公共常量来自utils。

def main():
    """分别按用途取数，生成三份结果，再保存并登记实际方法、状态和来源。"""
    # 直接运行本文件时，先由根入口准备同一次运行及必要前序任务。
    prepare_task_run(3)

    # 内部分析：按自己的字段规则读取服务案例，替换设备标识并保留关联能力。
    internal_inputs = select_inputs("内部服务分析")
    internal_cases = internal_inputs["service_cases.csv"]
    internal_result = prepare_internal(internal_cases)
    internal_record = save_and_register_result(
        result_name="内部服务分析",
        frame=internal_result["frame"], method=internal_result["method"],
        ready=internal_result["ready"], details=internal_result["details"],
        status=internal_result["status"], protection_config=internal_result["protection_config"],
    )

    # 研究数据：明确使用德维设备，按型号、泛化区域和运行时长区间分组。
    research_inputs = select_inputs("研究数据")
    research_devices = research_inputs["de_devices.csv"]
    research_result = prepare_research(research_devices)
    research_record = save_and_register_result(
        result_name="研究数据",
        frame=research_result["frame"], method=research_result["method"],
        ready=research_result["ready"], details=research_result["details"],
        status=research_result["status"], protection_config=research_result["protection_config"],
    )

    # 服务统计另按统计规则取案例字段；它与内部分析的必要输入不同。
    statistics_inputs = select_inputs("服务统计")
    statistics_cases = statistics_inputs["service_cases.csv"]
    statistics_result = prepare_statistics(statistics_cases)
    statistics_record = save_and_register_result(
        result_name="服务统计",
        frame=statistics_result["frame"], method=statistics_result["method"],
        ready=statistics_result["ready"], details=statistics_result["details"],
        status=statistics_result["status"], protection_config=statistics_result["protection_config"],
    )
    save_results(3, [internal_record, research_record, statistics_record])
    print("已保存三类结果及实际状态；等待治理时，匿名效用和停机统计尚未计算。")

def prepare_internal(cases):
    """为服务案例准备可关联设备的内部分析结果，并计算四项假名检查。
    输入案例表；返回包含结果表、方法、状态、说明及显式配置的字典。
    只生成当前假名检查文件，不修改输入；main负责保存最终业务结果。
    未接入假名方法时保留明文基线；正式结果仅含规定的三列。
    """
    # TODO06【核心填空与接入】：为内部服务分析接入稳定、按用途隔离的假名方法。
    # 处理对象：任务1清洗后的德维service_cases.csv；将device_id转为device_token，
    # 保留downtime_hours与confirmed_failure，生成本任务结果文件/内部服务分析.csv。
    # 输入(device_id, domain)，返回32个小写十六进制字符；沿用给定HMAC-SHA256协议与DEMO_TOKEN_KEY。
    # 保留字段和四项检查已提供；完成后查看假名检查.csv与内部服务分析结果。
    domain, other_domain = INTERNAL_TOKEN_DOMAIN, INTERNAL_TOKEN_DOMAIN + ":other"
    # 提示：先查看hmac_pseudonymization的方法签名；框架只传入编号和用途域，需在回调内补齐其余协议参数。
    # 输入是(device_id, domain)，返回一个字符串；此处填写可调用对象，不是一次调用得到的令牌。
    # 可在本函数内用普通def接收这两个参数，在函数体内补齐固定参数，再把函数名赋给make_token。
    # 函数对象与调用结果的区别，见任务1的任务背景“回调提示”。
    # 【待填 06-A】接入假名回调；None使初始程序保留本地原值基线。
    # 已给接入外壳：def make_token(value, domain): 后缩进return所选假名函数调用。
    # 固定调用参数为value、DEMO_TOKEN_KEY、domain、length=32；在框内导入所选方法及utils中的密钥。
    # ========== 作答区 06-A 开始 ==========
    from 可选算法.hmac_pseudonymization import pseudonymize
    from utils import DEMO_TOKEN_KEY

    def make_token(value, domain):
        return pseudonymize(value, DEMO_TOKEN_KEY, domain, length=32)
    # ========== 作答区 06-A 结束 ==========

    kept_fields = [
        "downtime_hours",
        "confirmed_failure",
    ]  # 已给定保留的案例字段；新生成的device_token由下面的代码加入。
    names = [
        "SAME_INPUT_STABLE",
        "DOMAIN_SEPARATED",
        "NO_TOKEN_COLLISION",
        "MINIMIZED_FIELDS",
    ]
    if make_token is None:
        checks = pd.DataFrame(
            [
                {"check": name, "passed": pd.NA, "reason": "NOT_IMPLEMENTED"}
                for name in names
            ]
        )
        frame, method, ready = cases.copy(), "NONE", False
    else:
        frame = cases[kept_fields].copy()

        def token_for_internal_use(value):
            return make_token(value, domain)

        def token_for_other_use(value):
            return make_token(value, other_domain)

        tokens = cases.device_id.map(token_for_internal_use)
        frame.insert(0, "device_token", tokens)
        repeated = cases.device_id.map(token_for_internal_use)
        other = cases.device_id.map(token_for_other_use)
        valid_tokens = (
            tokens.astype("string").str.fullmatch(r"[0-9a-f]{32}").fillna(False).all()
        )
        passed = [
            tokens.eq(repeated).all(),
            tokens.ne(other).all(),
            valid_tokens and tokens.nunique() == cases.device_id.nunique(),
            set(frame) == {"device_token", "downtime_hours", "confirmed_failure"},
        ]
        checks = pd.DataFrame(
            [
                {"check": name, "passed": bool(value), "reason": "由本次输入实际计算"}
                for name, value in zip(names, passed)
            ]
        )
        method, ready = "PSEUDONYM", all(passed)
    write_csv(3, "假名检查.csv", checks)
    if make_token is None:
        status = "BASELINE_ONLY"
        details = "尚未接入假名方法；当前仅保留本地原值基线，四项检查待完成"
    elif ready:
        status = "RECORDED"
        details = "四项实际检查见假名检查.csv"
    else:
        status = "CHECK_FAILED"
        details = "假名检查未全部通过；定位见假名检查.csv"
    return {
        "frame": frame, "method": method, "ready": bool(ready), "details": details,
        "status": status, "protection_config": {},
    }


def prepare_research(devices):
    """按自主选择的区间、k和区域映射泛化设备，并接入整组抑制。
    返回包含结果表、方法、状态、说明及显式配置的字典，不修改输入。
    必要时长未治理时暂停分组与评价；数值合格而选择或方法缺失时保留本地基线。
    本函数显式保存匿名配置和比较证据，最终业务结果由main保存。
    """
    issue = numeric_quality_issue(devices, {"operating_hours": (0, 744)})
    if issue:
        # 保留来源基线及待计算状态；不代替任务1估计数值，也不丢弃设备。
        write_csv(3, "匿名分组.csv", pd.DataFrame(columns=TABLE_COLUMNS["匿名分组.csv"]))
        write_csv(3, "匿名化比较.csv", pd.DataFrame(
            [{"strategy": "BASELINE", "status": "QUALITY_PENDING"}],
            columns=TABLE_COLUMNS["匿名化比较.csv"],
        ))
        write_csv(3, "评价样本.csv", pd.DataFrame(columns=TABLE_COLUMNS["评价样本.csv"]))
        (task_dir(3) / "匿名化方案.json").write_text("null", encoding="utf-8")
        return {
            "frame": devices.copy(), "method": "NONE", "ready": False,
            "details": "等待任务1数据治理，匿名分组与效用尚未计算：" + issue,
            "status": "QUALITY_PENDING", "protection_config": {},
        }
    quasi = ["device_model", "region_group", "hours_band"]
    baseline_config = {
        "hours_width": 100, "k": 3,
        "region_groups": {"NORTH": "NE", "EAST": "NE", "SOUTH": "SW", "WEST": "SW"},
    }
    # TODO07【选法与方案设计】：选择匿名配置，并在07-B完成泛化及整组抑制。
    # 处理对象：任务1清洗后的德维de_devices.csv（devices）；按device_model分组，泛化site_region与operating_hours。
    # failure_next_90d作为评价/输出标签保留，最终写入本任务结果文件/研究数据.csv。
    # 可选100小时/k=3、200小时/k=3、100小时/k=5；也可给出符合公开约束的宽度、k和区域映射。
    # 提示：配置沿用baseline_config的三键结构；宽度为int64范围内正整数、正式k>=3，映射覆盖当前区域。
    # 选择应依据分组用途和信息精度，先选后评价；不根据测试集AUC挑方案，不造行、不修改标签。
    # 【待填 07-A】填一份正式配置字典；None仅使用比较基线，尚未正式选择。
    # ========== 作答区 07-A 开始 ==========
    selected_config = {
        "hours_width": 100,
        "k": 3,
        "region_groups": {"NORTH": "NE", "EAST": "NE", "SOUTH": "SW", "WEST": "SW"},
    }
    # ========== 作答区 07-A 结束 ==========

    config = baseline_config if selected_config is None else selected_config
    try:
        config = validate_research_config(config, devices)
    except ValueError as error:
        write_csv(3, "匿名分组.csv", pd.DataFrame(columns=[*quasi, "group_size", "distinct_labels"]))
        write_csv(3, "匿名化比较.csv", pd.DataFrame(columns=["strategy", "status"]))
        (task_dir(3) / "匿名化方案.json").write_text("null", encoding="utf-8")
        write_csv(3, "评价样本.csv", pd.DataFrame(columns=TABLE_COLUMNS["评价样本.csv"]))
        return {
            "frame": pd.DataFrame(columns=[*quasi, "failure_next_90d"]),
            "method": "ANONYMIZED", "ready": False,
            "details": "参数检查失败：" + str(error),
            "status": "CHECK_FAILED", "protection_config": {},
        }
    # 已提供：正式配置自动加入plans，防止只改方案名称而没有实际生成。
    settings = [(100, 2), (100, 3), (100, 5), (200, 3)]
    selected_setting = (config["hours_width"], config["k"])
    if selected_setting not in settings:
        settings.append(selected_setting)
    selected = f"K{config['k']}_H{config['hours_width']}"
    region_groups = config["region_groups"]
    plans = {}
    for width, k in settings:
        protected = transform_research(devices.copy(deep=True), width, k, region_groups, quasi)
        if protected is None:
            write_csv(3, "匿名分组.csv", pd.DataFrame(columns=[*quasi, "group_size", "distinct_labels"]))
            write_csv(3, "匿名化比较.csv", pd.DataFrame([{"strategy": "NOT_IMPLEMENTED", "status": "BASELINE_ONLY"}]))
            write_csv(3, "评价样本.csv", pd.DataFrame(columns=TABLE_COLUMNS["评价样本.csv"]))
            (task_dir(3) / "匿名化方案.json").write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
            return {
                "frame": devices.copy(), "method": "NONE", "ready": False,
                "details": "07-B泛化与整组抑制尚未完成；未进行方案效用比较",
                "status": "BASELINE_ONLY", "protection_config": {},
            }
        check_research_transform(devices, protected, width, k, region_groups, quasi)

        plans[f"K{k}_H{width}"] = {
            "hours_width": width, "k": k, "protected_data": protected,
        }

    try:
        comparison, evaluation_devices = compare_research_plans(devices, plans, quasi)
    except ValueError as error:
        write_csv(3, "匿名分组.csv", pd.DataFrame(columns=[*quasi, "group_size", "distinct_labels"]))
        write_csv(3, "匿名化比较.csv", pd.DataFrame(columns=["strategy", "status"]))
        (task_dir(3) / "匿名化方案.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        write_csv(3, "评价样本.csv", pd.DataFrame(columns=TABLE_COLUMNS["评价样本.csv"]))
        return {
            "frame": pd.DataFrame(columns=[*quasi, "failure_next_90d"]),
            "method": "ANONYMIZED", "ready": False,
            "details": "参数检查失败：" + str(error),
            "status": "CHECK_FAILED", "protection_config": {},
        }
    write_csv(3, "匿名化比较.csv", comparison)
    write_csv(3, "评价样本.csv", evaluation_devices)
    protected = plans[selected]["protected_data"]
    (task_dir(3) / "匿名化方案.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if selected_config is None or protected is None:
        write_csv(3, "匿名分组.csv", pd.DataFrame(columns=[*quasi, "group_size", "distinct_labels"]))
        return {
            "frame": devices.copy(), "method": "NONE", "ready": False,
            "details": "正式匿名方案尚未就绪；当前保留本地原始基线",
            "status": "BASELINE_ONLY", "protection_config": {},
        }
    groups = (protected.groupby(quasi).failure_next_90d
              .agg(group_size="size", distinct_labels="nunique").reset_index())
    write_csv(3, "匿名分组.csv", groups)
    ready = len(groups) > 0 and groups.group_size.min() >= config["k"]
    if ready:
        status = "RECORDED"
    else:
        status = "CHECK_FAILED"
    return {
        "frame": protected[[*quasi, "failure_next_90d"]].copy(),
        "method": "ANONYMIZED", "ready": bool(ready),
        "details": f"实际正式方案{selected}；研究结果仅供本地课堂分析，比较见匿名化比较.csv",
        "status": status, "protection_config": {},
    }

def transform_research(devices, width, k, region_groups, quasi):
    """将一轮配置用于设备副本，返回整组抑制后的DataFrame；None表示未完成。"""
    # width/k为本轮比较参数，不能写死正式方案值；quasi为型号、泛化区域和时长区间三列。
    # 区域按region_groups映射；时长区间左闭右开，例如width=100时200进入[200,300)。
    # 可调用k_anonymity.suppress_small_groups(frame, quasi, k)，返回(保留表, 抑制报告)。
    # 保留原索引、标签与来源对应，不修改输入、不造行；返回保留表，报告由框架汇总。
    # 【待填 07-B】按本轮配置完成区域和时长泛化，再按准标识整组抑制。
    # ========== 作答区 07-B 开始 ==========
    from 可选算法.k_anonymity import suppress_small_groups

    protected = devices.copy()
    protected["region_group"] = protected["site_region"].map(region_groups)
    lower = (protected["operating_hours"] // width * width).astype(int)
    protected["hours_band"] = "[" + lower.astype(str) + "," + (lower + width).astype(str) + ")"
    protected, _ = suppress_small_groups(protected, quasi, k)
    return protected
    # ========== 作答区 07-B 结束 ==========

def prepare_statistics(cases):
    """按案例级固定分母口径发布两项统计，支持整组Laplace或Gaussian。
    返回包含结果表、方法、状态、说明及实际配置的字典，不修改案例表。
    必要数值合格后，配置未填时保留精确基线；数值未治理时均值留空并等待。
    本函数保存本地误差证据；配置或试验失败时整组拒绝发布。
    正式函数使用系统随机源，重复误差模拟使用独立且可复现的随机源。
    """
    if cases.empty:
        write_csv(3, "统计误差.csv", pd.DataFrame(columns=TABLE_COLUMNS["统计误差.csv"]))
        return failed_statistics_result({}, "参数检查失败：服务案例不能为空，公开记录总数必须为正整数")
    issue = numeric_quality_issue(cases, {"downtime_hours": (0, None)})
    if issue:
        # 全部案例仍计入公开n；均值等待治理，不跳过缺失值而暗中更换分母。
        rows = [{"query": query, "total_budget": budget, "trial_count": 0, "status": "QUALITY_PENDING"}
                for budget in [0.2, 0.5, 1.0] for query in ["confirmed_case_count", "mean_downtime"]]
        write_csv(3, "统计误差.csv", pd.DataFrame(rows, columns=TABLE_COLUMNS["统计误差.csv"]))
        baseline = pd.DataFrame([
            {"query": "confirmed_case_count", "released_value": float(cases.confirmed_failure.sum())},
            {"query": "mean_downtime", "released_value": pd.NA},
        ])
        return {
            "frame": baseline, "method": "NONE", "ready": False,
            "details": "等待任务1数据治理，停机均值与统计发布尚未计算：" + issue,
            "status": "QUALITY_PENDING", "protection_config": {},
        }
    # 本题约定每条案例最多按72小时计入统计，以限制单条记录的影响。
    # 以下裁剪只生成统计用的贡献副本，不改变cases中的真实停机时长。
    truth = {
        "confirmed_case_count": float(cases.confirmed_failure.sum()),
        "mean_downtime": float(cases.downtime_hours.clip(0, 72).mean()),
    }
    # TODO08【选法与方案设计】：推导两项敏感度，并从两个已给机制中选一条正式发布路线。
    # 处理对象：任务1清洗后的德维service_cases.csv（cases）；对confirmed_failure计数、对downtime_hours计算裁剪贡献均值。
    # 分母为案例条数，结果写入本任务结果文件/服务统计.csv，不改变清洗表中的原始停机值。
    # 候选laplace_mechanism.py或gaussian_mechanism.py；两查询使用同一机制，每项epsilon固定0.5。
    # Laplace每项delta=0，Gaussian每项delta=5e-6；固定案例级、替换一个案例和公开分母。
    # 【待填 08-A】填写"LAPLACE"或"GAUSSIAN"，框架据此提供对应delta。
    # ========== 作答区 08-A 开始 ==========
    mechanism_name = "LAPLACE"
    # ========== 作答区 08-A 结束 ==========

    if mechanism_name == "LAPLACE":
        delta = 0
    elif mechanism_name == "GAUSSIAN":
        delta = 5e-6
    else:
        delta = None
    settings = {
        "confirmed_case_count": {
            # 提示：单个案例的故障贡献为0或1，替换一个案例时比较两个端点对计数的最大影响。
            # 【待填 08-B】填写案例计数敏感度。
            # ========== 作答区 08-B 开始 ==========
            "sensitivity": 1,
            # ========== 作答区 08-B 结束 ==========

            "epsilon_share": 0.5, "delta": delta, "mechanism": mechanism_name,
        },
        "mean_downtime": {
            # 提示：停机贡献裁剪到[0,72]；先求总和最大变化，再考虑固定公开分母len(cases)。
            # 【待填 08-C】填写均值敏感度表达式，不硬编码当前案例数。
            # ========== 作答区 08-C 开始 ==========
            "sensitivity": 72 / len(cases),
            # ========== 作答区 08-C 结束 ==========

            "epsilon_share": 0.5, "delta": delta, "mechanism": mechanism_name,
        },
    }
    # 已提供：08-A只选择一次机制，自动映射真实发布函数；未选择时仍保持未完成。
    publishers = {"LAPLACE": laplace_release, "GAUSSIAN": gaussian_release}
    publish = publishers.get(mechanism_name) if isinstance(mechanism_name, str) else None

    # 先检查敏感度、预算和实际函数身份；这里只核验，不产生正式发布。
    checked = check_statistics_config(cases, truth, settings, publish)

    # 再用独立固定种子比较误差，并显式保存本地证据；其中真值不能作为发布结果。
    error_comparison, simulation_error = compare_statistics_errors(truth, settings, checked)
    write_csv(3, "统计误差.csv", error_comparison)
    if simulation_error:
        return failed_statistics_result(checked["config"], simulation_error)
    if not checked["configured"]:
        baseline = pd.DataFrame([
            {"query": query, "released_value": value} for query, value in truth.items()
        ])
        return {
            "frame": baseline, "method": "NONE", "ready": False,
            "details": "机制选择或敏感度尚未完成；当前保留精确统计基线，误差试验未开展",
            "status": "BASELINE_ONLY", "protection_config": {},
        }

    # 全部检查和模拟通过后，调用系统随机发布；返回实际配置，供main显式登记。
    return publish_statistics_result(cases, truth, settings, publish, checked)


if __name__ == "__main__":
    main()
