"""任务3已提供的匿名配置检查、共同样本比较与统计发布流程。

作答配置与方法接入保留在main.py；本模块接收实际选择并生成检查和发布证据。
"""

import numpy as np
import pandas as pd
from 可选算法.laplace_mechanism import laplace_release, _noise_scale as laplace_noise_scale
from 可选算法.gaussian_mechanism import gaussian_release, _noise_scale as gaussian_noise_scale
from 可选算法.utility_evaluation import compare_same_cases
from utils import STATISTIC_TOLERANCES, validate_dp_config


STATISTIC_COLUMNS = ["query", "released_value", "mechanism", "epsilon", "delta", "privacy_unit",
                     "adjacency", "public_n", "contribution_limit", "records_per_unit"]


def validate_research_config(config, devices):
    """核验正式匿名配置的整数参数、区域覆盖及设备索引，返回可直接登记的配置。
    只判断公开约束，不使用标签或测试集AUC决定正式方案。
    """
    if not isinstance(config, dict) or set(config) != {"hours_width", "k", "region_groups"}:
        raise ValueError("匿名配置须包含hours_width、k、region_groups三项")
    width, k, mapping = config["hours_width"], config["k"], config["region_groups"]
    if (type(width) is not int or not 1 <= width <= np.iinfo(np.int64).max
            or type(k) is not int or not 3 <= k <= len(devices)):
        raise ValueError("小时区间宽度须为1到9223372036854775807之间的整数（int64表示范围），正式k须为3到设备数之间的整数")
    if (not isinstance(mapping, dict) or not set(devices.site_region) <= set(mapping)
            or not all(isinstance(key, str) and key.strip() and isinstance(value, str) and value.strip()
                       for key, value in mapping.items())):
        raise ValueError("区域映射须覆盖当前所有区域，键和值为非空字符串")
    if not devices.index.is_unique:
        raise ValueError("设备表必须保留唯一索引")
    return {"hours_width": width, "k": k, "region_groups": dict(mapping)}


def compare_research_plans(devices, plans, quasi):
    """比较同一批设备的各匿名方案，返回比较表和评价设备表；不修改输入或写文件。"""
    # 已提供：生成本次比较证据，分析与解释写入任务报告。
    # 只在评价副本按稳定设备编号排序；保留来源索引，不改变原表或源CSV。
    common = devices.sort_values("device_id", kind="stable").index
    for plan in plans.values():
        protected = plan["protected_data"]
        if protected is not None:
            if not isinstance(protected, pd.DataFrame) or not set([*quasi, "failure_next_90d"]) <= set(protected):
                raise ValueError("匿名抑制须返回含准标识与真实标签的DataFrame")
            if protected.empty or protected[quasi].isna().any().any():
                raise ValueError("匿名方案须发布非空数据，准标识不能缺失")
            if (
                not protected.index.is_unique
                or not protected.index.isin(devices.index).all()
            ):
                raise ValueError("匿名方案须保留原始唯一索引")
            if not protected.failure_next_90d.equals(
                devices.loc[protected.index, "failure_next_90d"]
            ):
                raise ValueError("不能修改标签或重排索引后冒充同一设备")
            common = common.intersection(protected.index, sort=False)
    counts = devices.loc[common].failure_next_90d.value_counts()
    if len(common) < 20 or len(counts) != 2 or counts.min() < 2:
        raise ValueError("共同评价样本不足，请调整泛化或抑制方案")
    raw = devices.loc[common]
    baseline = compare_same_cases(
        raw_frame=raw,
        protected_frame=raw,
        label="failure_next_90d",
        raw_numeric=["operating_hours"],
        raw_category=["device_model", "site_region"],
        protected_numeric=["operating_hours"],
        protected_category=["device_model", "site_region"],
        seed=510,
    )
    base_groups = devices.groupby(
        ["device_model", "site_region", "operating_hours"]
    ).failure_next_90d.agg(["size", "nunique"])
    rows = [
        {
            "strategy": "BASELINE",
            "hours_width": 0,
            "k": 1,
            "released_rows": len(devices),
            "suppressed_rows": 0,
            "retention_rate": 1.0,
            "minimum_group": int(base_groups["size"].min()),
            "minimum_diversity": int(base_groups["nunique"].min()),
            "homogeneous_groups": int(base_groups["nunique"].eq(1).sum()),
            "evaluation_devices": len(common),
            "raw_auc": baseline["raw_auc"],
            "protected_auc": baseline["raw_auc"],
            "status": "BASELINE",
        }
    ]
    for name, plan in plans.items():
        width = plan["hours_width"]
        k = plan["k"]
        protected = plan["protected_data"]
        row = {key: pd.NA for key in rows[0]}
        row.update(strategy=name, hours_width=width, k=k, status="NOT_IMPLEMENTED")
        if protected is not None:
            utility = compare_same_cases(
                raw_frame=raw,
                protected_frame=protected.loc[common],
                label="failure_next_90d",
                raw_numeric=["operating_hours"],
                raw_category=["device_model", "site_region"],
                protected_numeric=[],
                protected_category=quasi,
                seed=510,
            )
            groups = protected.groupby(quasi).failure_next_90d.agg(["size", "nunique"])
            row.update(
                released_rows=len(protected),
                suppressed_rows=len(devices) - len(protected),
                retention_rate=len(protected) / len(devices),
                minimum_group=int(groups["size"].min()),
                minimum_diversity=int(groups["nunique"].min()),
                homogeneous_groups=int(groups["nunique"].eq(1).sum()),
                evaluation_devices=len(common),
                raw_auc=utility["raw_auc"],
                protected_auc=utility["protected_auc"],
                status="EVALUATED",
            )
        rows.append(row)
    evaluation_devices = devices.loc[common, ["device_id"]].copy()
    return pd.DataFrame(rows), evaluation_devices


def check_statistics_config(cases, truth, settings, publish):
    """检查配置、实际函数身份及三组模拟预算，返回检查记录；不发布或写文件。

    未填配置可作为基线；已填但不合法的配置必须失败。模拟尺度也在试验前全部核验。
    """
    mechanism_name = settings["confirmed_case_count"]["mechanism"]
    if publish is laplace_release:
        actual_mechanism = "LAPLACE"
    elif publish is gaussian_release:
        actual_mechanism = "GAUSSIAN"
    else:
        actual_mechanism = None
    configured = publish is not None and all(
        all(value is not None for value in item.values()) for item in settings.values()
    )
    config = {}
    for query, item in settings.items():
        config[query] = {
            "sensitivity": item["sensitivity"], "epsilon": item["epsilon_share"],
            "delta": item["delta"], "mechanism": item["mechanism"],
        }
    parameter_error = ""
    simulation_scales = {}
    try:
        validate_dp_config(config, len(cases), allow_missing=not configured)
        if publish is not None and actual_mechanism is None:
            raise ValueError("publish须直接选择所给laplace_release或gaussian_release函数")
        if mechanism_name is not None and actual_mechanism is not None and mechanism_name != actual_mechanism:
            raise ValueError("机制名称与实际发布函数不一致，不能仅改名称冒充另一机制")
        if configured:
            for budget in [0.2, 0.5, 1.0]:
                for query, value in truth.items():
                    epsilon = budget * settings[query]["epsilon_share"]
                    sensitivity = settings[query]["sensitivity"]
                    if actual_mechanism == "GAUSSIAN":
                        scale = gaussian_noise_scale(value, sensitivity, epsilon, settings[query]["delta"])
                    else:
                        scale = laplace_noise_scale(value, sensitivity, epsilon)
                    simulation_scales[(query, epsilon)] = scale
    except (ValueError, OverflowError, FloatingPointError) as error:
        parameter_error = "参数检查失败：" + str(error)
    return {
        "configured": configured,
        "actual_mechanism": actual_mechanism,
        "requested_mechanism": mechanism_name,
        "config": config,
        "simulation_scales": simulation_scales,
        "error": parameter_error,
    }


def compare_statistics_errors(truth, settings, checked):
    """用独立固定种子比较三组预算的误差，返回误差表及失败原因；不发布或写文件。

    任一查询或预算的模拟失败，会清空整组试验成绩，防止部分成功冒充完整发布条件。
    """
    parameter_error = checked["error"]
    can_simulate = checked["configured"] and not parameter_error
    actual_mechanism = checked["actual_mechanism"]
    evaluation = []
    for budget in [0.2, 0.5, 1.0]:
        for number, (query, value) in enumerate(truth.items()):
            if parameter_error:
                trial_status = "CHECK_FAILED"
            else:
                trial_status = "NOT_IMPLEMENTED"
            if can_simulate:
                mechanism = actual_mechanism
            else:
                mechanism = checked["requested_mechanism"]
            row = {
                "query": query, "mechanism": mechanism,
                "total_budget": budget, "true_value": value, "epsilon": pd.NA, "delta": pd.NA,
                "trial_count": 0, "mean_absolute_error": pd.NA, "error_std": pd.NA,
                "tolerance": STATISTIC_TOLERANCES[query], "mean_error_within_tolerance": pd.NA,
                "status": trial_status,
            }
            if can_simulate:
                epsilon = budget * settings[query]["epsilon_share"]
                scale = checked["simulation_scales"][(query, epsilon)]
                rng = np.random.default_rng(511 + 1000 * number)
                try:
                    with np.errstate(over="raise", invalid="raise", divide="raise"):
                        if actual_mechanism == "GAUSSIAN":
                            noise = rng.normal(0, scale, size=200)
                        else:
                            noise = rng.laplace(0, scale, size=200)
                        estimates = value + noise
                        if not np.isfinite(estimates).all():
                            raise ValueError("噪声试验产生非有限值")
                        errors = estimates - value
                        mean_error = float(np.abs(errors).mean())
                        error_std = float(errors.std())
                        if not np.isfinite([mean_error, error_std]).all():
                            raise ValueError("误差聚合产生非有限值")
                    row.update(
                        epsilon=epsilon, delta=settings[query]["delta"], trial_count=len(errors),
                        mean_absolute_error=mean_error, error_std=error_std,
                        mean_error_within_tolerance=bool(mean_error <= row["tolerance"]),
                        status="EVALUATED",
                    )
                except (ValueError, OverflowError, FloatingPointError) as error:
                    parameter_error = "参数检查失败：误差模拟超出有限数值范围；" + str(error)
                    can_simulate = False
            evaluation.append(row)
    if parameter_error:
        for row in evaluation:
            row.update(
                epsilon=pd.NA, delta=pd.NA, trial_count=0, mean_absolute_error=pd.NA,
                error_std=pd.NA, mean_error_within_tolerance=pd.NA, status="CHECK_FAILED",
            )
    return pd.DataFrame(evaluation), parameter_error


def evidence_value(value):
    """将错误配置中的非有限值转换成诊断文本，避免保存失败证据时再次报错。"""
    if value is None or isinstance(value, (str, bool)):
        return value
    try:
        if np.isfinite(float(value)):
            if isinstance(value, (int, np.integer)):
                return int(value)
            return float(value)
    except (OverflowError, TypeError, ValueError):
        pass
    try:
        return str(value)
    except (OverflowError, TypeError, ValueError):
        if isinstance(value, (int, np.integer)):
            return f"整数超出可表示范围（二进制位数{int(value).bit_length()}）"
        return type(value).__name__ + "：无法转换为数值或诊断文本"


def failed_statistics_result(config, reason):
    """返回未发布的空统计表及可保存的错误配置；不写文件，不保留部分发布结果。"""
    evidence = {}
    for query, item in config.items():
        evidence[query] = {key: evidence_value(value) for key, value in item.items()}
    return {
        "frame": pd.DataFrame(columns=STATISTIC_COLUMNS),
        "method": "DP", "ready": False, "details": reason,
        "status": "CHECK_FAILED", "protection_config": evidence,
    }


def publish_statistics_result(cases, truth, settings, publish, checked):
    """执行已经检查的两项正式发布，返回结果、明确状态及实际配置；不写文件。

    只调用给定发布函数的系统随机源；任何一项失败都不返回部分发布表。
    """
    config = checked["config"]
    if not checked["configured"] or checked["error"]:
        return failed_statistics_result(config, "发布前必须完成配置和预算检查")
    actual_mechanism = checked["actual_mechanism"]
    if actual_mechanism == "LAPLACE":
        expected_publish = laplace_release
    elif actual_mechanism == "GAUSSIAN":
        expected_publish = gaussian_release
    else:
        return failed_statistics_result(config, "正式发布机制尚未核验")
    if publish is not expected_publish:
        return failed_statistics_result(config, "检查后不能更换正式发布函数")
    rows = []
    try:
        for query, value in truth.items():
            item = settings[query]
            if actual_mechanism == "GAUSSIAN":
                released_value = publish(value, item["sensitivity"], item["epsilon_share"], item["delta"])
            else:
                released_value = publish(value, item["sensitivity"], item["epsilon_share"])
            if not np.isfinite(released_value):
                raise ValueError(query + "的正式发布结果不是有限数值")
            if query == "confirmed_case_count":
                contribution_limit = 1
            else:
                contribution_limit = 72
            rows.append({
                "query": query, "released_value": released_value,
                "mechanism": actual_mechanism, "epsilon": item["epsilon_share"], "delta": item["delta"],
                "privacy_unit": "SERVICE_CASE", "adjacency": "REPLACE_ONE_FIXED_PUBLIC_N",
                "public_n": len(cases), "contribution_limit": contribution_limit, "records_per_unit": 1,
            })
    except (ValueError, OverflowError, FloatingPointError) as error:
        return failed_statistics_result(config, "正式发布失败：" + str(error))
    return {
        "frame": pd.DataFrame(rows), "method": "DP", "ready": True,
        "details": "本次实际使用" + actual_mechanism + "系统随机发布；真值与误差仅保存在本地统计误差.csv",
        "status": "RECORDED", "protection_config": config,
    }


def check_research_transform(devices, protected, width, k, region_groups, quasi):
    """独立核对07-B的映射、区间及完整保留组；不替学生生成发布结果。"""
    expected = devices.copy()
    expected["region_group"] = expected.site_region.map(region_groups)
    lower = (expected.operating_hours // width * width).astype(int)
    expected["hours_band"] = "[" + lower.astype(str) + "," + (lower + width).astype(str) + ")"
    counts = expected.groupby(quasi, dropna=False)["device_id"].transform("size")
    expected = expected.loc[counts.ge(k)]
    if (not isinstance(protected, pd.DataFrame) or not protected.index.is_unique
            or set(protected.columns) != set(expected.columns)
            or not protected[expected.columns].sort_index().equals(expected.sort_index())):
        raise ValueError("07-B须按本轮配置泛化并整组抑制，保留来源索引、原字段和值；不能造行或修改标签")
