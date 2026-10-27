"""任务5已提供的结果核验：独立检查实际文件、来源、规则和匹配依赖。
内部分析的作答判断由 main.py 通过参数传入；本模块不导入任务入口。
"""

from pathlib import Path
import hashlib
import hmac
import json
import numpy as np
import pandas as pd
from utils import (
    task_dir, run_dir, read_clean, clean_path, sha256, record_hash,
    load_rules, load_results, read_events, FEATURES, USES, read_table,
    verify_lineage, LOCAL_EVIDENCE, DEMO_TOKEN_KEY, MATCH_TOKEN_DOMAIN,
    validate_dp_config, result_stage_status,
)


def tokens_for(values, domain, length):
    """按给定协议独立重建设备令牌，供内容和匹配核验使用。
    输入标识序列、用途域及长度，返回同索引令牌序列；不信任生成阶段的完成声明。
    """
    # 按给定HMAC表示独立核验，不调用任务3/4的生成结果或ready判断。
    def token_for(value):
        """按同一用途域重建单个设备令牌。"""
        message = f"{domain}:{value}".encode()
        return hmac.new(DEMO_TOKEN_KEY, message, hashlib.sha256).hexdigest()[:length]

    return values.map(token_for)


def same_records(actual, expected):
    """核对实际表与预期表的完整记录及重复次数是否相同。
    输入两张表，返回布尔值；允许行重排和微小浮点往返误差。
    """
    # 比较每条记录及重复次数，允许CSV行重排；仅容忍浮点往返误差。
    columns = list(expected)
    try:
        pd.testing.assert_frame_equal(
            actual[columns].sort_values(columns).reset_index(drop=True),
            expected.sort_values(columns).reset_index(drop=True),
            check_dtype=False, check_exact=False, rtol=0, atol=1e-10,
        )
    except AssertionError:
        return False
    return True


def inspect_result(name, artifact, validate_internal, artifacts=None):
    """独立核对实际结果、来源及匹配依赖；内部分析判断调用第12题，返回状态和原因。"""
    # 已提供：方法、字段、摘要、来源、依赖及五类结果的独立内容检查。
    # 内部服务分析的待补内容判断位于validate_internal，其余检查已提供。
    def validate(rule, artifact, frame):
        """按用途核对实际结果表与已登记的方法、配置和来源内容。
        输入规则、登记和结果表，返回布尔判断；内部分析未填时允许返回None。
        """
        if artifact["ready"] is not True or artifact["method"] != rule["protection"]:
            return False
        method = artifact["method"]
        expected_methods = {
            "内部服务分析": "PSEUDONYM",
            "研究数据": "ANONYMIZED",
            "服务统计": "DP",
            "共同设备匹配": "PSI",
            "联合告警统计": "SECURE_SUM_DP",
            "联合故障模型": "FEDAVG",
        }
        if method != expected_methods[artifact["result_name"]] or set(frame) != set(
            rule["output_fields"].split("|")
        ):
            return False
        if method == "PSEUDONYM":
            return validate_internal(frame, read_clean("service_cases.csv"))
        elif method == "PSI":
            cn, de = read_clean("cn_devices.csv"), read_clean("de_devices.csv")
            left = tokens_for(cn.device_id, MATCH_TOKEN_DOMAIN, 64)
            right = tokens_for(de.device_id, MATCH_TOKEN_DOMAIN, 64)
            valid = (
                set(frame) == {"device_token"}
                and frame.device_token.astype(str).str.fullmatch(r"[0-9a-f]{64}").all()
                and frame.device_token.is_unique
                and cn.device_id.is_unique and de.device_id.is_unique
                and left.nunique() == len(cn) and right.nunique() == len(de)
                and set(frame.device_token) == set(left) & set(right)
                and len(frame) == len(set(cn.device_id) & set(de.device_id))
            )
        elif method in {"DP", "SECURE_SUM_DP"}:
            expected = {
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
            }
            queries = (
                {"confirmed_case_count", "mean_downtime"}
                if method == "DP"
                else {"fleet_alarm_sum"}
            )
            if method == "DP":
                public_n = len(read_clean("service_cases.csv"))
                unit, limits = "SERVICE_CASE", {
                    "confirmed_case_count": 1,
                    "mean_downtime": 72,
                }
            else:
                public_n = len(
                    set(read_clean("cn_devices.csv").device_id)
                    | set(read_clean("de_devices.csv").device_id)
                )
                unit, limits = "DEVICE", {"fleet_alarm_sum": 8}
            config = json.loads(artifact["protection_config"])
            validate_dp_config(config, public_n, unit)
            # 登记的实际配置与发布声明逐项对应；不能只凭机制名称判定通过。
            for row in frame.to_dict("records"):
                item = config[row["query"]]
                if any(row[field] != item[field] for field in ["epsilon", "delta", "mechanism"]):
                    raise ValueError("发布声明与实际保护配置不一致")
            valid = (
                set(frame) == expected
                and set(frame["query"]) == queries
                and len(frame) == len(queries)
                and np.isfinite(frame.released_value).all()
                and frame.epsilon.gt(0).all()
                and frame.epsilon.sum() <= 1
                and frame.delta.ge(0).all()
                and frame.delta.sum() <= 1e-5
                and frame.privacy_unit.eq(unit).all()
                and frame.adjacency.eq("REPLACE_ONE_FIXED_PUBLIC_N").all()
                and frame.public_n.eq(public_n).all()
                and frame.records_per_unit.eq(1).all()
                and frame.contribution_limit.eq(frame["query"].map(limits)).all()
                # 配置已按用途逐机制核验；案例可选两路线，设备合计仍固定Laplace。
                and frame.mechanism.isin({"LAPLACE", "GAUSSIAN"} if method == "DP" else {"LAPLACE"}).all()
                and frame.mechanism.nunique() == 1
            )
        elif method == "FEDAVG":
            valid = (
                set(frame) == {"parameter", "value"}
                and set(frame.parameter) == {*FEATURES, "bias"}
                and len(frame) == 4
                and np.isfinite(frame.value).all()
            )
        elif method == "ANONYMIZED":
            quasi = ["device_model", "region_group", "hours_band"]
            scheme = json.loads((task_dir(3) / "匿名化方案.json").read_text(encoding="utf-8"))
            if not isinstance(scheme, dict) or set(scheme) != {"hours_width", "k", "region_groups"}:
                return False
            width, k, mapping = scheme["hours_width"], scheme["k"], scheme["region_groups"]
            devices = read_clean("de_devices.csv")
            if (
                type(width) is not int or not 1 <= width <= np.iinfo(np.int64).max
                or type(k) is not int or not 3 <= k <= len(devices)
                or not isinstance(mapping, dict)
                or not set(devices.site_region) <= set(mapping)
                or not all(isinstance(value, str) and value.strip() for value in mapping.values())
            ):
                return False
            # 从当前原记录和声明的方案独立重建，保留整组及其真实标签。
            expected = devices[["device_model", "failure_next_90d"]].copy()
            expected["region_group"] = devices.site_region.map(mapping)
            lower = (devices.operating_hours // width * width).astype(int)
            expected["hours_band"] = "[" + lower.astype(str) + "," + (lower + width).astype(str) + ")"
            group_size = expected.groupby(quasi).failure_next_90d.transform("size")
            expected = expected.loc[group_size >= k, [*quasi, "failure_next_90d"]]
            valid = (
                set(frame) == {*quasi, "failure_next_90d"}
                and 0 < len(frame) <= len(read_clean("de_devices.csv"))
                and frame.notna().all().all()
                and frame.failure_next_90d.isin([0, 1]).all()
                and frame.groupby(quasi).size().min() >= k
                and same_records(frame, expected)
            )
        else:
            valid = False

        return valid

    if artifact is None:
        return "NOT_RUN", "本次未生成该结果，请查看前置步骤的检查记录"
    try:
        # 保存时的结构化状态与登记摘要绑定；原因文字只供解释，不承担状态判断。
        stage_status = result_stage_status(artifact)
        if stage_status in {"CHECK_FAILED", "QUALITY_PENDING"}:
            return stage_status, artifact["details"]
        rule = load_rules()[name]
        if rule["ready"] is not True or artifact["method"] == "NONE":
            return "BASELINE_ONLY", "规则或目标保护尚未完成；" + str(artifact.get("details", ""))
        if artifact["ready"] is not True:
            return "CHECK_FAILED", "阶段检查未通过；" + str(artifact.get("details", "查看生成任务"))
        frame, parents = read_result_inputs(name, rule, artifact, artifacts)
        for parent in parents:
            status, _ = inspect_result(parent["result_name"], parent, validate_internal, artifacts)
            if status != "CHECK_PASSED":
                return "CHECK_FAILED", "前置匹配结果未通过当前检查"
        valid = validate(rule, artifact, frame)
        if valid is None:
            return "BASELINE_ONLY", "内部服务分析内容检查尚未完成"
    except OSError as error:
        filename = Path(error.filename).name if error.filename else "相关文件"
        return "CHECK_FAILED", f"文件缺失或不可读：{filename}，请检查本次结果及其来源"
    except ValueError as error:
        return "CHECK_FAILED", str(error)
    except (TypeError, KeyError, AttributeError) as error:
        return "CHECK_FAILED", f"结果或登记字段的结构、类型无效（{type(error).__name__}），请核对本任务输出要求"
    passed_reason = (
        "本课发布字段、配置及必要条件检查通过；方法执行结合代码、登记和报告判断"
        if artifact["method"] in {"DP", "SECURE_SUM_DP", "FEDAVG"}
        else "实际文件符合本次保护规则及内容约定"
    )
    if isinstance(valid, (bool, np.bool_)) and bool(valid):
        return "CHECK_PASSED", passed_reason
    return "CHECK_FAILED", "实际方法、字段或数值不符合输出要求"


def read_result_inputs(name, rule, artifact, artifacts):
    """核对自动登记、来源和匹配依赖；内部内容判断由validate_internal完成。"""
    root = run_dir().resolve()
    verify_lineage()
    if artifact["result_name"] != name or artifact["task"] != USES[name]["task"]:
        raise ValueError("结果名称或任务不一致")
    if artifact["rule_sha256"] != record_hash(rule):
        raise ValueError("规则已改变，请重新生成结果")
    if rule["sources"].split("|") != USES[name]["sources"]:
        raise ValueError("规则来源与用途不一致")
    selected = json.loads(rule["input_fields"])
    catalog = read_table(task_dir(1) / "字段分级.csv")
    if set(selected) != set(USES[name]["sources"]):
        raise ValueError("规则缺少必要来源")
    for source, fields in selected.items():
        if (
            not fields
            or len(fields) != len(set(fields))
            or not set(fields) <= set(read_clean(source))
        ):
            raise ValueError("规则中的输入字段无效")
        classified = catalog[(catalog.asset == source) & catalog.field.isin(fields)]
        if len(classified) != len(fields) or not classified.level.isin(["L1", "L2", "L3", "L4"]).all():
            raise ValueError("当前使用字段尚未完成分级")
    path = (root / artifact["file"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("结果路径超出本次运行目录")
    if sha256(path) != artifact["sha256"]:
        raise ValueError(f"结果文件摘要已变化：{path.name}，请重新生成结果")
    saved = [event for event in read_events()
             if event.get("action") == "保存处理结果" and event.get("result_name") == name]
    if len(saved) != 1 or saved[0].get("metadata_sha256") != record_hash(artifact):
        raise ValueError("结果登记或实际保护配置与保存时记录不一致，请重新生成")
    dependencies = json.loads(artifact["dependencies"])
    required = 1 if name in {"联合告警统计", "联合故障模型"} else 0
    if not isinstance(dependencies, list) or len(dependencies) != required:
        raise ValueError("结果缺少必要的实际匹配依赖，或含多余依赖")
    registry = artifacts if artifacts is not None else load_results()
    parents = []
    expected = {
        str(clean_path(source).relative_to(root)) for source in USES[name]["sources"]
    }
    if name in LOCAL_EVIDENCE:
        task, filename = LOCAL_EVIDENCE[name]
        expected.add(str((task_dir(task) / filename).relative_to(root)))
    for dependency in dependencies:
        parent = registry.get(dependency["result_name"])
        if (
            parent is None
            or parent["result_name"] != "共同设备匹配"
            or parent["method"] != "PSI"
            or record_hash(parent) != dependency["metadata_sha256"]
        ):
            raise ValueError("前置匹配的登记或方法已改变")
        expected.add(parent["file"])
        parents.append(parent)
    inputs = json.loads(artifact["input_sha256"])
    if set(inputs) != expected:
        raise ValueError("输入登记不完整或含多余来源")
    for relative, digest in inputs.items():
        source = (root / relative).resolve()
        if not source.is_relative_to(root) or sha256(source) != digest:
            raise ValueError(f"实际输入已改变：{relative}，请重新运行前置任务")
    if name == "研究数据":
        # 区域组名允许任意非空字符串；保留数字、前导零和NA/NULL等合法文本。
        categories = {field: str for field in ["device_model", "region_group", "hours_band"]}
        return pd.read_csv(path, dtype=categories, keep_default_na=False), parents
    return pd.read_csv(path), parents
