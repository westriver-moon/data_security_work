"""任务4：真实匹配、归属去重，再进行受保护统计与联合训练。"""
# 【待填】给出题目要求；每对“作答区”边框标出需要填写的位置。
# 本文件为第09—11题，真正空位查找【待填】；其余None是已提供框架的状态值，无需替换。

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# 先按题号理解作答的输入、判断与结果；需要调用上下文时再查main()。
import pandas as pd
import numpy as np
from analysis_support import (prepare_training, train_models, save_analysis_check,
                              validate_common_sources, check_owned_records)
from 可选算法.hmac_psi import tokenize
from 可选算法.laplace_mechanism import laplace_release
from utils import (
    prepare_task_run,
    read_clean,
    select_inputs,
    save_and_register_result,
    save_results,
    write_csv,
    DEMO_TOKEN_KEY,
    MATCH_TOKEN_DOMAIN,
)

# 求交与保护求和候选位于本任务的可选算法目录；选定后在对应作答框内导入。


def main():
    """依次匹配设备、确定归属、保护告警统计、划分设备并训练模型。"""
    prepare_task_run(4)
    cn = read_clean("cn_devices.csv")
    de = read_clean("de_devices.csv")

    # 1. 按任务2的规则选匹配字段，接入第09题并保存实际匹配结果。
    matching_inputs = select_inputs("共同设备匹配")
    intersection, common_ids, matching_method, matching_ready = match_devices(
        matching_inputs["cn_devices.csv"], matching_inputs["de_devices.csv"]
    )
    if matching_method == "NONE":
        matching_details = "尚未接入匹配方法；当前仅保留本地明文交集基线"
        matching_status = "BASELINE_ONLY"
        matching_check = "尚未实现"
    elif matching_ready:
        matching_details = "实际共同设备匹配结果"
        matching_status = "RECORDED"
        matching_check = "通过"
    else:
        matching_details = "实际共同设备匹配未通过令牌及来源核对"
        matching_status = "CHECK_FAILED"
        matching_check = "未通过"
    matching = save_and_register_result(
        "共同设备匹配", intersection, matching_method, matching_ready,
        matching_details, status=matching_status,
    )
    results = [matching]
    save_analysis_check(
        "共同设备匹配", len(set(cn.device_id) & set(de.device_id)),
        len(common_ids), matching_check,
        "以真实令牌交集及本地编号映射核对，明文仅供课堂校验",
    )
    if matching_method != "NONE" and (matching_method != "PSI" or not matching_ready):
        save_results(4, results)
        print("匹配未通过，本次不生成联合统计和模型。")
        return
    match_ready = matching_method == "PSI" and matching["ready"]

    # 2. 第09-B题核对共同记录并确定归属；每台共同设备只保留在德维一方。
    cn_owned, de_owned, ownership_ready = assign_records(cn, de, common_ids)
    cn_ownership = cn_owned[["device_id"]].assign(owner="CN")
    de_ownership = de_owned[["device_id"]].assign(owner="DE")
    ownership = pd.concat([cn_ownership, de_ownership], ignore_index=True)
    ownership["is_common"] = ownership.device_id.isin(common_ids)
    ownership["split"] = "NOT_TRAINED"
    write_csv(4, "设备划分.csv", ownership)
    if ownership_ready:
        ownership_status = "通过"
        ownership_reason = "共同设备只归德维，华辰非共同记录已核对"
    else:
        ownership_status = "尚未实现"
        ownership_reason = "未填筛选，使用框架归属基线，待填步骤仍未完成"
    save_analysis_check(
        "归属去重", len(set(cn.device_id) | set(de.device_id)), len(ownership),
        ownership_status, ownership_reason,
    )

    # 3. 第10题处理双方告警贡献，保护求和、核对，再准备带噪统计。
    alarm_inputs = select_inputs(
        "联合告警统计", {"cn_devices.csv": cn_owned, "de_devices.csv": de_owned}
    )
    alarm_result = aggregate_alarms(
        alarm_inputs["cn_devices.csv"], alarm_inputs["de_devices.csv"]
    )
    if not match_ready or not ownership_ready:
        alarm_result["ready"] = False
        alarm_result["details"] += "；前置匹配、规则或归属筛选尚未完成"
        if alarm_result["status"] == "RECORDED":
            alarm_result["method"] = "NONE"
            alarm_result["status"] = "BASELINE_ONLY"
    alarms = save_and_register_result(
        "联合告警统计", alarm_result["frame"], alarm_result["method"],
        alarm_result["ready"], alarm_result["details"], dependencies=[matching],
        protection_config=alarm_result["protection_config"], status=alarm_result["status"],
    )
    results.append(alarms)

    # 4. 模型输入单独按规则选择。按设备编号稳定排序，再划分同一训练/测试集。
    model_inputs = select_inputs(
        "联合故障模型", {"cn_devices.csv": cn_owned, "de_devices.csv": de_owned}
    )
    training = prepare_training(
        model_inputs["cn_devices.csv"], model_inputs["de_devices.csv"], ownership
    )
    write_csv(4, "设备划分.csv", training["split"])

    # 5. 第11题在每轮联邦训练中聚合参数；用同一划分比较四个模型。
    model_result = train_models(training, aggregate_local_models)
    write_csv(4, "模型比较.csv", model_result["comparisons"])
    if model_result["status"] == "QUALITY_PENDING":
        print(model_result["details"])
    if not match_ready or not ownership_ready:
        model_result["ready"] = False
        model_result["details"] += "；前置匹配、规则或归属筛选尚未完成"
        if model_result["status"] == "RECORDED":
            model_result["method"] = "NONE"
            model_result["status"] = "BASELINE_ONLY"
    model = save_and_register_result(
        "联合故障模型", model_result["frame"], model_result["method"],
        model_result["ready"], model_result["details"], dependencies=[matching],
        protection_config=model_result["protection_config"], status=model_result["status"],
    )
    results.append(model)
    save_results(4, results)
    if match_ready and ownership_ready:
        scope = "保护匹配与归属已接入"
    else:
        scope = "本地基线（保护匹配、规则或归属筛选尚未完成）"
    print(f"{scope}：共同设备{len(common_ids)}台；华辰{len(cn_owned)}台、德维{len(de_owned)}台，各设备参与一次。")


def match_devices(cn, de):
    """核对真实令牌交集与本地编号，返回交集表、共同编号、方法和完成状态。"""
    # TODO09【选法与方案设计】：接入保护交集并完成共同设备归属筛选；单编号令牌已提供。
    # 处理对象：任务1清洗后的华辰cn_devices.csv（cn）与德维de_devices.csv（de）。
    # 09-A按device_id求交集并生成结果文件/共同设备匹配.csv；09-B按共同编号分配后续参与记录。
    # 可选hmac_psi.py的哈希集合求交，或sorted_token_intersection.py的排序归并求交，完成一条即可。
    # 两者沿用给定HMAC-SHA256协议、DEMO_TOKEN_KEY与MATCH_TOKEN_DOMAIN。
    # 输入双方编号；返回64个小写十六进制字符的令牌及其交集，不返回明文编号。
    # None表示未接入，真实空交集返回[]；框架逐个核对令牌和原编号，不只核对数量。
    # 09-A在本函数填写，09-B在下方assign_records填写；完成后查看匹配与统计.csv及设备划分.csv。
    def token_for(device_id):
        """按给定匹配协议生成单编号令牌，供本地映射核对。"""
        # 已提供：64个十六进制字符，使用与交集函数一致的密钥和用途域。
        return tokenize(device_id, DEMO_TOKEN_KEY, MATCH_TOKEN_DOMAIN)

    def protected_intersection(cn_ids, de_ids):
        """输入两个编号列表，返回保护交集令牌列表；完成后无交集应返回[]。"""
        # 提示：从两个候选中选一个交集函数，两者参数顺序相同；密钥和用途域必须与token_for一致。
        # 选择排序方案时需实际调用排序归并函数；不能只更改名称、返回明文集合或预设答案。
        # 返回令牌列表；真实空交集返回[]，既不能返回交集数量，也不能返回明文编号。
        # 【待填 09-A】替换为保护交集调用；None保留尚未接入状态。
        # ========== 作答区 09-A 开始 ==========
        from 可选算法.hmac_psi import hmac_intersection

        return hmac_intersection(cn_ids, de_ids, DEMO_TOKEN_KEY, MATCH_TOKEN_DOMAIN)
        # ========== 作答区 09-A 结束 ==========


    expected = set(cn.device_id) & set(
        de.device_id
    )  # 仅用于课堂校验或未实现时的明文基线。
    protected = protected_intersection(cn.device_id.tolist(), de.device_id.tolist())
    if protected is None:
        common = sorted(expected)
        return pd.DataFrame({"device_id": common}), common, "NONE", False
    left = {token_for(value): value for value in cn.device_id}
    right = {token_for(value): value for value in de.device_id}
    if None in left or None in right:
        common = sorted(expected)
        return pd.DataFrame({"device_id": common}), common, "NONE", False
    tokens = list(protected)
    common = sorted(
        left[token]
        for token in set(tokens)
        if token in left and token in right and left[token] == right[token]
    )
    format_ok = (
        pd.Series(tokens, dtype="string")
        .str.fullmatch(r"[0-9a-f]{64}")
        .fillna(False)
        .all()
    )
    protocol_ok = all(
        token == tokenize(value, DEMO_TOKEN_KEY, MATCH_TOKEN_DOMAIN)
        for mapping in (left, right) for token, value in mapping.items()
    )
    ready = (
        format_ok
        and protocol_ok
        and len(tokens) == len(set(tokens)) == len(common)
        and set(tokens) == left.keys() & right.keys()
        and set(common) == expected
    )
    return pd.DataFrame({"device_token": tokens}), common, "PSI", bool(ready)


def assign_records(cn, de, common_ids):
    """固定业务约定：同一期的共同记录只由德维参与汇总和训练。"""
    if cn.device_id.duplicated().any() or de.device_id.duplicated().any():
        raise ValueError("请先在任务1处理重复设备")
    # 已提供：保留来源快照，避免原地修改后再校验时丢失真实参照。
    source_cn, source_de = cn.copy(deep=True), de.copy(deep=True)
    source_common = list(common_ids)
    # 【待填 09-B】先按device_id对齐两方共同记录，得到一致性bool records_match，再筛华辰非共同记录cn_owned。
    # 共同记录可能顺序不同；字段和值须一致。共同设备按课堂约定仅由德维贡献。
    # 保留非共同记录的索引、字段和值，不修改cn/de/common_ids；可查set_index、sort_index、equals、isin。
    # ========== 作答区 09-B 开始 ==========
    cn_common = cn.loc[cn.device_id.isin(common_ids)].set_index("device_id").sort_index()
    de_common = de.loc[de.device_id.isin(common_ids)].set_index("device_id").sort_index()
    records_match = cn_common.equals(de_common)
    cn_owned = cn.loc[~cn.device_id.isin(common_ids)].copy()
    # ========== 作答区 09-B 结束 ==========

    # 已提供：独立核对完整记录；未填写时提供归属基线，但不把筛选标为已完成。
    if not cn.equals(source_cn) or not de.equals(source_de) or list(common_ids) != source_common:
        raise ValueError("归属筛选不能修改来源表或共同编号")
    # 框架独立检查来源冲突；未填写不能静默接受冲突记录。
    validate_common_sources(source_cn, source_de, source_common)
    ready = records_match is not None and cn_owned is not None
    if records_match is not None and (not isinstance(records_match, (bool, np.bool_)) or not records_match):
        raise ValueError("09-B共同记录一致性判断错误，请按设备编号对齐后比较")
    cn_owned = check_owned_records(cn_owned, source_cn, source_common)
    return cn_owned.copy(), de.copy(), ready



def aggregate_alarms(cn, de):
    """保护求和并加噪，返回统计表、方法、状态、说明和保护配置组成的字典。"""
    # TODO10【核心填空与接入】：处理两方告警贡献，执行保护求和并核对重构值。
    # 处理对象：上述两张设备清洗表经09-B归属筛选后的alarm_count列（cn/de），共同设备仅由德维计入。
    # 10-A生成逐设备贡献，10-B对双方贡献执行协议；结果写入本任务结果文件/联合告警统计.csv。
    # 两处空位分别负责贡献处理与协议调用；重构核对、敏感度、预算与加噪发布已提供。
    # 未填写时输出本地精确基线；已填写的错误结果会被独立核验并拒绝。
    def prepare_contributions(values):
        """输入一方告警Series，按给定贡献范围处理，返回保持行顺序的整数列表；None表示未接入。"""
        # 【待填 10-A】将每条告警贡献限制在[0,8]并转成整数列表，两方共用此函数。
        # 输入values是已治理的告警Series；可查clip、astype、tolist，不改变原Series或记录顺序。
        # ========== 作答区 10-A 开始 ==========
        return values.clip(0, 8).astype(int).tolist()
        # ========== 作答区 10-A 结束 ==========

    def secure_sum(left_values, right_values):
        """输入两方已经去重并裁剪的整数列表，返回保护协议实际重构的整数合计。"""
        # 【待填 10-B】接入secret_sharing_sum的保护求和函数并返回重构整数，不能用明文sum代替协议。
        # 可用固定seed复现课堂份额演示；种子用途与边界写进报告，固定种子不构成生产安全保证。
        # ========== 作答区 10-B 开始 ==========
        from 可选算法.secret_sharing_sum import two_party_secure_sum

        return two_party_secure_sum(left_values, right_values, seed=510)
        # ========== 作答区 10-B 结束 ==========

    # 已提供：根据来源独立重建贡献，只接受同顺序、同长度且内容正确的整数列表。
    expected_left = cn.alarm_count.clip(0, 8).astype(int).tolist()
    expected_right = de.alarm_count.clip(0, 8).astype(int).tolist()
    left = prepare_contributions(cn.alarm_count.copy())
    right = prepare_contributions(de.alarm_count.copy())
    contributions_ready = left is not None and right is not None
    for actual, expected in [(left, expected_left), (right, expected_right)]:
        if actual is not None and (
            not isinstance(actual, list)
            or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in actual)
            or actual != expected
        ):
            raise ValueError("告警贡献须按来源顺序裁剪到[0,8]并转成整数列表，不得丢行、重排或改写贡献")
    left = expected_left if left is None else left
    right = expected_right if right is None else right
    sensitivity, epsilon = 8, 1
    truth = sum(expected_left) + sum(expected_right)
    secured = secure_sum(left, right)
    # 已提供：无协议输出时保留未完成；有输出时核对重构值，不把加噪后的结果用于精确比较。
    reconstruction_ok = None if secured is None else bool(secured == truth)
    if reconstruction_ok is not None and not isinstance(reconstruction_ok, (bool, np.bool_)):
        raise ValueError("重构一致性判断须为bool或尚未接入时的None")
    actual_equal = (
        isinstance(secured, (int, np.integer)) and not isinstance(secured, (bool, np.bool_))
        and secured == truth
    )
    equal = None if secured is None or reconstruction_ok is None else bool(reconstruction_ok and actual_equal)
    save_analysis_check("告警贡献处理", len(expected_left) + len(expected_right), len(left) + len(right),
                        "通过" if contributions_ready else "尚未实现",
                        "两方共用裁剪函数，逐条贡献已核对" if contributions_ready else "贡献回调未填，使用框架基线")
    save_analysis_check("华辰告警小计", sum(left), sum(left), "本地计算", "每设备告警贡献裁剪到[0,8]")
    save_analysis_check("德维告警小计", sum(right), sum(right), "本地计算", "每设备告警贡献裁剪到[0,8]")
    save_analysis_check("联合告警保护求和", truth, secured,
                        "尚未实现" if secured is None else "通过" if actual_equal else "未通过",
                        "独立核对协议实际合计；精确值仅供本地课堂校验")
    save_analysis_check("联合告警重构核对", truth, secured,
                        "尚未实现" if equal is None else "通过" if equal else "未通过",
                        "框架根据实际协议输出自动核对，精确重构与最终带噪发布分别记录")
    if secured is not None and (not actual_equal or (reconstruction_ok is not None and not reconstruction_ok)):
        return {
            "frame": pd.DataFrame(columns=["query", "released_value"]),
            "method": "SECURE_SUM_DP", "ready": False, "status": "CHECK_FAILED",
            "details": "协议重构结果或一致性判断不符合课堂校验，不能准备带噪统计",
            "protection_config": {},
        }
    if not contributions_ready or secured is None or reconstruction_ok is None:
        return {
            "frame": pd.DataFrame([{"query": "fleet_alarm_sum", "released_value": truth}]),
            "method": "NONE", "ready": False, "status": "BASELINE_ONLY",
            "details": "裁剪后的精确基线；贡献处理或保护求和尚未完成；重构核对由框架自动执行",
            "protection_config": {},
        }
    value = laplace_release(secured, sensitivity, epsilon)
    frame = pd.DataFrame(
        [
            {
                "query": "fleet_alarm_sum",
                "released_value": value,
                "mechanism": "LAPLACE",
                "epsilon": epsilon,
                "delta": 0,
                "privacy_unit": "DEVICE",
                "adjacency": "REPLACE_ONE_FIXED_PUBLIC_N",
                "public_n": len(left) + len(right),
                "contribution_limit": 8,
                "records_per_unit": 1,
            }
        ]
    )
    protection_config = {
        "fleet_alarm_sum": {
            "sensitivity": float(sensitivity),
            "epsilon": float(epsilon),
            "delta": 0,
            "mechanism": "LAPLACE",
        }
    }
    ready = sensitivity == 8 and 0 < epsilon <= 1 and np.isfinite(value)
    return {
        "frame": frame, "method": "SECURE_SUM_DP", "ready": bool(ready),
        "status": "RECORDED" if ready else "CHECK_FAILED",
        "details": "先检查保护合计，再发布带噪结果", "protection_config": protection_config,
    }


def aggregate_local_models(local_models):
    """输入本轮非空(size, weights, bias)列表，返回(全局权重, 偏置)；None表示未完成。"""
    # TODO11【模型计算】：按各方本轮训练样本数聚合权重向量和偏置。
    # 输入已核验：size为正整数，weights为同维有限向量，bias为有限标量。
    # 各方影响按size占总样本量的比例确定；不能直接平均客户端，也不能用测试集样本数。
    # 本函数由FedAvg每轮实际调用。本地更新、输入校验和模型比较已提供。
    # 观察模型比较.csv与联合故障模型参数；未完成时FedAvg不生成参数，其他模型仍可比较。
    # 【待填 11-A】按训练样本量分别聚合权重与偏置，返回二元组。
    # ========== 作答区 11-A 开始 ==========
    total_size = sum(size for size, _, _ in local_models)
    weights = sum(size * values for size, values, _ in local_models) / total_size
    bias = sum(size * value for size, _, value in local_models) / total_size
    return weights, bias
    # ========== 作答区 11-A 结束 ==========

if __name__ == "__main__":
    main()
