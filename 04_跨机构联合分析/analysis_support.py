"""任务4已提供的训练数据准备、模型比较和检查表保存。

匹配、归属、保护求和及模型聚合的作答保留在main.py。
训练数据准备与模型训练返回结果，由main.py明确保存设备划分和模型比较表。
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from 可选算法.fedavg_logistic import fedavg_train, AggregationPending
from utils import FEATURES, read_table, task_dir, write_csv, numeric_quality_issue, TABLE_COLUMNS


# 顺序与FEATURES一致；训练和导出共用固定尺度，不从测试集估计参数。
FEATURE_CENTER = np.array([20.0, 360.0, 3.0])
FEATURE_SCALE = np.array([6.0, 200.0, 3.0])


def prepare_training(cn, de, ownership):
    """核对数值质量并按设备编号稳定划分，返回训练数据和划分表，不读写文件。

    每台设备只属于一方。质量未完成时保留NOT_TRAINED，不转换非法特征。
    """
    cn = cn.sort_values("device_id").copy()
    de = de.sort_values("device_id").copy()
    fleet = pd.concat([cn.assign(owner="CN"), de.assign(owner="DE")], ignore_index=True)
    fleet = fleet.sort_values("device_id").reset_index(drop=True)
    if fleet.device_id.duplicated().any():
        raise ValueError("训练数据中有重复设备")
    split = ownership.sort_values("device_id").reset_index(drop=True).copy()
    if not split.device_id.is_unique or set(split.device_id) != set(fleet.device_id):
        raise ValueError("设备划分与本次训练设备不一致")
    split["split"] = "NOT_TRAINED"
    bounds = {"temperature_c": (-10, 45), "operating_hours": (0, 744), "alarm_count": (0, None)}
    issues = []
    cn_issue = numeric_quality_issue(cn, bounds)
    de_issue = numeric_quality_issue(de, bounds)
    if cn_issue:
        issues.append("华辰：" + cn_issue)
    if de_issue:
        issues.append("德维：" + de_issue)
    if issues:
        return {"split": split, "clients": [], "test_x": None, "test_y": None, "quality_issues": issues}

    # 固定种子作用于按编号排列的数据；合法行重排不会改变训练/测试设备。
    train_ids, test_ids = train_test_split(
        fleet.device_id, test_size=0.25, random_state=510,
        stratify=fleet.failure_next_90d,
    )
    split["split"] = np.where(split.device_id.isin(train_ids), "TRAIN", "TEST")
    cn_train = cn.loc[cn.device_id.isin(train_ids)]
    de_train = de.loc[de.device_id.isin(train_ids)]
    cn_x = (cn_train[FEATURES].to_numpy(float) - FEATURE_CENTER) / FEATURE_SCALE
    cn_y = cn_train.failure_next_90d.to_numpy(float)
    de_x = (de_train[FEATURES].to_numpy(float) - FEATURE_CENTER) / FEATURE_SCALE
    de_y = de_train.failure_next_90d.to_numpy(float)
    test = fleet.loc[fleet.device_id.isin(test_ids)]
    return {
        "split": split,
        "clients": {"CN": (cn_x, cn_y), "DE": (de_x, de_y)},
        "test_x": (test[FEATURES].to_numpy(float) - FEATURE_CENTER) / FEATURE_SCALE,
        "test_y": test.failure_next_90d.to_numpy(float),
        "quality_issues": [],
    }


def evaluate_model(mode, scores, labels, training_rows, rounds):
    """在同一测试集计算AUC及前10%设备捕获的故障比例。"""
    selected_count = max(1, int(np.ceil(len(labels) * 0.10)))
    selected = np.argsort(-np.asarray(scores), kind="stable")[:selected_count]
    captured = int(np.asarray(labels)[selected].sum())
    total = int(np.asarray(labels).sum())
    return {
        "mode": mode, "auc": roc_auc_score(labels, scores),
        "training_rows": training_rows, "rounds": rounds, "test_devices": len(labels),
        "selected_devices": len(selected), "captured_failures": captured,
        "total_failures": total,
        "top10_capture_rate": captured / total if total else float("nan"),
    }


def train_models(training, aggregate_models):
    """在已准备的同一划分上比较集中模型、两方本地模型和FedAvg。

    返回模型参数、比较表及状态，由main.py保存；数值未治理时不训练或伪造指标。
    """
    if training["quality_issues"]:
        comparisons = []
        for mode in ["CENTRALIZED", "CN_LOCAL", "DE_LOCAL", "FEDAVG"]:
            comparisons.append({
                "mode": mode, "training_rows": 0, "test_devices": 0, "rounds": 0,
                "status": "QUALITY_PENDING",
            })
        return {
            "frame": pd.DataFrame(columns=["parameter", "value"]),
            "method": "NONE", "ready": False, "status": "QUALITY_PENDING",
            "details": "等待任务1数据治理，未训练模型：" + "；".join(training["quality_issues"]),
            "protection_config": {},
            "comparisons": pd.DataFrame(comparisons, columns=TABLE_COLUMNS["模型比较.csv"]),
        }

    cn_x, cn_y = training["clients"]["CN"]
    de_x, de_y = training["clients"]["DE"]
    test_x, test_y = training["test_x"], training["test_y"]
    all_x = np.vstack([cn_x, de_x])
    all_y = np.concatenate([cn_y, de_y])

    # 三个普通模型的训练对象分别写明，统一在test_x/test_y上评价。
    centralized = LogisticRegression(max_iter=1000, C=np.inf).fit(all_x, all_y)
    cn_local = LogisticRegression(max_iter=1000, C=np.inf).fit(cn_x, cn_y)
    de_local = LogisticRegression(max_iter=1000, C=np.inf).fit(de_x, de_y)
    comparisons = [
        evaluate_model("CENTRALIZED", centralized.predict_proba(test_x)[:, 1], test_y, len(all_y), 0),
        evaluate_model("CN_LOCAL", cn_local.predict_proba(test_x)[:, 1], test_y, len(cn_y), 0),
        evaluate_model("DE_LOCAL", de_local.predict_proba(test_x)[:, 1], test_y, len(de_y), 0),
    ]
    rounds, local_epochs, learning_rate = 50, 8, 0.2
    clients = [(cn_x, cn_y), (de_x, de_y)]
    try:
        weights, bias = fedavg_train(clients, rounds, local_epochs, learning_rate, aggregate_models=aggregate_models)
    except (AggregationPending, ValueError, TypeError) as error:
        pending = isinstance(error, AggregationPending)
        status = "BASELINE_ONLY" if pending else "CHECK_FAILED"
        for row in comparisons:
            row["status"] = "EVALUATED"
        comparisons.append({"mode": "FEDAVG", "status": status, "training_rows": len(all_y),
                            "test_devices": len(test_y), "rounds": 0})
        return {"frame": pd.DataFrame(columns=["parameter", "value"]), "method": "NONE",
                "ready": False, "status": status, "details": str(error), "protection_config": {},
                "comparisons": pd.DataFrame(comparisons, columns=TABLE_COLUMNS["模型比较.csv"])}
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (len(FEATURES),) or not np.isfinite(weights).all() or not np.isfinite(bias):
        raise ValueError("联邦训练应返回三个有限权重与一个有限偏置")
    scores = 1 / (1 + np.exp(-np.clip(test_x @ weights + bias, -30, 30)))
    comparisons.append(evaluate_model("FEDAVG", scores, test_y, len(all_y), rounds))
    for row in comparisons:
        row["status"] = "EVALUATED"

    # 内部使用固定尺度；导出转回原单位，预测为sigmoid(x @ weights_raw + bias_raw)。
    weights_raw = weights / FEATURE_SCALE
    bias_raw = float(bias - FEATURE_CENTER @ weights_raw)
    parameters = pd.DataFrame({"parameter": [*FEATURES, "bias"], "value": [*weights_raw, bias_raw]})
    return {
        "frame": parameters, "method": "FEDAVG", "ready": True, "status": "RECORDED",
        "details": "第11题聚合已逐轮执行并核对；四模型同一测试集，完整完成状态还取决于前置规则与匹配",
        "protection_config": {}, "comparisons": pd.DataFrame(comparisons),
    }


def save_analysis_check(check, baseline_value, actual_value, status, reason):
    """保存一个检查项；匹配、归属和统计共用匹配与统计.csv。"""
    frame = read_table(task_dir(4) / "匹配与统计.csv", missing_ok=True)
    if len(frame):
        frame = frame[frame["check"] != check]
    row = pd.DataFrame([{
        "check": check, "baseline_value": baseline_value, "actual_value": actual_value,
        "status": status, "reason": reason,
    }])
    if len(frame):
        frame = pd.concat([frame, row], ignore_index=True)
    else:
        frame = row
    write_csv(4, "匹配与统计.csv", frame)


def validate_common_sources(cn, de, common_ids):
    """独立复核两方共同来源；发现冲突时报告字段，不代填09-B。"""
    left = cn.set_index("device_id").loc[common_ids].sort_index()
    right = de.set_index("device_id").loc[common_ids].sort_index()
    if not left.equals(right):
        conflicts = []
        for device_id in common_ids:
            left_row, right_row = left.loc[device_id], right.loc[device_id]
            for field in left.columns:
                left_value, right_value = left_row[field], right_row[field]
                same_missing = pd.isna(left_value) and pd.isna(right_value)
                if not same_missing and (pd.isna(left_value) or pd.isna(right_value) or left_value != right_value):
                    conflicts.append(f"{device_id}/{field}：华辰={left_value!r}，德维={right_value!r}")
                if len(conflicts) >= 5:
                    break
            if len(conflicts) >= 5:
                break
        detail = "；".join(conflicts) or "共同记录的字段类型不一致"
        raise ValueError("共同设备的同一期记录冲突，请回查任务1修复；" + detail)


def check_owned_records(cn_owned, source_cn, source_common):
    """独立核对华辰贡献；未填写时返回明确的归属基线，完成状态由main保留。"""
    expected = source_cn.loc[~source_cn.device_id.isin(source_common)].copy()
    if cn_owned is None:
        return expected
    if (not isinstance(cn_owned, pd.DataFrame) or not cn_owned.index.is_unique
            or not cn_owned.sort_index().equals(expected.sort_index())):
        raise ValueError("归属筛选须完整保留华辰非共同记录及其原索引、字段和值；共同设备只归德维")
    return cn_owned
