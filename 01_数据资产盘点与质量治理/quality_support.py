"""任务1已提供的清洗、字段目录和证据处理。
接收主流程明确传入的表、字段或方法，返回处理值与证据；本模块不导入任务入口或保存文件。
"""

import re

import pandas as pd
import numpy as np
from 可选算法.field_classification import classify_field
from 可选算法.field_tags import make_tags
from 可选算法.fixed_value_repair import numeric_copy
from 可选算法.group_median_repair import build_group_reference
from utils import ASSETS


def collect_quality_changes(raw, cleaned):
    """比较原始表与处理表字典，返回逐字段修改记录DataFrame。
    
    按原始索引定位重复行和修改，注明删除、估计、置空或规范化原因；
    不修改输入、不保存文件，主流程负责保存证据。
    """
    rows = []
    for asset, original in raw.items():
        for source_row in original.index[original.duplicated()]:
            rows.append([asset, int(source_row), "ROW", "DUPLICATE", "REMOVED"])
        before = original.drop_duplicates()
        after = cleaned[asset]
        for field in after.columns:
            old_values = before[field].astype("string").fillna("<MISSING>")
            new_values = after[field].astype("string").fillna("<MISSING>")
            changed = new_values.ne(old_values)
            for source_row in after.index[changed]:
                rows.append([asset, int(source_row), field,
                             before.at[source_row, field], after.at[source_row, field]])
    changes = pd.DataFrame(rows, columns=["asset", "source_row", "field", "before", "after"])
    treatments = []
    for row in changes.itertuples():
        if row.field == "ROW":
            treatment = "DUPLICATE_REMOVED"
        elif row.field in {"temperature_c", "operating_hours", "downtime_hours"}:
            treatment = "ESTIMATED"
        elif pd.isna(row.after):
            treatment = "INVALID_REMOVED"
        else:
            treatment = "NORMALIZED"
        treatments.append(treatment)
    changes["treatment"] = treatments
    changes["reason"] = changes.treatment.map({
        "DUPLICATE_REMOVED": "去除完全相同的重复记录",
        "ESTIMATED": "用于维持计算的估计值，需复核；不能解释为可靠观测",
        "INVALID_REMOVED": "无法确定正确值，置空等待核实",
        "NORMALIZED": "规范已有内容的表示格式",
    })
    return changes


def check_case_dates(cases):
    """已提供：返回关闭日期已核对的副本；不处理联系信息、邮编或停机时长。

    已填写但无法解析或早于创建日期的关闭日期置空，允许原本为空的关闭日期。
    """
    cases = cases.copy()
    opened = pd.to_datetime(cases.opened_date, errors="coerce")
    closed = pd.to_datetime(cases.closed_date, errors="coerce")
    invalid = cases.closed_date.notna() & (closed.isna() | (closed < opened))
    cases.loc[invalid, "closed_date"] = pd.NA
    return cases


def contact_format_issue(field, value, country):
    """核对当前值是否仍有格式问题，返回原因或None；只检查，不替学生清洗。"""
    if pd.isna(value):
        return None  # 缺失值由统一质量记录处理。
    text = str(value)
    if field in {"contact_name", "engineer_name"}:
        if not text.strip():
            return "姓名为空白，需核实；不能编造姓名"
        if text != text.strip():
            return "姓名仍有首尾空白，格式治理尚待完成"
    elif field in {"contact_email", "engineer_email"}:
        if text != text.strip().lower() or re.fullmatch(r"[^@\s]+@[^@\s]+\.invalid", text) is None:
            return "邮箱尚未规范或格式无效，需处理或核实"
    elif field in {"contact_phone", "engineer_phone"}:
        if re.fullmatch(r"\+[1-9][0-9]{7,14}", text) is None:
            return "电话尚未规范或格式无效，需处理或核实"
    elif field == "site_postcode":
        pattern = {"CN": r"[0-9]{6}", "DE": r"[0-9]{5}"}.get(country)
        if pattern is None or re.fullmatch(pattern, text) is None:
            return "邮编未满足站点国家的格式要求，需处理或核实"
    return None


def prepare_quality_references(cn, de, cases):
    """从三张基础规范化表生成共同参考，返回按两个时长字段命名的统计字典。
    
    输入为去重后的华辰设备、德维设备和案例快照；设备按德维归属去重，
    运行时长按型号分组，停机时长按案例所在国家分组，只纳入合法观测。
    小组不足3条时使用共同全局中位数，全局为空时分别回退到360小时和0小时。
    编号或共同设备记录冲突时抛出异常；不使用故障标签、不修复输入或保存文件。
    """
    for name, frame in (("cn_devices.csv", cn), ("de_devices.csv", de)):
        if frame.device_id.isna().any() or frame.device_id.duplicated().any():
            raise ValueError(f"{name} 存在缺失或冲突设备编号，请先核对")
    common = sorted(set(cn.device_id) & set(de.device_id))
    left = cn.set_index("device_id").loc[common].sort_index()
    right = de.set_index("device_id").loc[common].sort_index()
    if not left.equals(right):
        raise ValueError("共同设备的基础记录冲突，不能用分组估计掩盖冲突")
    fleet = pd.concat([cn[~cn.device_id.isin(common)], de], ignore_index=True)
    if cases.case_id.isna().any() or cases.case_id.duplicated().any():
        raise ValueError("服务案例编号缺失或冲突，不能重复计入参考样本")
    # 运行时长的统计单位是设备，共同设备只采用德维的一份观测。
    hours_reference = build_group_reference(
        fleet["operating_hours"], fleet["device_model"],
        lower=0, upper=744, fallback=360, min_samples=3,
    )
    hours_reference["group_field"] = "device_model"
    # 停机时长的统计单位是案例，不能与设备运行小时混合估计。
    downtime_reference = build_group_reference(
        cases["downtime_hours"], cases["site_country"],
        lower=0, upper=None, fallback=0, min_samples=3,
    )
    downtime_reference["group_field"] = "site_country"
    return {"operating_hours": hours_reference, "downtime_hours": downtime_reference}



def repair_numeric_values(original, groups, reference, repair):
    """调用所选方法修复时长，返回保持原索引和行顺序的数值Series。
    
    输入原时长列、分组列、共同参考与修复方法；复制数值和分组后调用方法。
    repair=None时直接返回原值副本；已接入时校验所有结果有限、符合参考范围，
    并保留原本合法的值，否则抛出异常。字段写回和文件保存由主流程负责。
    """
    if repair is None:
        return original.copy()
    if not callable(repair):
        raise TypeError("修复方法必须为回调函数或None")
    result = repair(original.copy(), groups.copy(), reference)
    if not isinstance(result, pd.Series) or not result.index.equals(original.index):
        raise ValueError("修复结果必须为保留原索引及顺序的Series")
    values = numeric_copy(result)
    before = numeric_copy(original)
    lower, upper = reference["lower"], reference["upper"]
    valid_before = np.isfinite(before) & before.ge(lower)
    valid_after = np.isfinite(values) & values.ge(lower)
    if upper is not None:
        valid_before &= before.le(upper)
        valid_after &= values.le(upper)
    if not valid_after.all() or not values.loc[valid_before].eq(before.loc[valid_before]).all():
        raise ValueError("修复须保留合法原值，且所有结果有限并满足公开质量范围")
    return values


def build_quality_evidence(raw, cleaned, changes):
    """比较治理效果并生成复核线索，返回比较表、待复核表两个DataFrame。
    
    输入原始表字典、清洗表字典和修改记录。两条比较路径分别使用当前处理值
    和排除原始异常后的值，记录各自分母；停机均值统一按[0,72]裁剪统计贡献，
    不改变清洗表中的合法长停机。函数只生成证据，不修改输入或保存文件。
    """
    comparisons, pending = [], []
    bounds = {
        "temperature_c": (-10, 45),
        "operating_hours": (0, 744),
        "downtime_hours": (0, np.inf),  # 合法长停机不属于质量异常。
    }
    for asset, original in raw.items():
        original = original.drop_duplicates()
        current = cleaned[asset]
        for field, (lower, upper) in bounds.items():
            if field not in original:
                continue
            original_values = pd.to_numeric(original[field], errors="coerce")
            current_values = pd.to_numeric(current[field], errors="coerce")
            valid = np.isfinite(original_values) & original_values.between(lower, upper)
            # 当前方法保留治理后的有限数值；对照方法排除原始质量异常。
            # 两者可能使用不同数量的记录，不能仅把均值差解释为修正幅度。
            used = current_values[np.isfinite(current_values)]
            excluded = original_values[valid]
            current_clipped, excluded_clipped = 0, 0
            if field == "downtime_hours":
                # 72小时仅限制统计贡献，两种方法使用同一规则，原表不截断。
                current_clipped = int((~used.between(0, 72)).sum())
                excluded_clipped = int((~excluded.between(0, 72)).sum())
                used, excluded = used.clip(0, 72), excluded.clip(0, 72)
            comparisons.append(
                {
                    "asset": asset,
                    "field": field,
                    "current_mean": float(used.mean()),
                    "exclude_invalid_mean": float(excluded.mean()),
                    "current_rows": len(used),
                    "exclude_invalid_rows": len(excluded),
                    "affected_rows": int((~valid).sum()),
                    "current_clipped_rows": current_clipped,
                    "exclude_invalid_clipped_rows": excluded_clipped,
                    "contribution_rule": (
                        "停机贡献裁剪到[0,72]；原表保留合法长停机"
                        if field == "downtime_hours"
                        else "不作统计裁剪"
                    ),
                    "downstream_use": (
                        "任务3停机均值"
                        if field == "downtime_hours"
                        else "任务4模型特征"
                    ),
                }
            )
        for position, source_row in enumerate(original.index):
            for field in current:
                # 数值治理尚未接入时，原值仍可运行，但不应从质量记录中消失。
                if field in bounds:
                    value = pd.to_numeric(current.iloc[position][field], errors="coerce")
                    lower, upper = bounds[field]
                    if not np.isfinite(value) or not lower <= value <= upper:
                        pending.append(
                            {
                                "asset": asset,
                                "source_row": int(source_row),
                                "field": field,
                                "reason": "当前数值不满足公开质量范围，数值治理尚待完成",
                            }
                        )
                format_reason = contact_format_issue(
                    field, current.iloc[position][field], current.iloc[position].get("site_country")
                )
                if format_reason:
                    pending.append(
                        {"asset": asset, "source_row": int(source_row),
                         "field": field, "reason": format_reason}
                    )
                if field != "closed_date" and pd.isna(current.iloc[position][field]):
                    pending.append(
                        {
                            "asset": asset,
                            "source_row": int(source_row),
                            "field": field,
                            "reason": "值缺失或已剔除，需核实；不编造联系资料",
                        }
                    )
    for row in changes.itertuples():
        if row.treatment == "ESTIMATED" or (
            row.field == "closed_date" and pd.isna(row.after)
        ):
            pending.append(
                {
                    "asset": row.asset,
                    "source_row": row.source_row,
                    "field": row.field,
                    "reason": row.reason,
                }
            )
    comparison_table = pd.DataFrame(comparisons)
    pending_table = pd.DataFrame(pending, columns=["asset", "source_row", "field", "reason"])
    return comparison_table, pending_table


def review_text_field(findings, category, level, basis):
    """已提供：依据备注扫描复核字段分级；未扫描仍返回None，不代表没有风险。"""
    if findings.empty or findings.review.eq("UNASSESSED").any():
        return None
    found_clues = set()
    for value in findings.clues:
        for clue in str(value).split("|"):
            if clue and clue not in {"NOT_SCANNED", "nan", "<NA>"}:
                found_clues.add(clue)
    clues = sorted(found_clues)
    if clues:
        return "标识", "L4", "内容命中：" + "|".join(clues)
    # 未命中不降低原有保护级别；规则只能覆盖已定义的格式。
    return category, level, "已扫描未命中；" + basis


def classify_fields(frames, text_findings):
    # 已提供：基础字段规则、完整56行目录和标签转换。
    # 文本字段的复核由本文件已提供的 review_text_field 完成。
    """接收业务表与扫描线索，使用已提供的复核逻辑返回每个字段的分级目录DataFrame。
    
    先按字段规则分级，再调用review_text_field复核有文本扫描记录的字段；
    尚未完成的复核状态如实保留。不修改输入表或保存文件。
    """
    exact = {
        **{
            f: ("标识", "L4")
            for f in ["device_id", "case_id", "contact_name", "contact_email"]
        },
        **{
            f: ("业务结果", "L3")
            for f in [
                "alarm_count",
                "failure_next_90d",
                "confirmed_failure",
                "downtime_hours",
            ]
        },
        **{
            f: ("组合线索", "L2")
            for f in ["device_model", "site_region", "opened_date", "closed_date"]
        },
    }
    exact.update(
        {
            field: ("标识", "L4")
            for field in [
                "contact_phone",
                "contact_employee_id",
                "engineer_name",
                "engineer_email",
                "engineer_phone",
                "site_address",
            ]
        }
    )
    exact.update(
        {
            field: ("组合线索", "L2")
            for field in [
                "site_country",
                "site_city",
                "site_postcode",
                "contact_department",
                "contact_job_title",
            ]
        }
    )
    exact.update(
        {
            field: ("业务结果", "L3")
            for field in ["service_contract_no", "service_note"]
        }
    )
    keywords = [(("temperature", "operating"), ("运行指标", "L3"))]
    tags = {
        "标识": "IDENTIFIER",
        "业务结果": "SENSITIVE",
        "组合线索": "QUASI_IDENTIFIER",
        "登记信息": "METADATA",
        "运行指标": "FEATURE",
    }
    rows = []
    for name, frame in frames.items():
        for field in frame:
            category, level = classify_field(field, exact, keywords, ("待复核", "L4"))
            matches = text_findings[
                (text_findings.asset == name) & (text_findings.field == field)
            ]
            basis = (
                "精确字段规则"
                if field in exact
                else "默认复核" if category == "待复核" else "关键词规则"
            )
            if len(matches):
                decision = review_text_field(matches, category, level, basis)
                category, level, basis = (
                    decision if decision is not None
                    else ("UNASSESSED", "UNASSESSED", "文本内容分级尚未完成")
                )
            tag, _ = make_tags(category, level, tags, {"L3", "L4"})
            if len(matches):
                tag += "|FREE_TEXT"
            rows.append(
                {
                    "asset": name,
                    "field": field,
                    "category": category,
                    "level": level,
                    "tags": tag,
                    "basis": basis,
                }
            )
    return pd.DataFrame(rows)


def review_assets(frames, catalog):
    # 按本次字段分级自动生成三张表的盘点概览。
    # 每张表一行；未完成分级时保留未核对状态，详细依据查看字段分级.csv。
    """汇总清洗表字典与字段目录，返回每张表一行的资产核对DataFrame。
    
    按实际字段判断分级是否完成，并标明持有方；不修改输入或保存文件。
    """
    rows = []
    for name, frame in frames.items():
        fields = catalog[catalog.asset == name]
        classified = (
            len(fields) == len(frame.columns)
            and set(fields.field) == set(frame.columns)
            and fields.level.isin(["L1", "L2", "L3", "L4"]).all()
        )
        rows.append(
            {
                "asset_id": name,
                "kind": "CSV",
                "owner": ASSETS[name][1],
                "clues": "见字段分级.csv",
                "review": "FIELD_REVIEW" if classified else "UNASSESSED",
            }
        )
    return pd.DataFrame(rows)


def build_quality_records(raw, changes, pending):
    """合并修改与复核依据，返回按来源、原始行号和字段排序的质量记录DataFrame。
    
    输入原始表字典、修改记录与待复核记录；同一行同一字段合成一条记录，
    通过原始表补齐业务编号及未修改原值。不修改输入、不保存文件。
    """
    keys = ["asset", "source_row", "field"]
    review = pending.groupby(keys, as_index=False).reason.agg(
        lambda values: "；".join(dict.fromkeys(values))
    )
    review = review.rename(columns={"reason": "review_reason"})
    merged = changes.merge(review, on=keys, how="outer", validate="one_to_one")
    merged["needs_review"] = merged.review_reason.notna()
    id_fields = {
        "cn_devices.csv": "device_id",
        "de_devices.csv": "device_id",
        "service_cases.csv": "case_id",
    }
    merged["record_id"] = [
        raw[row.asset].loc[row.source_row, id_fields[row.asset]]
        for row in merged.itertuples()
    ]
    for index in merged.index[merged.treatment.isna()]:
        row = merged.loc[index]
        value = raw[row.asset].loc[row.source_row, row.field]
        merged.loc[index, ["before", "after", "treatment", "reason"]] = [
            value,
            value,
            "UNCHANGED",
            "尚未修改，保持原值；具体问题见复核原因",
        ]
    return merged.sort_values(keys).reset_index(drop=True)
