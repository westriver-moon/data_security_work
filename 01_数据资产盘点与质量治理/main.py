"""任务1：处理三张业务表，完成数值与格式治理，保存备注扫描和字段分级证据。"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from utils import prepare_task_run, raw_path, write_csv, CSV_TEXT_TYPES
from quality_support import (
    check_case_dates, prepare_quality_references,
    repair_numeric_values, collect_quality_changes, build_quality_evidence,
    build_quality_records, classify_fields, review_assets,
)

# 从main()按表阅读；01—03题的6处作答框均在本文件。
# 候选位于本任务的可选算法目录；选定后在对应作答框中导入实际使用的方法。
# 其他None是已提供的运行状态。


def main():
    """依次规范三张表、建立共同参考、修复时长，再扫描备注并保存质量证据。"""
    # 直接启动时先执行必要前序步骤；返回后使用本次运行的输出目录。
    prepare_task_run(1)

    cn_raw = pd.read_csv(raw_path("cn_devices.csv"), dtype=CSV_TEXT_TYPES)
    de_raw = pd.read_csv(raw_path("de_devices.csv"), dtype=CSV_TEXT_TYPES)
    cases_raw = pd.read_csv(raw_path("service_cases.csv"), dtype=CSV_TEXT_TYPES)

    # 分别去重，保留当前CSV来源索引；打乱后仍以业务编号关联。
    cn_base = cn_raw.drop_duplicates().copy()
    de_base = de_raw.drop_duplicates().copy()
    cases_base = cases_raw.drop_duplicates().copy()

    # TODO01【核心填空与接入】：规范三张表中已有的联系信息，并按站点国家检查邮编。
    # 处理对象：华辰cn_devices.csv、德维de_devices.csv和service_cases.csv的去重副本。
    # 01-A处理表中已有的contact_*或engineer_*姓名、邮箱、电话列，01-B处理site_postcode。
    # frame依次指向去重后的两张设备表和案例表；在这些副本上修改，保留索引、行和其他字段。
    # 本题首先规范三张表，随后由TODO02修复时长、TODO03扫描备注。
    for frame in (cn_base, de_base, cases_base):
        # 【待填 01-A】姓名只去首尾空白；邮箱去首尾空白并转小写；电话去空白、括号和连字符。
        # 姓名列：contact_name、engineer_name；邮箱列：contact_email、engineer_email；
        # 电话列：contact_phone、engineer_phone。只处理当前表实际存在的列，不改service_note。
        # 邮箱须完整符合本课.invalid合成域格式；电话须为+和8—15位数字，首位非0。
        # 规范后仍非法的邮箱、电话置空待核实；保留缺失值，不编造联系信息。
        # 可用.str.strip()、.str.lower()、.str.replace(..., regex=True)、.str.fullmatch(..., na=False)
        # 和.where(合法条件)；电话保留+，不转整数。把pass换为对frame相应列的处理。
        # ========== 作答区 01-A 开始 ==========
        for column in ("contact_name", "engineer_name"):
            if column in frame:
                frame[column] = frame[column].astype("string").str.strip()
        for column in ("contact_email", "engineer_email"):
            if column in frame:
                normalized = frame[column].astype("string").str.strip().str.lower()
                valid = normalized.str.fullmatch(
                    r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9.-]+\.invalid",
                    na=False,
                )
                frame[column] = normalized.where(valid)
        for column in ("contact_phone", "engineer_phone"):
            if column in frame:
                normalized = frame[column].astype("string").str.replace(
                    r"[\s()\-]", "", regex=True
                )
                valid = normalized.str.fullmatch(r"\+[1-9]\d{7,14}", na=False)
                frame[column] = normalized.where(valid)
        # ========== 作答区 01-A 结束 ==========

        # 【待填 01-B】site_postcode先去首尾空白，再按同一行site_country检查。
        # CN须为6位数字，DE须为5位数字；保留字符串和前导零，非法值置空待核实。
        # 持有方目录不能代替site_country；不能补零或截断来猜测邮编。
        # 可用.astype("string")、.str.fullmatch(..., na=False)，用&、|组合国家与格式条件。
        # ========== 作答区 01-B 开始 ==========
        postcodes = frame["site_postcode"].astype("string").str.strip()
        valid = (
            frame["site_country"].eq("CN") & postcodes.str.fullmatch(r"\d{6}", na=False)
        ) | (
            frame["site_country"].eq("DE") & postcodes.str.fullmatch(r"\d{5}", na=False)
        )
        frame["site_postcode"] = postcodes.where(valid)
        # ========== 作答区 01-B 结束 ==========

    # 已提供：仅核对案例关闭日期，联系信息与邮编由上面的TODO01处理。
    cases_base = check_case_dates(cases_base)
    # 已提供的温度缺失估计，后续在质量处理记录中标注，不能当作真实观测。
    cn_base["temperature_c"] = cn_base["temperature_c"].fillna(20.0)
    de_base["temperature_c"] = de_base["temperature_c"].fillna(20.0)

    # 三张基础表准备完毕后，只建立一次共同参考；此时尚未修复任何时长。
    references = prepare_quality_references(cn_base, de_base, cases_base)
    hours_reference = references["operating_hours"]
    downtime_reference = references["downtime_hours"]
    repair_hours = choose_hours_repair()          # 02-A：两张设备表共用一种方法。
    repair_downtime = choose_downtime_repair()    # 02-B：案例停机单独选择方法。

    cn_clean = cn_base.copy()
    de_clean = de_base.copy()
    cases_clean = cases_base.copy()
    cn_clean["operating_hours"] = repair_numeric_values(
        cn_base["operating_hours"], cn_base["device_model"], hours_reference, repair_hours,
    )
    de_clean["operating_hours"] = repair_numeric_values(
        de_base["operating_hours"], de_base["device_model"], hours_reference, repair_hours,
    )
    cases_clean["downtime_hours"] = repair_numeric_values(
        cases_base["downtime_hours"], cases_base["site_country"], downtime_reference, repair_downtime,
    )

    text_findings = scan_service_notes(cases_clean)    # 03-A/B：原备注只用于读取。
    raw = {"cn_devices.csv": cn_raw, "de_devices.csv": de_raw, "service_cases.csv": cases_raw}
    cleaned = {"cn_devices.csv": cn_clean, "de_devices.csv": de_clean, "service_cases.csv": cases_clean}
    # 已提供：依据实际备注扫描复核字段分级，未扫描仍显示未核对。
    catalog = classify_fields(cleaned, text_findings)
    assets = review_assets(cleaned, catalog)
    changes = collect_quality_changes(raw, cleaned)
    comparison, pending = build_quality_evidence(raw, cleaned, changes)
    quality_records = build_quality_records(raw, changes, pending)

    # 清洗表保留业务字段；质量报告使用中文表头，保存时自动记录来源。
    write_csv(1, "清洗数据/cn_devices.csv", cn_clean)
    write_csv(1, "清洗数据/de_devices.csv", de_clean)
    write_csv(1, "清洗数据/service_cases.csv", cases_clean)
    write_csv(1, "质量处理.csv", quality_records)
    write_csv(1, "清洗对比.csv", comparison)
    write_csv(1, "文本线索.csv", text_findings)
    write_csv(1, "字段分级.csv", catalog)
    write_csv(1, "资产核对.csv", assets)
    print(f"已处理3张表，登记{len(catalog)}个字段；未治理项目见质量处理.csv。")


def choose_hours_repair():
    """返回运行时长修复函数；函数接收(values, groups, reference)，未选时返回None。"""
    # TODO02【选法与方案设计】：02-A、02-B分别选择两类时长的治理方法。
    # 处理对象：02-A修复华辰cn_devices.csv与德维de_devices.csv的operating_hours；
    # 02-B修复德维service_cases.csv的downtime_hours。三表均来自数据集，本题写入清洗副本。
    # 候选repair_fixed或repair_group_median；每类接入一条合格路径即可。
    # 返回的函数由主流程用于两张设备表；groups为型号，reference为同一共同统计。
    # 只修复缺失、无法解析、非有限、负数或超过744的值，合法值和原索引保持不变。
    # 【待填 02-A】将None换成回调；需补参数时可在框内写普通def，再返回其函数名。
    # 已给接入写法：在框内定义 def repair_hours(values, groups, reference):，
    # 下一行缩进return所选方法的调用；固定替代需给业务参数，分组中位数直接接收这三个参数。
    # ========== 作答区 02-A 开始 ==========
    from 可选算法.group_median_repair import repair_group_median

    repair_hours = repair_group_median
    # ========== 作答区 02-A 结束 ==========
    return repair_hours


def choose_downtime_repair():
    """返回停机时长修复函数；函数接收(values, groups, reference)，未选时返回None。"""
    # 主流程只将此方法用于案例表，groups为国家，reference来自去重案例的共同统计。
    # 处理对象02-B：德维service_cases.csv的downtime_hours列，不用于设备表的operating_hours。
    # 停机时长须有限且非负，没有给定真实上界；不使用未来故障标签估计。
    # 【待填 02-B】接入修复回调，返回同索引Series；72仅用于任务3统计贡献，不能清除合法长停机。
    # 已给接入写法：在框内定义 def repair_downtime(values, groups, reference):，
    # 下一行缩进return所选方法的调用；固定替代需给业务参数，分组中位数直接接收这三个参数。
    # ========== 作答区 02-B 开始 ==========
    from 可选算法.group_median_repair import repair_group_median

    repair_downtime = repair_group_median
    # ========== 作答区 02-B 结束 ==========
    return repair_downtime


def scan_service_notes(cases):
    """逐案例扫描service_note，返回含case_id的线索表；不修改原备注或输入表。"""
    # TODO03【选法与方案设计】：选择原文或规范化副本扫描，设计EMAIL和PHONE规则。
    # 处理对象：德维service_cases.csv清洗副本中的service_note列；用case_id定位案例。
    # 本题只扫描备注并生成文本线索.csv，不改写备注，也不处理两张设备表的联系列。
    # 候选detect_tokens、detect_normalized_tokens；返回不重复的命中标签列表。
    # 【待填 03-A】接入(text, patterns) -> list[str]回调，需要选项时用普通def包装。
    # ========== 作答区 03-A 开始 ==========
    from 可选算法.normalized_content_detection import detect_normalized_tokens

    scanner = detect_normalized_tokens
    # ========== 作答区 03-A 结束 ==========

    # 【待填 03-B】已给基础EMAIL规则；将PHONE的None换成电话正则，可按扫描方式调整规则。
    # 必做范围为本课合成邮箱与带+区号的电话，考虑括号、空白和连字符；其他格式和线索选做。
    # EMPLOYEE、LOCATION、NAMED_CONTACT为选做规则，保留None时跳过。
    # 邮箱基础规则只覆盖.invalid演示域；解释识别边界即可，不要求通用邮箱或全球号码验证。
    # ========== 作答区 03-B 开始 ==========
    patterns = {
        "EMAIL": r"[\w.+-]+@[\w.-]+\.invalid",
        "PHONE": r"\+[1-9](?:[\s()\-]*\d){7,14}",
        "EMPLOYEE": r"员工编号\s*:\s*(?:HC|DW)-E[0-9]{4}\b",
        "LOCATION": (
            r"(?:维修地点|现场)\s*:\s*(?:[\u4e00-\u9fff]+(?:路|街|巷)\s*[0-9]+号"
            r"|[A-Za-z][A-Za-z -]*\s+[0-9]+\s*,\s*Halle\s+[0-9]+\s*,\s*[A-Za-z]+)"
        ),
        "NAMED_CONTACT": (
            r"(?:请联系\s*|(?:现场联络人|现场工程师)\s*:?\s*)(?!待确认|待核实|未知|暂无)"
            r"(?:[\u4e00-\u9fff]{2,4}|[A-Za-z]+(?:[ -][A-Za-z]+)+)"
            r"(?=\s*(?:\(|[;；。]|已|$))"
        ),
    }
    # ========== 作答区 03-B 结束 ==========

    if scanner is not None and not callable(scanner):
        raise TypeError("scanner必须为扫描函数或None")
    if not isinstance(patterns, dict):
        raise TypeError("patterns必须为规则字典")
    # None表示未填写；必做规则未填时不扫描，选做规则未填时跳过。
    if any(pattern is not None and (not isinstance(pattern, str) or not pattern.strip())
           for pattern in patterns.values()):
        raise ValueError("正则须为非空字符串；未填写的规则保留None")
    active_patterns = {name: pattern for name, pattern in patterns.items() if pattern is not None}
    if scanner is not None and patterns and not {"EMAIL", "PHONE"} <= patterns.keys():
        raise ValueError("文本规则至少包含EMAIL和PHONE两类")
    scanner_ready = scanner is not None and {"EMAIL", "PHONE"} <= active_patterns.keys()

    rows = []
    for case in cases.to_dict("records"):
        note = case["service_note"]
        if pd.isna(note):
            note = ""
        else:
            note = str(note)
        if not scanner_ready:
            clues = "NOT_SCANNED"
            review = "UNASSESSED"
        else:
            hits = scanner(note, active_patterns)
            if not isinstance(hits, list):
                raise ValueError("扫描结果必须是标签列表；空列表表示已扫描未命中")
            if any(not isinstance(hit, str) or hit not in active_patterns for hit in hits):
                raise ValueError("扫描返回了规则之外的标签")
            if len(hits) != len(set(hits)):
                raise ValueError("扫描返回的标签不能重复")
            clues = "|".join(hits)
            if hits:
                review = "REVIEW_REQUIRED"
            else:
                review = "NOT_DETECTED"
        rows.append({"asset": "service_cases.csv", "record_id": case["case_id"],
                     "field": "service_note", "clues": clues, "review": review})
    return pd.DataFrame(rows, columns=["asset", "record_id", "field", "clues", "review"])


if __name__ == "__main__":
    main()
