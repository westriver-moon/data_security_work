"""任务5：核对实际结果，把前序保存和本次检查的真实记录写入有限数量的区块。"""
# 【待填】给出题目要求；每对“作答区”边框标出需要填写的位置。

# 本文件包含第12—14题：搜索【待填】定位来源重建、查询及审计核对。

# 修改下面数字即可调整区块总数，包含创世块0；默认生成区块0、1、2、3。
# 至少填2，最多填“本次真实操作记录数+1”；程序会均分保存，再读回检查并查询。
BLOCK_COUNT = 4

from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import pandas as pd
import numpy as np
from utils import (
    prepare_task_run, task_dir, run_dir, write_csv, load_results, read_events,
    record_event, USES, INTERNAL_TOKEN_DOMAIN,
)
from result_checks import inspect_result, tokens_for
from audit_support import (
    append_audit, validate_block_count, split_records, verify_chain_structure,
    save_block_records, show_block, verify_tampering, save_audit_results,
    read_saved_chain, checked_chain_events, pending_queries, summarize_query,
    locate_queries, unavailable_verification,
)

# 结果核验和区块辅助逻辑在同目录的两个文件中；本文件保留主流程与全部作答区。
# 作答时在对应作答框内导入实际调用的辅助函数，避免预置未使用的导入。


def main():
    """核对六类结果并将实际保存、检查事件写入审计链和查询表。
    读取本次结果与过程日志，保存检查、操作和审计证据。
    前序日志损坏时保留原件，单独记录本次失败并在保存后失败退出。
    """
    prepare_task_run(5)
    evidence_dir = task_dir(5) / "辅助材料"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    raw_log = evidence_dir / "审计原始记录.jsonl"
    failure_log = evidence_dir / "失败过程记录.jsonl"
    if raw_log.exists() or failure_log.exists():
        raise SystemExit("本次目录已有审计链或失败过程记录，请开始一次新的运行。")
    # 已提供：先核验前序日志。原文件保持原样，损坏时另存本次失败事件。
    log_error = ""
    try:
        if not read_events():
            raise ValueError("过程记录.jsonl缺失或没有有效事件")
    except (OSError, ValueError) as error:
        log_error = f"前序过程记录无法核验：{error}"
    event_log = failure_log if log_error else None
    if log_error:
        record_event(
            5, "读取过程记录", "过程记录", run_dir() / "过程记录.jsonl",
            "CHECK_FAILED", log_error + "；后续审计仅含本次失败记录",
            log_path=event_log,
        )
    # 已提供：登记损坏也必须留下失败事实，完成审计后再报告入口失败。
    registry_error = ""
    try:
        artifacts = load_results()
    except (OSError, ValueError, TypeError, KeyError) as error:
        artifacts = {}
        registry_error = str(error)
        record_event(5, "读取结果登记", "结果登记", None, "CHECK_FAILED", registry_error, log_path=event_log)
    checks = []
    # 六类结果逐项读取、检查；缺失或失败也留下真实检查记录。
    for name in USES:
        artifact = artifacts.get(name)
        if log_error:
            status, reason = "CHECK_FAILED", log_error
        elif registry_error:
            status, reason = "CHECK_FAILED", "无法核验结果登记；" + registry_error
        else:
            status, reason = inspect_result(name, artifact, validate_internal, artifacts)
        relative = artifact["file"] if artifact else ""
        candidate_path = (run_dir() / relative).resolve() if relative else None
        if candidate_path and candidate_path.is_relative_to(run_dir().resolve()):
            safe_path = candidate_path
        else:
            safe_path = None
        record_event(
            5,
            "检查结果",
            name,
            safe_path,
            status,
            reason,
            artifact["method"] if artifact else "NONE",
            log_path=event_log,
        )
        checks.append(
            {"result_name": name, "file": relative, "status": status, "reason": reason}
        )
    write_csv(5, "结果检查.csv", pd.DataFrame(checks))
    # 这些记录已在前序保存与本次检查时产生，不在这里编造操作时间或重复填充。
    events = read_events(event_log)
    # 失败日志只有本次真实事件；区块数过多时缩减，不编造事件填充。
    if log_error and type(BLOCK_COUNT) is int:
        block_count = min(BLOCK_COUNT, len(events) + 1)
    else:
        block_count = BLOCK_COUNT
    validate_block_count(len(events), block_count)
    # 写入完成后关闭文件，再从磁盘读取；验链与查询都使用读回的内容。
    chain = []
    disk_error = ""
    try:
        with raw_log.open("x", encoding="utf-8") as handle:
            for group in split_records(events, block_count - 1):
                old_size = len(chain)
                append_audit(chain, group)
                if len(chain) != old_size + (2 if old_size == 0 else 1):
                    raise ValueError("每组记录追加一个业务块，首次还需生成创世块")
                for block in chain[old_size:]:
                    handle.write(json.dumps(block, ensure_ascii=False) + "\n")
        saved_chain = read_saved_chain(raw_log)
    except (OSError, ValueError, TypeError) as error:
        saved_chain = None
        disk_error = f"审计链文件写入或读回失败：{error}"

    source_scope = "仅本次失败记录" if log_error else "本次完整操作记录"
    locations = {}
    query_events = events  # 未通过查询前检查时，仅作为明确标注的诊断明细。
    record_source = source_scope + "的原始日志（诊断，未通过链检查）"
    event_error = ""
    if saved_chain is None:
        save_block_records([])
        chain_status = "READ_FAILED"
        verification = unavailable_verification(disk_error)
        queries = pending_queries("NOT_EXECUTED", source_scope, disk_error)
    else:
        save_block_records(saved_chain)
        for block in saved_chain:
            show_block(block)
        # 原链只验证一次；变化演示复用此结论，防止查询和验证表口径不一致。
        try:
            original_result = verify_audit_chain(deepcopy(saved_chain))
        except Exception as error:
            original_result = (False, f"原链验证执行失败：{type(error).__name__}：{error}")
        passed, original_reason = original_result
        verification = verify_tampering(saved_chain, verify_audit_chain, original_result)
        if passed is None:
            chain_status = "PENDING"
            queries = pending_queries("WAITING_CHAIN", source_scope, "14-A尚未完成，等待验链；尚未执行第13题查询")
        elif not passed:
            chain_status = "FAILED"
            queries = pending_queries("NOT_EXECUTED", source_scope, original_reason)
        else:
            try:
                query_events, locations = checked_chain_events(saved_chain, events)
            except (ValueError, TypeError, KeyError, OverflowError) as error:
                event_error = f"链内事件或本次保存一致性检查失败：{error}"
                chain_status = "EVENT_FAILED"
                queries = pending_queries("NOT_EXECUTED", source_scope, event_error)
                verification.loc[verification["test"] == "ORIGINAL", "reason"] += "；" + event_error
            else:
                chain_status = "VERIFIED"
                record_source = source_scope + "的已检查链内事件"
                queries = locate_queries(audit_queries(query_events), locations, source_scope)
                verification.loc[verification["test"] == "ORIGINAL", "reason"] += "；链内事件及本次写入、读回内容一致"
    if log_error:
        verification["reason"] += "；仅检查本次失败记录，前序历史无法核验"
    save_audit_results(query_events, verification, queries, locations, record_source, chain_status)
    print(f"已保存结果检查、操作查询与审计证据；查询前检查状态：{chain_status}。", flush=True)
    if log_error:
        print("前序过程记录无法核验；六类结果均检查失败。本次失败记录、检查表与审计链已保存，请开始新的运行。", flush=True)
    if registry_error:
        print("结果登记读取失败；完整可读日志中的失败检查与审计记录已保存。" if not log_error else "结果登记也读取失败；原因见本次失败记录。", flush=True)
    if disk_error or event_error:
        print(disk_error or event_error, flush=True)
    if log_error or registry_error or disk_error or event_error:
        raise SystemExit(1)


def validate_internal(frame, cases):
    """检查内部服务分析的字段、数值及其与来源案例的完整对应关系。
    输入实际表与清洗案例；返回bool，未填写检查时返回None。
    合法行重排可通过，合法范围内的错配仍需拒绝。
    """
    # TODO12【核心填空与接入】：从来源构造预期记录并比较；字段格式和业务值检查已提供。
    # 处理对象：frame是任务3结果文件/内部服务分析.csv；cases是任务1清洗数据/service_cases.csv。
    # 比较device_token与downtime_hours、confirmed_failure的完整记录，结论进入本任务结果检查.csv。
    # 输入：实际内部分析frame、清洗案例cases；输出：bool，未完成返回None。
    # 要求：恰好规定三列，令牌32个小写十六进制字符，停机非负有限、故障为0/1。
    # 每条业务值应与来源对应，允许行重排；保留合法长停机，不能把统计贡献上限72小时用于明细。
    # 已提供令牌函数及same_records；预期表由12-A构造；完成后看结果检查.csv和操作记录.csv。
    # 框架先核对字段和业务值；只有它们合格时，才进入12-A比较完整记录。
    # 已提供：列名不能重复；同设备可有多条案例，令牌无需逐行唯一。
    fields_ok = (
        frame.columns.is_unique
        and set(frame) == {"device_token", "downtime_hours", "confirmed_failure"}
        and frame.device_token.astype(str).str.fullmatch(r"[0-9a-f]{32}").all()
        and len(frame) == len(cases)
        and frame.device_token.nunique() == cases.device_id.nunique()
    )
    if not fields_ok:
        return False
    # 已提供：不接受缺失、无穷、非法文本、负停机或非0/1故障；合法长停机保留。
    try:
        values_ok = (
            frame[["downtime_hours", "confirmed_failure"]].notna().all().all()
            and np.isfinite(frame[["downtime_hours", "confirmed_failure"]].to_numpy(dtype=float)).all()
            and frame.confirmed_failure.isin([0, 1]).all()
            and frame.downtime_hours.ge(0).all()
        )
    except (TypeError, ValueError, OverflowError):
        return False
    if not values_ok:
        return False

    # 【待填 12-A】从来源cases构造预期三列表，再比较实际frame的完整记录和重复次数。
    # 可用tokens_for(device_ids, INTERNAL_TOKEN_DOMAIN, 32)重建假名，same_records比较完整行。
    # 预期业务值来自cases，不可复制frame自证正确；同时核对重建令牌没有设备间碰撞。
    # 允许完整行重排，不修改来源或实际结果；返回判断赋给records_ok，未完成保留None。
    # ========== 作答区 12-A 开始 ==========
    from result_checks import same_records

    expected = cases[["downtime_hours", "confirmed_failure"]].copy()
    expected.insert(0, "device_token", tokens_for(cases["device_id"], INTERNAL_TOKEN_DOMAIN, 32))
    no_collision = expected["device_token"].nunique() == cases["device_id"].nunique()
    records_ok = no_collision and same_records(frame, expected)
    # ========== 作答区 12-A 结束 ==========

    if records_ok is None:
        return None  # 可运行占位：检查未完成，不代表通过或失败。
    return bool(fields_ok and values_ok and records_ok)


def audit_queries(events):
    """按真实事件生成两条已提供查询和一条自主设计查询。
    输入已通过查询前检查的链内业务事件，返回问题、状态、次数及命中编号。
    """
    # TODO13【自由设计】：提出一个操作查询问题，填写问题和筛选条件；事件来源范围已提供。
    # 处理对象：本次辅助材料/审计原始记录.jsonl中已通过检查的业务事件，不是业务CSV；损坏日志时仅使用明确标注的本次失败记录。
    # 可按result_name、action、status、time等事件字段筛选，实际次数写入本任务操作查询.csv。
    # 输入：真实事件events；输出：查询表中的一行，次数由框架计算。
    # 自行选择某个结果或一类操作；返回逐事件bool，不硬编码预期次数。
    # 两条已有查询可参考；在操作查询.csv核对本题结果，并解释一条匹配事件。
    # 问题应服务于一个实际核查目的；已有事件充足时可按结果、操作、状态或其组合筛选。
    # 零次也是有效观察：说明所用条件和本次范围；不为得到非零答案而编造事件或硬编码数量。
    # 【待填 13-A】替换问题文字：写成能用一条事件的字段判断是否计数的问题。
    # 提示：可围绕一种操作、一个结果或某种状态设计；答案必须来自events。
    # ========== 作答区 13-A 开始 ==========
    question = "实际检查通过了多少次业务结果"
    # ========== 作答区 13-A 结束 ==========

    # 已提供：scope说明事件来源；13-A的问题说明筛选目标，13-B执行筛选。
    # 日志损坏时，main还会明确标注“仅本次失败记录”，不把局部记录当完整历史。
    scope = "按自主问题与条件筛选"

    def is_saved_result(event):
        """判断真实事件是否记录了处理结果保存。"""
        return event["action"] == "保存处理结果"

    rows = [summarize_query(events, "实际保存了多少次处理结果", "保存处理结果", is_saved_result)]

    # 每个筛选函数接收一条事件字典，返回该事件是否应计数。
    # 浏览events里的action、status、result_name和time，保持13-A与筛选条件一致。
    # 失败查询已提供；13-B填写belongs_to_result的逐事件条件，不能写死参考次数。
    def is_failed_check(event):
        """判断一条事件是否记录了结果检查失败，返回布尔值。"""
        return event["action"] == "检查结果" and event["status"] == "CHECK_FAILED"

    def belongs_to_result(event):
        """按自定义问题判断一条事件是否属于统计范围。
        输入事件字典，完成后返回bool；占位None表示尚未设计条件。
        """
        # 【待填 13-B】替换None：输入一条事件字典，返回是否属于13-A所述范围的bool。
        # 提示：使用event中的action、status或result_name比较，可组合条件；不要返回次数。
        # 可运行占位：未设计条件，统计留空；真实无匹配应返回False。
        # ========== 作答区 13-B 开始 ==========
        return event["action"] == "检查结果" and event["status"] == "CHECK_PASSED"
        # ========== 作答区 13-B 结束 ==========


    # 每条查询的计数与命中编号来自同一轮判断；不为了定位再次执行条件。
    rows.append(summarize_query(events, "实际检查失败了多少次", "检查结果", is_failed_check))
    question_ready = isinstance(question, str) and bool(question.strip()) and question.strip() != "待填写查询问题"
    rows.append(summarize_query(events, question, scope, belongs_to_result, question_ready))
    return pd.DataFrame(rows)


def verify_audit_chain(chain):
    """用第14题逐块核对传入链；主流程传入磁盘读回链，本函数不修复内容。"""
    return verify_chain_structure(chain, check_audit_block)


def check_audit_block(chain, position, computed):
    """输入已校验结构的链、位置和重算摘要；返回(一致性bool, 原因)，未填返回None。"""
    # TODO14【审计验证】：检查编号连续、前序摘要链接及本块内容摘要。
    # 框架已拒绝空链及字段类型错误，computed是排除block_hash后计算的SHA-256。
    # 创世块previous_hash为64个0；其他块链接前一块保存的block_hash。
    # 不能在验证时修改链或重写摘要。成功仅表示当前结构一致，不证明历史完整。
    # 观察审计链验证.csv；框架提供原链和六种变化，不要求学生构造这些变化。
    # 【待填 14-A】核对本块编号、前序链接和内容摘要，返回(bool, 原因)。
    # ========== 作答区 14-A 开始 ==========
    block = chain[position]
    if block["block_index"] != position:
        return False, f"区块 #{position} 的编号不连续"
    expected_previous = "0" * 64 if position == 0 else chain[position - 1]["block_hash"]
    if block["previous_hash"] != expected_previous:
        return False, f"区块 #{position} 的前序摘要链接不一致"
    if block["block_hash"] != computed:
        return False, f"区块 #{position} 的内容摘要不一致"
    return True, f"区块 #{position} 的编号、链接和内容摘要一致"
    # ========== 作答区 14-A 结束 ==========

if __name__ == "__main__":
    main()
