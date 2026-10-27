"""任务5已提供的审计辅助：追加区块、构造验证变化并保存审计证据。
main.py组织真实事件，读回并检查保存的链，再查询链内事件；本文件提供读写与定位。
"""

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import pandas as pd
from utils import write_csv, TASKS
from 可选算法.simple_blockchain import calculate_block_hash, verify_blockchain


def append_audit(chain, batch):
    """把一组真实操作追加为一个业务块，首次同时建立创世块。
    输入可修改的链列表和事件批次；原地追加，不重复填充或伪造业务事件。
    """
    # 已提供：内容摘要、创世块、前序链接与真实事件追加。
    # 直接运行并阅读一块的结构；不需要补写摘要或区块追加代码。
    timestamp = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    if not chain:
        genesis = {
            "block_index": 0,
            "timestamp": timestamp,
            "transactions": [
                {"transaction_id": "GENESIS", "operation": "CREATE_CHAIN"}
            ],
            "previous_hash": "0" * 64,
        }
        genesis["block_hash"] = calculate_block_hash(genesis)
        chain.append(genesis)
    previous_hash = chain[-1]["block_hash"]  # 链接上一块的实际摘要。
    block = {
        "block_index": len(chain),
        "timestamp": timestamp,
        "transactions": deepcopy(batch),
        "previous_hash": previous_hash,
    }
    block["block_hash"] = calculate_block_hash(block)
    chain.append(block)


def verify_tampering(chain, verifier=verify_blockchain, original_result=None):
    """仅检验链结构变化，复用原链结论；不把事件检查误当成14-A的能力。"""
    rows = []
    for name, variant in audit_variants(chain).items():
        if variant is None:
            outcome, reason = "无法构造该操作", "事件不足，无法构造该变化"
        else:
            try:
                passed, reason = original_result if name == "ORIGINAL" and original_result is not None else verifier(variant)
                if passed is None:
                    outcome = "尚未实现"
                elif name == "ORIGINAL":
                    outcome = "原始结构一致" if passed else "原始结构不一致"
                elif name == "EMPTY":
                    outcome = "空链被接受，需检查实现" if passed else "已拒绝空链"
                else:
                    outcome = "未检出此变化" if passed else "检出结构或内容变化"
            except Exception as error:
                outcome, reason = "验证执行失败", f"{type(error).__name__}：{error}"
        if name == "DROP_LAST_BLOCK":
            reason += "；仅核对当前链时，有效前缀可能无法暴露末尾缺失"
        elif name == "REWRITE_AND_REHASH":
            reason += "；内容与摘要同时重算可能仍一致，当前链没有外部可信末端"
        rows.append({"test": name, "outcome": outcome, "reason": reason})
    return pd.DataFrame(rows)


def unavailable_verification(reason):
    """磁盘链不可用时仍保存明确的未开展记录，不拿内存好链代替。"""
    tests = ["ORIGINAL", "MODIFY", "DELETE", "REORDER", "DROP_LAST_BLOCK", "EMPTY", "REWRITE_AND_REHASH"]
    return pd.DataFrame([{"test": name, "outcome": "无法验证" if name == "ORIGINAL" else "未开展",
                          "reason": reason} for name in tests])


def audit_variants(chain):
    """已提供：在副本上构造变化；整链重算用示例摘要函数，原始链不受影响。"""
    variants = {
        "ORIGINAL": deepcopy(chain),
        "DROP_LAST_BLOCK": deepcopy(chain[:-1]),
        "EMPTY": [],
    }
    for name in ["MODIFY", "DELETE", "REORDER", "REWRITE_AND_REHASH"]:
        variants[name] = deepcopy(chain)
    if len(chain) > 1 and chain[1]["transactions"]:
        variants["MODIFY"][1]["transactions"][0]["status"] = "TAMPERED"
        del variants["DELETE"][1]["transactions"][0]
        if len(chain[1]["transactions"]) > 1:
            events = variants["REORDER"][1]["transactions"]
            events[0], events[1] = events[1], events[0]
        elif len(chain) > 2:
            variants["REORDER"][1], variants["REORDER"][2] = (
                variants["REORDER"][2],
                variants["REORDER"][1],
            )
        else:
            variants["REORDER"] = None
        rewritten = variants["REWRITE_AND_REHASH"]
        rewritten[1]["transactions"][0]["status"] = "REWRITTEN"
        for i, block in enumerate(rewritten):
            block["previous_hash"] = (
                "0" * 64 if i == 0 else rewritten[i - 1]["block_hash"]
            )
            block["block_hash"] = calculate_block_hash(block)
    else:
        for name in ["MODIFY", "DELETE", "REORDER", "REWRITE_AND_REHASH"]:
            variants[name] = None
    order = [
        "ORIGINAL",
        "MODIFY",
        "DELETE",
        "REORDER",
        "DROP_LAST_BLOCK",
        "EMPTY",
        "REWRITE_AND_REHASH",
    ]
    return {name: variants[name] for name in order}


def validate_block_count(record_count, block_count):
    """已提供：普通运行包含一个创世块，每个业务块至少保存一条记录。"""
    if type(block_count) is not int or not 2 <= block_count <= record_count + 1:
        raise SystemExit(f"区块总数需为2至{record_count + 1}之间的整数，包含创世块0。")


def split_records(records, block_count):
    """已提供：按原顺序尽量均分；余数从前面的业务块开始分配。"""
    size, extra = divmod(len(records), block_count)
    groups, offset = [], 0
    for number in range(block_count):
        extra_record = 1 if number < extra else 0
        end = offset + size + extra_record
        groups.append(records[offset:end])
        offset = end
    return groups


def hash_text(value):
    """摘要缺失时明确提示；CSV和原始记录中的有效摘要不截断。"""
    return "摘要缺失" if value is None else str(value)


def block_description(block):
    """概括区块保存的业务对象与事件条数。
    输入一个区块，返回说明与记录数；创世块单独说明初始化用途。
    """
    if block["block_index"] == 0:
        return "初始化审计链", 0
    events = block["transactions"]
    names = []
    for event in events:
        if "/清洗数据/" in str(event.get("file", "")):
            name = "清洗数据"
        else:
            name = str(event.get("result_name", "字段无效的事件"))
        if name not in names:
            names.append(name)
    return "、".join(names) + f"的操作记录（{len(events)}条）", len(events)


def show_block(block):
    """已提供：只显示本块基本信息，摘要在终端截取前12位。"""
    content, _ = block_description(block)
    print(
        f"【区块{block['block_index']}】\n保存内容：{content}\n时间：{block['timestamp']}\n前序哈希：{hash_text(block['previous_hash'])[:12]}\n本块哈希：{hash_text(block['block_hash'])[:12]}\n",
        flush=True,
    )


def save_block_records(chain):
    """已提供：每块一行，包含创世块；完整业务事件另存在唯一的JSONL文件。"""
    rows = []
    for block in chain:
        content, count = block_description(block)
        rows.append(
            {
                "block_index": block["block_index"],
                "content": content,
                "record_count": count,
                "timestamp": block["timestamp"],
                "previous_hash": hash_text(block["previous_hash"]),
                "block_hash": hash_text(block["block_hash"]),
            }
        )
    write_csv(5, "辅助材料/审计区块概览.csv", pd.DataFrame(rows, columns=[
        "block_index", "content", "record_count", "timestamp", "previous_hash", "block_hash",
    ]))


def save_audit_results(events, verification, queries, locations, record_source, chain_status):
    """保存已计算的结论及明细；诊断日志不标记为已经验链的记录。"""
    event_fields = ["event_id", "task", "action", "result_name", "file", "status", "reason", "time"]
    usage = pd.DataFrame(events, columns=event_fields)
    usage["block_index"] = pd.array([locations.get(event["event_id"]) for event in events], dtype="Int64")
    usage["record_source"] = record_source
    usage["chain_status"] = chain_status
    write_csv(5, "操作记录.csv", usage)
    write_csv(5, "审计链验证.csv", verification[["test", "outcome", "reason"]])
    output = queries.copy()
    output["count"] = pd.array(output["count"], dtype="Int64")
    write_csv(5, "操作查询.csv", output)


def read_saved_chain(path):
    """严格读回本次磁盘链，不跳过坏行、不修改内容或重算摘要。"""
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"JSON含重复字段：{key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"JSON含非有限常量：{value}")

    rows = Path(path).read_text(encoding="utf-8").splitlines()
    if not rows:
        raise ValueError("审计链文件为空")
    chain = []
    for number, line in enumerate(rows, 1):
        try:
            block = json.loads(line, object_pairs_hook=unique_object, parse_constant=reject_constant)
        except ValueError as error:
            raise ValueError(f"审计链第{number}行无法读取：{error}") from error
        if (not isinstance(block, dict) or type(block.get("block_index")) is not int
                or not isinstance(block.get("timestamp"), str)
                or not isinstance(block.get("transactions"), list)
                or any(not isinstance(item, dict) for item in block["transactions"])
                or not isinstance(block.get("previous_hash"), str)
                or not isinstance(block.get("block_hash"), str)):
            raise ValueError(f"审计链第{number}行字段缺失或类型无效")
        chain.append(block)
    return chain


def checked_chain_events(chain, written_events):
    """结构验链后检查查询输入及本次保存一致性；不证明历史真实完整。"""
    if chain[0]["transactions"] != [{"transaction_id": "GENESIS", "operation": "CREATE_CHAIN"}]:
        raise ValueError("创世块建链标记无效，不能计为业务事件")
    fields = {"action", "result_name", "file", "sha256", "method", "metadata_sha256", "status", "reason", "time"}
    events, locations = [], {}
    previous_event_time = previous_block_time = None

    def parse_time(value, label):
        time = datetime.fromisoformat(value)
        if time.tzinfo is None or time.utcoffset() is None:
            raise ValueError(label + "缺少时区")
        return time

    for position, block in enumerate(chain):
        block_time = parse_time(block["timestamp"], "区块时间")
        if previous_block_time is not None and block_time < previous_block_time:
            raise ValueError("区块组链时间倒序")
        previous_block_time = block_time
        if position == 0:
            continue
        if not block["transactions"]:
            raise ValueError("业务区块没有实际事件")
        for event in block["transactions"]:
            if (not fields.union({"event_id", "task", "sources"}) <= event.keys()
                    or any(not isinstance(event[field], str) for field in fields)
                    or type(event["event_id"]) is not int or event["event_id"] != len(events) + 1
                    or type(event["task"]) is not int or event["task"] not in TASKS
                    or not isinstance(event["sources"], dict)
                    or any(not isinstance(k, str) or not isinstance(v, str) for k, v in event["sources"].items())):
                raise ValueError(f"区块{position}含字段、类型或编号无效的业务事件")
            event_time = parse_time(event["time"], "事件时间")
            if previous_event_time is not None and event_time < previous_event_time:
                raise ValueError("业务事件时间倒序")
            if event_time > block_time:
                raise ValueError("事件时间晚于所在区块的组链时间")
            previous_event_time = event_time
            locations[event["event_id"]] = block["block_index"]
            events.append(deepcopy(event))
    if not events:
        raise ValueError("链中没有可供查询的业务事件")
    if events != written_events:
        raise ValueError("写入与读回的业务事件数量、顺序或内容不一致")
    return events, locations


def pending_queries(status, scope, reason):
    """未取得已检查链时只保存待执行行，绝不对空列表统计零次。"""
    names = ["实际保存了多少次处理结果", "实际检查失败了多少次", "自主查询（第13题，尚未执行）"]
    return pd.DataFrame([{"query": name, "scope": scope, "query_status": status,
                          "count": pd.NA, "matched_locations": "", "reason": reason} for name in names])


def summarize_query(events, question, scope, predicate, question_ready=True):
    """每事件仅调用一次条件；收集计数及序号，不改变原事件。"""
    row = {"query": question if isinstance(question, str) else "查询问题格式无效", "scope": scope,
           "query_status": "ANSWER_PENDING", "count": pd.NA, "_matched_ids": [], "reason": "查询问题或条件尚未完成"}
    if not events:
        row.update(query_status="NO_EVENTS", reason="没有可用于执行查询的业务事件")
        return row
    if not question_ready:
        return row
    matches, pending = [], False
    try:
        for source in events:
            event = deepcopy(source)
            decision = predicate(event)
            if event != source:
                raise ValueError("查询条件不能修改事件内容")
            if decision is None:
                pending = True
            elif not isinstance(decision, (bool, np.bool_)):
                raise ValueError("逐事件条件必须返回bool；None仅表示尚未作答")
            elif decision:
                matches.append(source["event_id"])
    except Exception as error:
        row.update(query_status="QUERY_FAILED", reason=f"{type(error).__name__}：{error}")
        return row
    if pending:
        return row
    row.update(query_status="COMPLETED", count=len(matches), _matched_ids=matches,
               reason="按本次实际事件统计；次数为0不等于全部作答完成" if not matches else "按本次实际命中事件统计")
    return row


def locate_queries(queries, locations, source_scope):
    """用独立索引补充所在区块；不重跑学生条件，不向链内事件加字段。"""
    output = queries.copy()
    def describe(ids):
        return "；".join(f"事件{event_id}（区块{locations[event_id]}）" for event_id in ids)
    output["matched_locations"] = output.pop("_matched_ids").map(describe)
    output["scope"] = source_scope + "；" + output["scope"]
    return output[["query", "scope", "query_status", "count", "matched_locations", "reason"]]


def verify_chain_structure(chain, check_block):
    """已提供结构与序列化检查；编号、链接和内容一致性由14-A实际判断。"""
    if not isinstance(chain, list) or not chain:
        return False, "空链或无效链缺少创世区块"
    snapshot = deepcopy(chain)
    for position, block in enumerate(chain):
        if not isinstance(block, dict):
            return False, f"区块位置{position}不是JSON对象"
        if (type(block.get("block_index")) is not int
                or not isinstance(block.get("timestamp"), str) or not block["timestamp"]
                or not isinstance(block.get("transactions"), list)
                or any(not isinstance(item, dict) for item in block["transactions"])
                or not isinstance(block.get("previous_hash"), str)
                or not isinstance(block.get("block_hash"), str)):
            return False, f"区块 #{position} 的字段类型或交易结构无效"
        try:
            computed = calculate_block_hash(block)
        except (ValueError, TypeError):
            return False, f"区块 #{position} 含有不能稳定序列化的内容"
        result = check_block(chain, position, computed)
        if chain != snapshot:
            raise ValueError("14-A不能修改链或重写摘要")
        if result is None:
            return None, "14-A编号、链接和内容核对尚未完成"
        if (not isinstance(result, tuple) or len(result) != 2
                or type(result[0]) is not bool or not isinstance(result[1], str)):
            raise ValueError("14-A须返回(bool, 原因)，未作答返回None")
        passed, reason = result
        if not passed:
            return False, reason
    return True, f"结构一致，共{len(chain)}个区块；未证明历史无缺失"
