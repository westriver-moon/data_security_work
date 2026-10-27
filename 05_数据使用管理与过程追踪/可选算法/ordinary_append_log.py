"""普通JSON Lines追加日志。

算法背景：追加日志按照事件发生顺序将记录写到文件末尾。JSON Lines格式让每一行保存一个
独立JSON对象，便于流式写入、逐行读取和故障后的局部恢复。日志常用于运行诊断、行为分析、
审计取证和系统监控。

核心步骤：①为事件补充UTC时间；②把对象稳定序列化为一行JSON；③以追加模式写入文件；
④读取时逐行解析。追加模式避免正常写入覆盖既有内容，但它不是只写存储机制。

输入与输出：输入是JSON兼容的事件字典和日志路径，输出是在指定路径增加的一条文本记录；
读取函数输出按文件顺序排列的事件列表。示例使用临时目录，程序结束后自动清理演示文件。

作用与局限：普通日志能留下事件记录，却没有密码学链接、数字签名、可信时间戳或外部副本。
拥有文件写权限的人可以修改内容、删除某行、调整顺序或重写整个文件，而读取程序通常无法
仅靠现有内容发现这些变化。读取时会拒绝坏行与非对象事件，追加前先确认旧内容可读，
并为合法但未以换行结束的最后一行补分隔符。本例没有并发锁，不适合多个写入者同时追加。日志可靠性还取决于权限管理、备份、轮换和时钟准确性。
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile

# 独立演示数据：事件内容在文件开头定义，运行时只补充记录时间。
DEMO_EVENTS = [
    {"event_id": "EVT-001", "actor": "operator-A", "operation": "OPEN"},
    {"event_id": "EVT-002", "actor": "operator-B", "operation": "UPDATE"},
    {"event_id": "EVT-003", "actor": "operator-A", "operation": "CLOSE"},
]


def append_json_event(log_path: Path, event: dict[str, object]) -> None:
    """为事件补充UTC时间，并以一行JSON追加到日志文件。"""
    log_path = Path(log_path)
    if not isinstance(event, dict):
        raise TypeError("事件必须是JSON对象")
    if log_path.exists():
        read_json_events(log_path)  # 既有坏日志应保留，不能继续追加掩盖错误。
    recorded_event = {
        **event,
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    payload = (json.dumps(recorded_event, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    with log_path.open("ab+") as log_file:
        log_file.seek(0, 2)
        if log_file.tell():
            log_file.seek(-1, 2)
            if log_file.read(1) != b"\n":
                log_file.write(b"\n")
        log_file.write(payload)


def read_json_events(log_path: Path) -> list[dict[str, object]]:
    """按文件中的原始顺序逐行读取事件。"""
    records = []
    with Path(log_path).open("r", encoding="utf-8") as log_file:
        for number, line in enumerate(log_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("事件必须是JSON对象")
                json.dumps(record, allow_nan=False)
            except (ValueError, TypeError) as error:
                raise ValueError(f"日志第{number}行无效：{error}") from error
            records.append(record)
    return records


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as demo_directory:
        demo_log_path = Path(demo_directory) / "events.jsonl"
        for demo_event in DEMO_EVENTS:
            append_json_event(demo_log_path, demo_event)
        print(json.dumps(read_json_events(demo_log_path), ensure_ascii=False, indent=2))
