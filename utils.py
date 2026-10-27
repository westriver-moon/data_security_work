"""共用工具：数据读写、实际操作登记，以及项目运行、日志与概览。

学生先读任务背景与对应TODO，按需回查根main.py的任务交接；本文件均为已提供辅助。
任务2生成保护规则，任务3、4读取规则，任务5检查实际结果。这里不提供审批系统。
"""

from pathlib import Path
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import json
import os
import sys
import math
import signal
import subprocess
import traceback
from numbers import Real
import pandas as pd

ROOT = Path(__file__).resolve().parent
# 设备运行指标统计整月；3月有31天，按每天24小时计，上限为31 × 24 = 744小时。
OBSERVATION_MONTH = "2026-03"
# 随后90天的标签窗口：含4月1日、不含6月30日，即4月1日至6月29日。
PREDICTION_TIME = "2026-04-01"
LABEL_END_EXCLUSIVE = "2026-06-30"
STATISTIC_TOLERANCES = {"confirmed_case_count": 5.0, "mean_downtime": 1.0}
FEATURES = ["temperature_c", "operating_hours", "alarm_count"]
# 公开的课堂演示配置；任务3/4生成与任务5本地核验使用同一配置。
# 如更换方法、密钥或用途域，须同步修改生成与核验，并重新运行。
DEMO_TOKEN_KEY = b"teaching-only-key"
INTERNAL_TOKEN_DOMAIN = "service-review"
MATCH_TOKEN_DOMAIN = "fleet-match"
LOCAL_EVIDENCE = {"研究数据": (3, "匿名化方案.json")}
# 联系列即使整列缺失也按文本读取，保证.str规范化可用，并保留电话加号与邮编前导零。
CSV_TEXT_TYPES = {
    field: "string" for field in [
        "contact_name", "engineer_name", "contact_email", "engineer_email",
        "contact_phone", "engineer_phone", "site_postcode",
    ]
}
ASSETS = {
    "cn_devices.csv": ("华辰设备制造有限公司", "CN"),
    "de_devices.csv": ("德维工业服务有限公司", "DE"),
    "service_cases.csv": ("德维工业服务有限公司", "DE"),
}
TASKS = {
    1: "01_数据资产盘点与质量治理",
    2: "02_数据流转分析与使用决策",
    3: "03_数据共享与安全发布",
    4: "04_跨机构联合分析",
    5: "05_数据使用管理与过程追踪",
}

# 给定的实际用途；每个名称对应一个结果，不是申请编号。
USES = {
    "内部服务分析": {
        "task": 3,
        "sources": ["service_cases.csv"],
        "need": "关联同一设备的故障结论和停机时长，不带出原设备编号及联系资料",
    },
    "研究数据": {
        "task": 3,
        "sources": ["de_devices.csv"],
        "need": "按型号、区域和运行时长研究故障，降低明细识别风险",
    },
    "服务统计": {
        "task": 3,
        "sources": ["service_cases.csv"],
        "need": "统计确认故障案例数与全部案例的裁剪后平均停机小时",
    },
    "共同设备匹配": {
        "task": 4,
        "sources": ["cn_devices.csv", "de_devices.csv"],
        "need": "确定共同设备，供后续归属与去重使用",
    },
    "联合告警统计": {
        "task": 4,
        "sources": ["cn_devices.csv", "de_devices.csv"],
        "need": "每台设备参与一次，形成受保护的联合告警合计",
    },
    "联合故障模型": {
        "task": 4,
        "sources": ["cn_devices.csv", "de_devices.csv"],
        "need": "以三个观察期特征训练模型，只保存参数",
    },
}


def prepare_task_run(task):
    # 根入口会给子进程设置 COURSE_RUN_DIR；没有它就需要先进入根入口。
    """准备指定任务的运行目录，返回该输出目录的Path对象。
    
    输入为任务编号。由根入口启动时创建本次任务目录；直接启动任务时，
    先切换工作目录并以根入口替换当前进程，重新运行必要前置任务。
    """
    if not os.environ.get("COURSE_RUN_DIR"):
        parser = argparse.ArgumentParser(description=f"运行任务{task}及必要前置任务")
        parser.add_argument("--output-dir", type=Path, help="本次输出目录，必须是新目录")
        parser.add_argument("--verbose", action="store_true", help="回显各阶段完整日志")
        args = parser.parse_args()
        command = [sys.executable, str(ROOT / "main.py"), "--task", str(task)]
        if args.output_dir is not None:
            # 相对路径以用户启动任务时的工作目录为基准。
            command.extend(["--output-dir", str(args.output_dir.resolve())])
        if args.verbose:
            command.append("--verbose")
        os.chdir(ROOT)
        # execv 用根入口替换当前进程；成功后不会继续执行下面的目录创建。
        # 根入口再次启动本任务时已设置运行目录，因此不会反复转交。
        os.execv(sys.executable, command)
    folder = task_dir(task)
    # 核心结果直接保存在任务目录；清洗数据、处理结果需要时再建子目录。
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def run_dir():
    """返回本次运行的根目录；须先由根入口设置 COURSE_RUN_DIR。"""
    return Path(os.environ["COURSE_RUN_DIR"])


def task_dir(task):
    """把任务编号转换为本次运行中的任务输出目录，不是任务源码目录。"""
    return run_dir() / TASKS[task]


def raw_path(asset):
    """根据资产文件名返回原始数据路径，例如数据集/持有方/cn_devices.csv。"""
    return ROOT / "数据集" / ASSETS[asset][0] / asset


def clean_path(asset):
    """返回任务1在本次运行中生成的清洗表路径，供后续任务统一读取。"""
    return task_dir(1) / "清洗数据" / asset


def read_clean(asset):
    """读取指定清洗表并返回 DataFrame；本函数不执行清洗或补跑前置任务。"""
    return pd.read_csv(clean_path(asset), dtype=CSV_TEXT_TYPES)


def numeric_quality_issue(frame, bounds):
    """按字段上下界检查数值，返回问题说明；全部合法时返回空字符串。
    
    输入为数据表及字段到(lower, upper)的映射，upper=None表示不设上界。
    缺失、非数值、非有限或越界值均计入问题；不修复来源、不保存文件。
    """
    issues = []
    for field, (lower, upper) in bounds.items():
        values = pd.to_numeric(frame[field], errors="coerce")
        def is_finite_number(value):
            if pd.isna(value):
                return False
            return math.isfinite(value)

        valid = values.map(is_finite_number)
        valid &= values.ge(lower)
        if upper is not None:
            valid &= values.le(upper)
        count = int((~valid).sum())
        if count:
            issues.append(f"{field}有{count}处缺失、无法解析、非有限或越界值")
    return "；".join(issues)


def translate_cell(value, labels, field):
    """固定状态逐项转换；文本线索与标签的多值仍用竖线分隔。"""
    if pd.isna(value):
        return value
    if field in {"tags", "clues"} and isinstance(value, str):
        return "|".join(labels.get(part, part) for part in value.split("|"))
    return labels.get(value, value)


def save_csv(path, frame):
    """保存数据表并返回文件Path，必要时创建父目录。
    
    按文件名识别报告表，导出中文表头与状态；清洗表和业务结果保留约定字段。
    转换在副本上进行，不修改输入表；本函数本身不登记过程事件。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = TABLE_COLUMNS.get(path.name)
    if columns is None:
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        return path
    output = frame.copy()
    for field in output:
        if field in BOOL_LABELS:
            no, yes, missing = BOOL_LABELS[field]
            def export_boolean_label(value):
                """缺失写为缺失标签，True或1写为是；其他值沿用否标签。"""
                if pd.isna(value):
                    return missing
                if value == True:
                    return yes
                # 保留已有导出约定：False、0及未识别文本均使用否标签。
                return no

            output[field] = output[field].map(export_boolean_label)
        elif field in VALUE_LABELS:
            labels = VALUE_LABELS[field]
            output[field] = output[field].map(
                lambda x: translate_cell(x, labels, field)
            )
    order = [field for field in columns if field in output]
    order += [field for field in output if field not in columns]
    output[order].rename(columns=columns).to_csv(
        path, index=False, encoding="utf-8-sig", float_format="%.6f"
    )
    return path


def read_table(path, missing_ok=False):
    """读取结果CSV，返回恢复内部字段名与状态的DataFrame。
    
    missing_ok=True且文件不存在时返回空表。已知报告按文件名恢复中文列名，
    清洗数据另由read_clean读取，原始输入由任务1读取；本函数不写文件。
    """
    path = Path(path)
    if missing_ok and not path.exists():
        return pd.DataFrame()
    columns = TABLE_COLUMNS.get(path.name, {})
    # 旧表头只在明确列出的表中兼容，不能让两列表头悄悄合并到同一字段。
    reverse = {title: field for field, title in columns.items()}
    reverse.update(LEGACY_TABLE_HEADERS.get(path.name, {}))
    with path.open(encoding="utf-8-sig", newline="") as handle:
        headers = next(csv.reader(handle), [])
    restored_headers = [reverse.get(title, title) for title in headers]
    if len(restored_headers) != len(set(restored_headers)):
        raise ValueError(f"{path.name}：重复列或新旧表头同时对应同一字段，无法确定应读取哪列")
    text_fields = {
        title: "string" for title, field in zip(headers, restored_headers)
        if field.endswith("sha256") or field in {"previous_hash", "block_hash"}
    }
    frame = pd.read_csv(path, dtype=text_fields).rename(columns=reverse)
    for field in frame:
        if field in BOOL_LABELS:
            no, yes, missing = BOOL_LABELS[field]
            def restore_boolean_value(value):
                """缺失恢复为pd.NA，否/是标签恢复为布尔值；其他值原样保留。"""
                if pd.isna(value) or value == missing:
                    return pd.NA
                if value == no:
                    return False
                if value == yes:
                    return True
                # 未识别文本和已有的0/1等值留给后续校验，不猜测其状态。
                return value

            frame[field] = frame[field].map(restore_boolean_value)
        elif field in VALUE_LABELS:
            labels = {v: k for k, v in VALUE_LABELS[field].items()}
            labels.update(LEGACY_VALUE_LABELS.get(field, {}))
            frame[field] = frame[field].map(lambda x: translate_cell(x, labels, field))
    return frame


def sha256(path):
    """返回文件内容的SHA-256摘要；摘要用于一致性核对，不负责加密。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record_hash(record):
    # 固定键顺序和分隔符，让仅有字典键排列差异的记录得到相同摘要。
    # default=str 用于转换日期等JSON默认不支持的对象。
    """将一条字典记录按固定格式序列化，再计算摘要。"""
    content = json.dumps(
        record, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":")
    )
    return hashlib.sha256(content.encode()).hexdigest()


# 自动保存与操作记录：只登记文件名、摘要、方法和时间，不复制数据行。
def write_csv(task, name, frame):
    """保存任务结果并返回文件Path，对治理关键文件与规则自动记录来源。
    
    输入任务编号、相对文件名和DataFrame。先调用save_csv保存文件，
    再按任务与文件名决定是否追加过程事件；普通比较辅助表只保存CSV。
    """
    path = save_csv(task_dir(task) / name, frame)
    important = (
        task == 1
        and (
            name.startswith("清洗数据/")
            or name in {"质量处理.csv", "字段分级.csv", "资产核对.csv"}
        )
        or task == 2
        and name == "流转与保护规则.csv"
    )
    if important:
        if task == 1:
            sources = (
                [raw_path(Path(name).name)]
                if name.startswith("清洗数据/")
                else [raw_path(asset) for asset in ASSETS]
            )
            origins = {
                "原始数据/" + str(source.relative_to(ROOT / "数据集")): sha256(source)
                for source in sources
            }
        else:
            sources = [
                task_dir(1) / "字段分级.csv",
                *[clean_path(asset) for asset in ASSETS],
            ]
            origins = {
                str(source.relative_to(run_dir())): sha256(source) for source in sources
            }
        record_event(
            task,
            "保存文件",
            Path(name).stem,
            path,
            "RECORDED",
            "已保存本次计算结果",
            sources=origins,
        )
    return path


def read_events(log_path=None):
    """严格读取实际事件；可指定失败日志，损坏记录不能跳过后当作完整历史。"""
    path = Path(log_path) if log_path is not None else run_dir() / "过程记录.jsonl"
    if not path.exists():
        return []
    events = []
    previous_time = None
    text_fields = {
        "action", "result_name", "file", "sha256", "method",
        "metadata_sha256", "status", "reason", "time",
    }
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path.name}第{number}行不是有效JSON，无法核验过程记录") from error
        if (
            not isinstance(event, dict)
            or not text_fields.union({"event_id", "task", "sources"}) <= event.keys()
            or any(not isinstance(event[field], str) for field in text_fields)
            or type(event["event_id"]) is not int
            or event["event_id"] != len(events) + 1
            or type(event["task"]) is not int or event["task"] not in TASKS
            or not isinstance(event["sources"], dict)
            or any(not isinstance(k, str) or not isinstance(v, str) for k, v in event["sources"].items())
        ):
            raise ValueError(f"{path.name}第{number}行的事件字段、类型或编号无效")
        # 时间须可解析并带时区；按事件顺序核对，避免无效日期或倒序时间混入审计。
        try:
            event_time = datetime.fromisoformat(event["time"])
        except ValueError as error:
            raise ValueError(f"{path.name}第{number}行的事件时间无效") from error
        if event_time.tzinfo is None or event_time.utcoffset() is None:
            raise ValueError(f"{path.name}第{number}行的事件时间缺少时区")
        if previous_time is not None and event_time < previous_time:
            raise ValueError(f"{path.name}第{number}行的事件时间早于前一事件")
        previous_time = event_time
        events.append(event)
    return events


def record_event(
    task, action, result_name, path, status, reason, method="NONE", sources=None, metadata_sha256="",
    *, log_path=None,
):
    """将一次真实保存或检查追加到过程日志，返回生成的事件字典。
    输入任务、操作、结果、状态和来源；可指定独立失败日志。
    编号来自已有有效事件，记录实际时间及文件摘要；坏日志不能静默跳过。
    """
    event_path = Path(log_path) if log_path is not None else run_dir() / "过程记录.jsonl"
    events = read_events(event_path)
    path = Path(path) if path else None
    try:
        digest = sha256(path) if path and path.is_file() else ""
    except OSError:
        digest = ""
        reason += "；文件不可读，未取得摘要"
    event = {
        "event_id": len(events) + 1,
        "task": task,
        "action": action,
        "result_name": result_name,
        "file": str(path.relative_to(run_dir())) if path else "",
        "sha256": digest,
        "method": method,
        "sources": sources or {},
        "metadata_sha256": metadata_sha256,
        "status": status,
        "reason": reason,
        "time": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
    }
    # 最后一条JSON完整但缺少换行时，先补分隔符，避免两条事件黏连。
    with event_path.open("ab+") as handle:
        if handle.tell():
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) not in {b"\n", b"\r"}:
                handle.write(b"\n")
        handle.write((json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8"))
    return event


def verify_lineage():
    """已提供：沿真实保存记录核对原始数据→清洗/分级→规则的来源。
    
    检查本次运行与当前输入的一致性；没有外部锚点时，不能证明整套记录未被重写。
    """
    root = run_dir().resolve()
    events = read_events()

    def verify_saved(path, task, sources):
        """核对指定上游文件是否有且只有一条相符的保存事件。
        输入文件、任务号和预期来源摘要；成功无返回值，缺失或变化时抛出异常。
        """
        relative = str(path.relative_to(root))
        saved = [
            event for event in events
            if event.get("task") == task
            and event.get("action") == "保存文件"
            and event.get("file") == relative
        ]
        if len(saved) != 1:
            raise ValueError(f"来源保存记录缺失或重复：{relative}")
        event = saved[0]
        if (
            event.get("status") != "RECORDED"
            or event.get("sha256") != sha256(path)
            or event.get("sources") != sources
        ):
            raise ValueError(f"上游内容或来源已改变，请重算：{relative}")

    raw_sources = {}
    clean_sources = {}
    for asset in ASSETS:
        source = raw_path(asset)
        relative = "原始数据/" + str(source.relative_to(ROOT / "数据集"))
        digest = sha256(source)
        raw_sources[relative] = digest
        cleaned = clean_path(asset)
        verify_saved(cleaned, 1, {relative: digest})
        clean_sources[str(cleaned.relative_to(root))] = sha256(cleaned)
    catalog = task_dir(1) / "字段分级.csv"
    verify_saved(catalog, 1, raw_sources)
    clean_sources[str(catalog.relative_to(root))] = sha256(catalog)
    verify_saved(task_dir(2) / "流转与保护规则.csv", 2, clean_sources)


def load_rules():
    """读回任务2的规则；每种实际结果恰好一条，不按行位置关联。"""
    frame = read_table(task_dir(2) / "流转与保护规则.csv").fillna("")
    required = set(TABLE_COLUMNS["流转与保护规则.csv"])
    if (
        not required <= set(frame)
        or frame.result_name.duplicated().any()
        or set(frame.result_name) != set(USES)
    ):
        raise ValueError("保护规则必须完整覆盖六类实际结果，且名称唯一")
    return {row["result_name"]: row for row in frame.to_dict("records")}


def select_inputs(result_name, frames=None):
    """为指定用途读取输入表，返回以资产文件名为键的DataFrame副本字典。
    
    frames给出已读取的表；未提供时读取该用途的任务1清洗表。
    按任务2规则选择字段，规则未完成时保留完整字段基线；不修复或保存数据。
    """
    rule = load_rules()[result_name]
    names = USES[result_name]["sources"]
    data = frames if frames is not None else {name: read_clean(name) for name in names}
    if rule["ready"] is not True:
        return {name: data[name].copy() for name in names}
    selected = json.loads(rule["input_fields"])
    if set(selected) != set(names) or rule["sources"].split("|") != names:
        raise ValueError("规则的来源与实际用途不一致")
    output = {}
    for name in names:
        fields = selected[name]
        if (
            not fields
            or len(fields) != len(set(fields))
            or not set(fields) <= set(data[name])
        ):
            raise ValueError("规则包含缺失、重复或不存在的输入字段")
        output[name] = data[name][fields].copy()
    return output


def finite_real(value):
    """判断值能否表示为有限实数；布尔值及超出浮点表示范围的整数均不接受。"""
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        return False


def validate_dp_config(
    config, public_n, privacy_unit="SERVICE_CASE", allow_missing=False
):
    """核验固定公开记录数下的查询敏感度、机制与本次预算。
    服务案例允许整组Laplace或Gaussian；设备合计仍只接入Laplace。
    缺题可保留None，但已填写的非法值不能按未完成放行；不证明具体函数执行正确。
    """
    if (
        not finite_real(public_n)
        or public_n <= 0
        or public_n != int(public_n)
    ):
        raise ValueError("公开记录总数必须为正整数")
    if not isinstance(privacy_unit, str) or privacy_unit not in {"SERVICE_CASE", "DEVICE"}:
        raise ValueError("保护单位必须是SERVICE_CASE或DEVICE")
    lower_bounds = (
        {"confirmed_case_count": 1.0, "mean_downtime": 72.0 / public_n}
        if privacy_unit == "SERVICE_CASE" else {"fleet_alarm_sum": 8.0}
    )
    if not isinstance(config, dict) or set(config) != set(lower_bounds):
        raise ValueError("统计配置必须完整对应本次查询及保护单位")
    allowed = {"LAPLACE", "GAUSSIAN"} if privacy_unit == "SERVICE_CASE" else {"LAPLACE"}
    mechanisms = set()
    for query, lower in lower_bounds.items():
        item = config[query]
        label = {
            "confirmed_case_count": "确认故障案例数",
            "mean_downtime": "平均停机时长",
            "fleet_alarm_sum": "联合告警合计",
        }[query]
        if not isinstance(item, dict) or set(item) != {
            "sensitivity", "epsilon", "delta", "mechanism"
        }:
            raise ValueError(f"{label}：统计配置字段不完整")
        for field in ["sensitivity", "epsilon", "delta"]:
            value = item[field]
            if value is None and allow_missing:
                continue
            if not finite_real(value):
                raise ValueError(f"{label}：{field}必须是有限数")
        sensitivity, epsilon, delta = item["sensitivity"], item["epsilon"], item["delta"]
        if sensitivity is not None and sensitivity < lower:
            raise ValueError(f"{label}：敏感度低于查询所需下界，请按替换一个单位的口径重新推导")
        if epsilon is not None and not 0 < epsilon <= 1:
            raise ValueError(f"{label}：epsilon必须大于0且不超过1")
        if delta is not None and not 0 <= delta <= 1e-5:
            raise ValueError(f"{label}：delta必须位于[0, 1e-5]")
        mechanism = item["mechanism"]
        if mechanism is None and allow_missing:
            continue
        if not isinstance(mechanism, str) or mechanism not in allowed:
            names = "Laplace或Gaussian" if privacy_unit == "SERVICE_CASE" else "Laplace"
            raise ValueError(f"{label}：本用途只接入{names}机制")
        mechanisms.add(mechanism)
        if delta is not None and (
            mechanism == "LAPLACE" and delta != 0
            or mechanism == "GAUSSIAN" and delta <= 0
        ):
            raise ValueError(f"{label}：Laplace的delta必须为0，Gaussian的delta必须为正")
        if all(value is not None for value in [sensitivity, epsilon, delta]):
            scale = sensitivity / epsilon
            if mechanism == "GAUSSIAN":
                scale *= math.sqrt(2 * (math.log(1.25) - math.log(delta)))
            if not math.isfinite(scale):
                raise ValueError(f"{label}：参数导致噪声尺度非有限，请调整预算或敏感度")
    if len(mechanisms) > 1:
        raise ValueError("本题两项服务统计须选择同一种机制，不混合发布路线")
    if math.fsum(item["epsilon"] or 0 for item in config.values()) > 1:
        raise ValueError("全部查询的epsilon合计不能超过1")
    if math.fsum(item["delta"] or 0 for item in config.values()) > 1e-5:
        raise ValueError("全部查询的delta合计不能超过1e-5")


def save_and_register_result(
    result_name, frame, method, ready, details, dependencies=(),
    protection_config=None, status=None,
):
    """校验用途规则、保存结果并追加来源事件，返回本结果的登记字典。
    
    输入结果名称、数据表、方法、就绪情况、说明，以及显式状态、保护配置和依赖。
    函数读取规则及来源文件摘要，保存结果CSV，并把登记摘要绑定到过程事件；
    登记字典由save_results汇总保存。规则未完成会降低就绪状态，
    但不会覆盖已有计算失败或等待治理状态；同一次运行拒绝重复保存同名结果。
    """
    spec = USES[result_name]
    rule = load_rules()[result_name]
    inputs = {
        str(clean_path(name).relative_to(run_dir())): sha256(clean_path(name))
        for name in spec["sources"]
    }
    if result_name in LOCAL_EVIDENCE:
        task, filename = LOCAL_EVIDENCE[result_name]
        evidence = task_dir(task) / filename
        inputs[str(evidence.relative_to(run_dir()))] = sha256(evidence)
    for dependency in dependencies:
        inputs[dependency["file"]] = sha256(run_dir() / dependency["file"])
    # 先区分计算失败、等待数据与普通基线；规则未完成不能掩盖已有错误。
    if status is None:
        if method == "NONE":
            status = "BASELINE_ONLY"
        elif ready:
            status = "RECORDED"
        else:
            status = "CHECK_FAILED"
    allowed_statuses = {"RECORDED", "BASELINE_ONLY", "QUALITY_PENDING", "CHECK_FAILED"}
    if status not in allowed_statuses:
        raise ValueError("保存结果的状态必须为已生成、普通基线、等待治理或检查失败")
    if status in {"CHECK_FAILED", "QUALITY_PENDING"}:
        ready = False
    if rule["ready"] is not True:
        ready = False
        details += "；任务2规则尚未完成"
        if status == "RECORDED":
            status = "BASELINE_ONLY"
    elif method != rule["protection"] or set(frame) != set(rule["output_fields"].split("|")):
        ready = False
        details += "；实际保护或字段与任务2规则不一致"
        if status == "RECORDED":
            status = "BASELINE_ONLY" if method == "NONE" else "CHECK_FAILED"
    if status == "RECORDED" and not ready:
        status = "CHECK_FAILED"
    if status != "RECORDED":
        ready = False
    path = task_dir(spec["task"]) / "结果文件" / (result_name + ".csv")
    if path.exists():
        raise ValueError("同一次运行不能重复登记同名结果")
    save_csv(path, frame)
    record = {
        "result_name": result_name,
        "task": spec["task"],
        "file": str(path.relative_to(run_dir())),
        "method": method,
        "ready": bool(ready),
        "details": details,
        "protection_config": json.dumps(
            {} if protection_config is None else protection_config,
            ensure_ascii=False, sort_keys=True, allow_nan=False,
        ),
        "sha256": sha256(path),
        "rule_sha256": record_hash(rule),
        "input_sha256": json.dumps(inputs, ensure_ascii=False, sort_keys=True),
        "dependencies": json.dumps(
            [
                {
                    "result_name": item["result_name"],
                    "metadata_sha256": record_hash(item),
                }
                for item in dependencies
            ],
            ensure_ascii=False,
        ),
    }
    record_event(
        spec["task"],
        "保存处理结果",
        result_name,
        path,
        status,
        details,
        method,
        inputs,
        metadata_sha256=record_hash(record),
    )
    return record


def result_stage_status(artifact):
    """读取与登记字典摘要匹配的保存事件，返回状态字符串或None。
    
    输入为一项结果登记；按结果名和登记摘要核对本次日志，
    无匹配事件时返回None，日志损坏时抛出异常；不修改登记或日志。
    """
    expected_hash = record_hash(artifact)
    saved_status = None
    for event in read_events():
        same_result = event["result_name"] == artifact["result_name"]
        same_metadata = event["metadata_sha256"] == expected_hash
        if event["action"] == "保存处理结果" and same_result and same_metadata:
            saved_status = event["status"]
    return saved_status


def save_results(task, rows):
    """把本任务的结果登记字典列表保存为结果登记.csv，返回文件Path。
    
    登记连接实际结果与来源，由程序生成；本函数不另生成业务结果。
    """
    return write_csv(
        task, "结果登记.csv", pd.DataFrame(rows, columns=TABLE_COLUMNS["结果登记.csv"])
    )


def load_results():
    """严格读取登记；异常信息只含文件与类别，供任务5记录失败，不复制业务内容。"""
    results = {}
    for task in [3, 4]:
        location = f"任务{task}/结果登记.csv"
        try:
            frame = read_table(task_dir(task) / "结果登记.csv")
        except (OSError, ValueError, TypeError, KeyError) as error:
            raise ValueError(f"{location}：文件缺失、不可读或CSV格式无效") from error
        if frame.empty or set(frame) != set(TABLE_COLUMNS["结果登记.csv"]):
            raise ValueError(f"{location}：登记为空或表头不完整")
        for row in frame.to_dict("records"):
            name = row["result_name"]
            if (
                name in results
                or name not in USES
                or USES[name]["task"] != task
                or row["task"] != task
                or not isinstance(row["file"], str) or not row["file"]
                or not isinstance(row["method"], str) or not row["method"]
            ):
                raise ValueError(f"{location}：名称重复、任务不一致或必要字段无效")
            results[name] = row
    return results


# 中文表头与状态只用于显示；读回时恢复程序内部字段。
TABLE_COLUMNS = {
    "质量处理.csv": {
        "asset": "来源表",
        "source_row": "原始行号",
        "record_id": "记录编号",
        "field": "字段",
        "before": "原值",
        "after": "处理后值",
        "treatment": "处理方式",
        "needs_review": "是否待复核",
        "reason": "处理原因",
        "review_reason": "复核原因",
    },
    "清洗对比.csv": {
        "asset": "来源表",
        "field": "比较字段",
        "current_mean": "当前处理均值",
        "exclude_invalid_mean": "排除无效观测后的均值",
        "affected_rows": "原始质量异常数量",
        "current_rows": "当前方法样本数",
        "exclude_invalid_rows": "排除无效值方法样本数",
        "current_clipped_rows": "当前方法统计裁剪数量",
        "exclude_invalid_clipped_rows": "排除无效值方法统计裁剪数量",
        "contribution_rule": "统计贡献规则",
        "downstream_use": "下游用途",
    },
    "字段分级.csv": {
        "asset": "来源表",
        "field": "字段",
        "category": "类别",
        "level": "等级",
        "tags": "标签",
        "basis": "判断依据",
    },
    "资产核对.csv": {
        "asset_id": "资产编号",
        "kind": "类型",
        "owner": "持有方",
        "clues": "内容线索",
        "review": "核对结论",
    },
    "文本线索.csv": {
        "asset": "来源表",
        "record_id": "记录编号",
        "field": "文本字段",
        "clues": "内容线索",
        "review": "复核状态",
    },
    "假名检查.csv": {"check": "检查项", "passed": "检查结论", "reason": "原因"},
    "匿名化比较.csv": {
        "strategy": "方案",
        "hours_width": "小时区间宽度",
        "k": "分组规模要求",
        "released_rows": "保留记录数",
        "suppressed_rows": "抑制记录数",
        "retention_rate": "保留比例",
        "minimum_group": "最小组规模",
        "minimum_diversity": "最少标签种类",
        "homogeneous_groups": "同质分组数",
        "evaluation_devices": "共同评价设备数",
        "raw_auc": "原始AUC",
        "protected_auc": "处理后AUC",
        "status": "实现状态",
    },
    "匿名分组.csv": {
        "device_model": "设备型号",
        "region_group": "泛化区域",
        "hours_band": "小时区间",
        "group_size": "组内记录数",
        "distinct_labels": "标签种类数",
    },
    "评价样本.csv": {"device_id": "设备编号"},
    "统计误差.csv": {
        "mechanism": "发布机制",
        "query": "统计查询",
        "total_budget": "总预算",
        "epsilon": "本查询预算",
        "delta": "本查询delta",
        "trial_count": "试验次数",
        "true_value": "课堂真值",
        "mean_absolute_error": "平均绝对误差",
        "error_std": "误差标准差",
        "tolerance": "误差目标",
        "mean_error_within_tolerance": "平均误差是否达标",
        "status": "实现状态",
    },
    "设备划分.csv": {
        "device_id": "设备编号",
        "is_common": "是否共同设备",
        "owner": "记录归属",
        "split": "训练或测试用途",
    },
    "匹配与统计.csv": {
        "check": "检查项",
        "baseline_value": "明文校验值",
        "actual_value": "本次计算值",
        "status": "检查结论",
        "reason": "原因",
    },
    "模型比较.csv": {
        "mode": "模型",
        "training_rows": "训练数量",
        "test_devices": "测试数量",
        "rounds": "训练轮数",
        "auc": "AUC",
        "selected_devices": "优先检查数量",
        "captured_failures": "捕获故障数",
        "total_failures": "测试集故障数",
        "top10_capture_rate": "故障捕获比例",
        "status": "实现状态",
    },
    "操作查询.csv": {
        "query": "查询问题", "scope": "统计范围", "query_status": "查询状态",
        "count": "操作次数", "matched_locations": "命中位置", "reason": "说明",
    },
    "审计链验证.csv": {"test": "检查情形", "outcome": "实际结论", "reason": "检查说明"},
    "审计区块概览.csv": {
        "block_index": "区块编号", "content": "保存内容", "record_count": "审计记录数",
        "timestamp": "时间", "previous_hash": "前序哈希", "block_hash": "本块哈希",
    },
    # 旧文件仍可被明确读取；新运行不输出旧文件，也不自动回查历史运行。
    "审计验证.csv": {"test": "验证操作", "outcome": "实际检查结论", "reason": "原因"},
    "区块链运行记录.csv": {
        "block_index": "区块编号",
        "content": "保存内容",
        "record_count": "审计记录数",
        "timestamp": "时间",
        "previous_hash": "前序哈希",
        "block_hash": "本块哈希",
    },
    "流转与保护规则.csv": {
        "result_name": "结果名称",
        "sources": "来源表",
        "input_fields": "使用字段",
        "output_fields": "允许输出字段",
        "protection": "保护要求",
        "ready": "规则已完成",
        "reason": "理由",
    },
    "结果登记.csv": {
        "result_name": "结果名称",
        "task": "任务",
        "file": "结果文件",
        "method": "实际方法",
        "ready": "阶段检查",
        "details": "处理说明",
        "protection_config": "实际保护配置",
        "sha256": "文件摘要",
        "rule_sha256": "规则摘要",
        "input_sha256": "输入文件及摘要",
        "dependencies": "依赖结果及登记摘要",
    },
    "操作记录.csv": {
        "event_id": "事件序号",
        "task": "任务",
        "action": "操作",
        "result_name": "结果名称",
        "file": "文件",
        "status": "操作结果",
        "reason": "原因",
        "time": "时间",
        "block_index": "所在区块",
        "record_source": "记录来源",
        "chain_status": "链验证状态",
    },
    "结果检查.csv": {
        "result_name": "结果名称",
        "file": "文件",
        "status": "检查状态",
        "reason": "检查说明",
    },
    "结果索引.csv": {
        "task": "任务",
        "result_name": "结果名称",
        "file": "文件",
        "method": "实际方法",
        "status": "操作结果",
        "reason": "原因",
    },
}

LEGACY_TABLE_HEADERS = {
    "结果检查.csv": {"操作结果": "status", "原因": "reason"},
    "操作查询.csv": {"统计结果": "count"},
    "审计链验证.csv": {"验证操作": "test", "实际检查结论": "outcome", "原因": "reason"},
}

LEGACY_VALUE_LABELS = {"test": {"保持原始记录": "ORIGINAL"}}

VALUE_LABELS = {
    "protection": {
        "NONE": "尚未采用目标保护",
        "PSEUDONYM": "用途限定假名",
        "ANONYMIZED": "泛化与分组抑制",
        "DP": "带噪统计",
        "SECURE_SUM_DP": "保护求和后加噪",
        "PSI": "令牌交集匹配",
        "FEDAVG": "联邦平均训练",
    },
    "method": {
        "NONE": "尚未采用目标保护",
        "PSEUDONYM": "用途限定假名",
        "ANONYMIZED": "泛化与分组抑制",
        "DP": "带噪统计",
        "SECURE_SUM_DP": "保护求和后加噪",
        "PSI": "令牌交集匹配",
        "FEDAVG": "联邦平均训练",
    },
    "status": {
        "NOT_IMPLEMENTED": "尚未实现",
        "EVALUATED": "已实际计算",
        "QUALITY_PENDING": "等待任务1数据治理",
        "BASELINE": "原始基线",
        "CHECKED": "已核对",
        "RECORDED": "已保存",
        "CHECK_PASSED": "检查通过",
        "CHECK_FAILED": "检查失败",
        "BASELINE_ONLY": "仅基线，保护未完成",
        "NOT_RUN": "未生成结果",
    },
    "query_status": {
        "COMPLETED": "已完成",
        "ANSWER_PENDING": "查询作答未完成",
        "WAITING_CHAIN": "等待验链",
        "NOT_EXECUTED": "未执行",
        "QUERY_FAILED": "查询执行失败",
        "NO_EVENTS": "数据不足",
    },
    "chain_status": {
        "VERIFIED": "已通过查询前检查",
        "PENDING": "等待14-A验链",
        "FAILED": "原链验证失败",
        "READ_FAILED": "链文件读取失败",
        "EVENT_FAILED": "链内事件检查失败",
    },
    "treatment": {
        "DUPLICATE_REMOVED": "删除重复行",
        "ESTIMATED": "估计替代",
        "INVALID_REMOVED": "无效值置空",
        "NORMALIZED": "格式规范",
        "UNCHANGED": "保持原值",
    },
    "split": {"TRAIN": "训练集", "TEST": "测试集", "NOT_TRAINED": "尚未训练"},
    "mode": {
        "CENTRALIZED": "集中模型",
        "CN_LOCAL": "华辰本地模型",
        "DE_LOCAL": "德维本地模型",
        "FEDAVG": "联邦平均模型",
    },
    "query": {
        "confirmed_case_count": "确认故障案例数（例）",
        "mean_downtime": "平均停机时长（小时）",
    },
    "check": {
        "SAME_INPUT_STABLE": "同用途重复计算是否稳定",
        "DOMAIN_SEPARATED": "不同用途是否隔离",
        "NO_TOKEN_COLLISION": "当前输入是否无令牌碰撞",
        "MINIMIZED_FIELDS": "是否只保留必要字段",
    },
    "test": {
        "ORIGINAL": "原始保存链",
        "MODIFY": "修改一条记录",
        "DELETE": "删除块内记录",
        "REORDER": "重排记录或业务区块",
        "DROP_LAST_BLOCK": "删除末尾区块",
        "EMPTY": "使用空链",
        "REWRITE_AND_REHASH": "重写整链并重算摘要",
    },
    "category": {"UNASSESSED": "尚未分类"},
    "level": {"UNASSESSED": "尚未分级"},
    "tags": {
        "NONE": "尚未标注",
        "IDENTIFIER": "标识信息",
        "QUASI_IDENTIFIER": "组合识别线索",
        "FEATURE": "预测特征",
        "SENSITIVE": "敏感业务信息",
        "METADATA": "登记信息",
        "FREE_TEXT": "自由文本",
    },
    "clues": {
        "FIELD_REVIEW": "需核对字段",
        "NOT_SCANNED": "尚未扫描",
        "见field_catalog": "见字段分级.csv",
        "EMAIL": "邮箱线索",
        "EMPLOYEE": "工号线索",
        "NAMED_CONTACT": "姓名线索",
        "PHONE": "电话线索",
        "LOCATION": "位置线索",
        "": "未命中规则",
    },
    "review": {
        "UNASSESSED": "尚未核对",
        "FIELD_REVIEW": "按字段分级核对",
        "REVIEW_REQUIRED": "需要复核",
        "CONSISTENT": "声明与线索一致",
        "NOT_DETECTED": "规则未命中，不能据此排除隐私信息",
    },
    "kind": {
        "CSV": "数据表",
        "EMAIL": "邮件",
        "LOG": "日志",
        "REPORT": "报告",
        "CONTRACT": "合同",
    },
}

BOOL_LABELS = {
    "ready": ("未完成或未通过", "已完成且通过", "尚未检查"),
    "needs_review": ("否", "是", "未记录"),
    "is_common": ("否", "是", "未记录"),
    "passed": ("未通过", "通过", "尚未实现"),
    "mean_error_within_tolerance": ("未达标", "达标", "尚未计算"),
}


# 以下函数属于根入口的只读证据汇总；不重新执行任务5的业务核验。
def _evidence_text(value):
    """缺失单元格显示为空，不把 NaN 当成真实原因或文件名。"""
    return "" if pd.isna(value) else str(value)


def read_result_evidence(folder, executed):
    """读取本次已执行阶段的登记、保存事件和有效核验行。

    检查表允许部分有效：重复、未知状态等问题逐用途保留，不能字典覆盖。
    缺失核验行与不可读取的核验表分别记录，供概览表达尚未核验的范围。
    """
    folder = Path(folder)
    executed = set(executed)
    evidence = {
        "artifacts": {}, "checks": {}, "registry_errors": {},
        "registry_missing": [], "registry_empty": [],
        "check_error": "", "check_errors": {}, "check_missing": [],
        "event_error": "", "event_errors": {}, "stage_status": {},
    }
    for task in (3, 4):
        if task not in executed:
            continue
        path = folder / TASKS[task] / "结果登记.csv"
        if not path.exists():
            evidence["registry_missing"].append(task)
            continue
        try:
            frame = read_table(path)
            required = set(TABLE_COLUMNS[path.name])
            missing = required - set(frame)
            unexpected = set(frame) - required
            if missing:
                raise ValueError("必要列缺失：" + "、".join(sorted(missing)))
            if unexpected:
                raise ValueError("存在重复或无法识别的表头：" + "、".join(sorted(unexpected)))
            if frame.result_name.isna().any() or frame.result_name.duplicated().any():
                raise ValueError("用途名称缺失或重复")
            if frame.empty:
                evidence["registry_empty"].append(task)
            rows = frame.to_dict("records")
            for row in rows:
                name = row["result_name"]
                if name not in USES or USES[name]["task"] != task or row["task"] != task:
                    raise ValueError(f"用途与任务不对应：{name!r}，任务值 {row['task']!r}")
                if not isinstance(row["file"], str) or not row["file"]:
                    raise ValueError(f"{name}的结果文件路径缺失")
                relative = Path(row["file"])
                if relative.is_absolute() or not (folder / relative).resolve().is_relative_to(folder.resolve()):
                    raise ValueError(f"{name}的结果文件路径不属于本次运行")
                if row["method"] not in VALUE_LABELS["method"]:
                    raise ValueError(f"{name}的实际方法无法识别：{row['method']!r}")
            evidence["artifacts"].update({row["result_name"]: row for row in rows})
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
            evidence["registry_errors"][task] = f"任务{task}结果登记无法读取或结构无效：{error}；查看任务日志"

    if any(task in executed for task in (3, 4)):
        try:
            events = read_events(folder / "过程记录.jsonl")
            if not events:
                raise ValueError("过程记录.jsonl缺失或没有有效事件")
            saved = {}
            for event in events:
                if event["action"] != "保存处理结果" or event["task"] not in executed:
                    continue
                name = event["result_name"]
                if name not in USES or event["task"] != USES[name]["task"]:
                    raise ValueError(f"保存事件的用途与任务不对应：{name!r}")
                saved.setdefault(name, []).append(event)
            allowed = {"RECORDED", "BASELINE_ONLY", "QUALITY_PENDING", "CHECK_FAILED"}
            for name, records in saved.items():
                if len(records) != 1:
                    evidence["event_errors"][name] = "本结果存在重复保存事件，无法唯一确认生成阶段状态"
                    continue
                event = records[0]
                if event["status"] not in allowed:
                    evidence["event_errors"][name] = f"保存事件状态无法识别：{event['status']!r}"
                    continue
                artifact = evidence["artifacts"].get(name)
                if artifact and (
                    event["file"] != artifact["file"]
                    or event["method"] != artifact["method"]
                    or event["metadata_sha256"] != record_hash(artifact)
                ):
                    evidence["event_errors"][name] = "保存事件与当前结果登记不对应，无法确认生成阶段状态"
                    continue
                evidence["stage_status"][name] = event["status"]
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
            evidence["event_error"] = f"过程记录缺失或不可读，无法确认生成阶段状态：{error}"
            evidence["stage_status"].clear()

    if 5 in executed:
        path = folder / TASKS[5] / "结果检查.csv"
        try:
            frame = read_table(path)
            required = set(TABLE_COLUMNS[path.name])
            missing = required - set(frame)
            unexpected = set(frame) - required
            if missing:
                raise ValueError("必要列缺失：" + "、".join(sorted(missing)))
            if unexpected:
                raise ValueError("存在重复或无法识别的表头：" + "、".join(sorted(unexpected)))
            # CHECK_PASSED 只能来自有效检查；RECORDED 属于生成阶段，不能当作核验。
            allowed = {"CHECK_PASSED", "CHECK_FAILED", "BASELINE_ONLY", "QUALITY_PENDING", "NOT_RUN"}
            rows_by_name = {}
            problems = []
            for row in frame.to_dict("records"):
                name = row["result_name"]
                if not isinstance(name, str) or name not in USES:
                    problems.append(f"存在无法识别的用途名称：{name!r}")
                    continue
                rows_by_name.setdefault(name, []).append(row)
            for name, rows in rows_by_name.items():
                if len(rows) != 1:
                    evidence["check_errors"][name] = "结果检查中本用途名称重复，无法唯一确定核验结论"
                    continue
                row = rows[0]
                status = row["status"]
                if not isinstance(status, str) or status not in allowed:
                    evidence["check_errors"][name] = f"结果检查状态无法识别：{status!r}"
                    continue
                reason = _evidence_text(row["reason"])
                if not reason.strip():
                    evidence["check_errors"][name] = "结果检查缺少原因，无法确认核验结论"
                    continue
                evidence["checks"][name] = row
            if problems:
                evidence["check_error"] = "结果检查证据部分无效：" + "；".join(problems)
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
            evidence["check_error"] = f"任务5结果检查证据缺失或无法读取：{error}"
        evidence["check_missing"] = [
            name for name, spec in USES.items()
            if spec["task"] in executed
            and name not in evidence["checks"] and name not in evidence["check_errors"]
        ]
    else:
        evidence["check_missing"] = [name for name, spec in USES.items() if spec["task"] in executed]
    return evidence


def collect_result_index(folder, executed, evidence=None):
    """根据同一份证据构建原有六列索引；本函数不写文件。

    有效任务5结果优先；尚未核验时使用实际保存事件。未选择的用途不入表。
    """
    folder = Path(folder)
    executed = set(executed)
    if evidence is None:
        evidence = read_result_evidence(folder, executed)
    rows = []
    for task, name in ((1, "质量处理"), (1, "字段分级"), (2, "流转与保护规则")):
        path = folder / TASKS[task] / (name + ".csv")
        if task in executed and path.is_file():
            rows.append({
                "task": task, "result_name": name, "file": str(path.relative_to(folder)),
                "method": "", "status": "RECORDED", "reason": "查看表内实际判断与未完成项",
            })
    for name, spec in USES.items():
        task = spec["task"]
        if task not in executed:
            continue
        artifact = evidence["artifacts"].get(name, {})
        check = evidence["checks"].get(name, {})
        if check:
            status, reason = check["status"], _evidence_text(check["reason"])
        elif name in evidence["check_errors"]:
            status, reason = "CHECK_FAILED", evidence["check_errors"][name]
        elif evidence["check_error"]:
            status, reason = "CHECK_FAILED", evidence["check_error"]
        elif task in evidence["registry_errors"]:
            status, reason = "CHECK_FAILED", evidence["registry_errors"][task]
        elif not artifact:
            status, reason = "NOT_RUN", "本次未生成已登记结果；查看本任务日志"
        elif evidence["event_error"]:
            status, reason = "CHECK_FAILED", evidence["event_error"]
        elif name in evidence["event_errors"]:
            status, reason = "CHECK_FAILED", evidence["event_errors"][name]
        elif name not in evidence["stage_status"]:
            status, reason = "CHECK_FAILED", "缺少本结果的保存事件，无法确认生成阶段状态"
        else:
            status, reason = evidence["stage_status"][name], _evidence_text(artifact.get("details", ""))
        rows.append({
            "task": task, "result_name": name,
            "file": _evidence_text(artifact.get("file", check.get("file", ""))),
            "method": _evidence_text(artifact.get("method", "NONE")),
            "status": status, "reason": reason,
        })
    return pd.DataFrame(rows, columns=list(TABLE_COLUMNS["结果索引.csv"]))


def save_result_index(folder, executed):
    """保存并返回关键结果索引，保留既有列与中文导出约定。"""
    result_index = collect_result_index(folder, executed)
    save_csv(Path(folder) / "结果索引.csv", result_index)
    return result_index


# ===== 项目根入口：运行、交接与概览（已提供，无学生作答） =====
# 此区只组织任务和读取其真实产出；业务保存事件仍由各任务登记。
PROJECT_STAGES = {
    1: {
        "purpose": "治理双方设备表和服务案例表，盘点字段信息。",
        "handoff": "清洗表供任务2、3、4使用；字段分级供任务2设计用途规则。",
        "evidence": ["质量处理.csv", "字段分级.csv", "清洗对比.csv", "文本线索.csv"],
    },
    2: {
        "purpose": "把六项用途转成必要输入、允许输出与保护规则。",
        "handoff": "用途规则交给任务3、4执行，交给任务5核验。",
        "evidence": ["流转与保护规则.csv"],
    },
    3: {
        "purpose": "按德维需求生成内部服务分析、研究数据和服务统计。",
        "handoff": "三类共享结果及处理证据交给任务5核验。",
        "evidence": ["结果登记.csv", "假名检查.csv", "匿名化方案.json", "匿名化比较.csv", "统计误差.csv"],
    },
    4: {
        "purpose": "匹配共同设备并确定归属，再汇总告警和比较故障模型。",
        "handoff": "使用任务1、2的结果；三类联合分析结果及证据交给任务5。",
        "evidence": ["结果登记.csv", "匹配与统计.csv", "设备划分.csv", "模型比较.csv"],
    },
    5: {
        "purpose": "核对六类业务结果，再读回并检查已保存的审计链，查询通过检查的链内事件。",
        "handoff": "结果检查和过程证据支持项目报告的整体追溯。",
        "evidence": ["结果检查.csv", "操作查询.csv", "审计链验证.csv", "操作记录.csv"],
        "auxiliary_evidence": ["辅助材料/审计区块概览.csv", "辅助材料/审计原始记录.jsonl"],
    },
}
USE_EVIDENCE_FILES = {
    "内部服务分析": ["假名检查.csv"],
    "研究数据": ["匿名化方案.json", "匿名化比较.csv"],
    "服务统计": ["统计误差.csv"],
    "共同设备匹配": ["匹配与统计.csv"],
    "联合告警统计": ["设备划分.csv", "匹配与统计.csv"],
    "联合故障模型": ["设备划分.csv", "模型比较.csv"],
}
EXECUTION_LABELS = {
    "RUNNING": "正在执行", "EXITED": "正常结束", "FAILED": "异常退出",
    "START_FAILED": "启动失败", "INTERRUPTED": "中断",
}


class StageExecutionError(RuntimeError):
    """阶段执行异常；保留原退出码，日志路径写在错误说明中。"""

    def __init__(self, message, exit_code=1):
        super().__init__(message)
        self.exit_code = exit_code if exit_code > 0 else 128 - exit_code


def project_time():
    """返回带UTC时区的实际时间，不是业务观察期。"""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def create_project_run(args, sequence):
    """创建一个新运行目录并返回上下文；不覆盖现有目录。"""
    started = datetime.now(timezone.utc)
    folder = (args.output_dir or ROOT / "runs" / started.strftime("RUN-%Y%m%dT%H%M%S%fZ")).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    return {
        "folder": folder, "version": ROOT.name,
        "started_at": started.isoformat(timespec="microseconds"), "finished_at": None,
        "timezone": "UTC", "observation_month": OBSERVATION_MONTH,
        "prediction_time": PREDICTION_TIME, "label_end_exclusive": LABEL_END_EXCLUSIVE,
        "requested_task": args.task, "planned_tasks": list(sequence), "stages": [],
        "scope": "本地教学：结果检查与过程审计", "verbose": args.verbose,
        "failure": None, "summary_errors": [], "exit_code": None,
    }


def save_project_run(context):
    """保存计划、实际阶段和收尾信息，不向业务过程日志添加事件。"""
    data = {key: value for key, value in context.items() if key not in {"folder", "verbose"}}
    data["run_dir"] = str(context["folder"])
    (context["folder"] / "run.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def show_project_context(context):
    """展示合作目标、业务依赖和本次实际执行范围。"""
    print(f"《数据安全系统设计与实践》课程项目说明 · {context['version']}", flush=True)
    print("华辰需要提前发现故障；德维需要案例关联、设备研究和服务统计。", flush=True)
    print("双方还需匹配共同设备、联合统计告警并比较故障模型，共六项用途。", flush=True)
    print("业务依赖：1 → 2 →〔3、4两条分支〕→ 5；任务3、4都读取任务1清洗表和任务2规则。", flush=True)
    print("本次执行：" + " → ".join(map(str, context["planned_tasks"])), flush=True)
    print(f"本次目录：{context['folder']}", flush=True)
    print("14道代码TODO、27处作答框；策略说明和结果分析写入报告。", flush=True)


def stop_stage_process(process):
    """中断时终止并回收子进程；POSIX同时终止该任务创建的进程组。"""
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()
        process.wait()


def run_stage(number, context, records):
    """执行一个阶段、追加真实记录并显示交接摘要；失败先记录再抛出。

    子进程直接写本阶段日志，避免中断时丢失已输出内容。只对业务进程的
    正常返回记录实际退出码；启动失败或中断用null及独立执行状态表示。
    """
    folder = context["folder"]
    step = context["planned_tasks"].index(number) + 1
    print(f"\n第{step}/{len(context['planned_tasks'])}步 · 任务{number} {TASKS[number][3:]}", flush=True)
    print(PROJECT_STAGES[number]["purpose"], flush=True)
    relative_log = Path("运行日志") / (TASKS[number] + ".log")
    record = {
        "task": number, "exit_code": None, "execution_status": "RUNNING",
        "started_at": project_time(), "finished_at": None, "log": str(relative_log),
    }
    records.append(record)
    context["stages"] = records
    process = None
    try:
        save_project_run(context)
        (folder / "运行日志").mkdir(exist_ok=True)
        with (folder / relative_log).open("w", encoding="utf-8") as log:
            entry = ROOT / TASKS[number] / "main.py"
            process = subprocess.Popen(
                [sys.executable, str(entry)], cwd=entry.parent,
                # 子进程直接写文件描述符，不会继承父进程open的编码。
                # 显式统一UTF-8，避免Windows等默认编码环境生成乱码日志。
                env={**os.environ, "COURSE_RUN_DIR": str(folder), "PYTHONUNBUFFERED": "1",
                     "PYTHONIOENCODING": "utf-8"},
                stdout=log, stderr=subprocess.STDOUT, start_new_session=(os.name == "posix"),
            )
            return_code = process.wait()
        record["exit_code"] = return_code
        record["execution_status"] = "EXITED" if return_code == 0 else "FAILED"
        if return_code:
            raise StageExecutionError(f"任务{number}异常退出（{return_code}）；查看 {relative_log}", return_code)
    except KeyboardInterrupt:
        record["execution_status"] = "INTERRUPTED"
        if process is not None:
            stop_stage_process(process)
        record["exit_code"] = None
        raise
    except StageExecutionError:
        raise
    except Exception as error:
        record["execution_status"] = "START_FAILED" if process is None else "FAILED"
        if process is not None and process.poll() is None:
            stop_stage_process(process)
        record["error"] = f"{type(error).__name__}: {error}"
        raise StageExecutionError(f"任务{number}执行或运行记录失败：{error}；日志 {relative_log}") from error
    finally:
        record["finished_at"] = project_time()
        # 即使记录保存失败也保留内存中的原始阶段故障，交给最终收尾再次保存。
        try:
            save_project_run(context)
        except Exception as error:
            context["summary_errors"].append(f"任务{number}运行记录保存失败：{error}")
        if (context["verbose"] or record["execution_status"] != "EXITED") and (folder / relative_log).is_file():
            try:
                print((folder / relative_log).read_text(encoding="utf-8", errors="replace"), end="", flush=True)
            except OSError as error:
                print(f"阶段日志无法回显：{error}", flush=True)
    for line in read_stage_summary(folder, number):
        print(line, flush=True)
    print("交接：" + PROJECT_STAGES[number]["handoff"], flush=True)


def project_text(value, missing="无记录"):
    """展示CSV标量；缺失不转换成0或字符串nan。"""
    return missing if value is None or pd.isna(value) else str(value)


def project_status(status):
    """用既有中文标签展示业务状态，保留未知值供定位。"""
    value = project_text(status)
    return VALUE_LABELS["status"].get(value, f"无法识别状态：{value}")


def read_process_summary(directory):
    """分别读取本次原链结论、查询前检查和查询状态，不推断业务是否完成。

    三种证据独立读取，一张表损坏时仍显示其他实际结果；只读取新文件名。
    """
    directory = Path(directory)
    lines = []
    try:
        audit = read_table(directory / "审计链验证.csv")
        if not {"test", "outcome", "reason"} <= set(audit):
            raise ValueError("审计链验证表缺少检查情形、实际结论或检查说明")
        originals = audit.loc[audit.test.eq("ORIGINAL")]
        if len(originals) != 1:
            raise ValueError("原始保存链结论缺失或重复")
        original = originals.iloc[0]
        lines.append(f"原链结构检查：{project_text(original.outcome)}；{project_text(original.reason)}。")
        pending = int(audit.outcome.eq("尚未实现").sum())
        if pending:
            lines.append(f"审计链验证：{pending}种情形尚未实现，请完成14-A；业务结果核验不能代替验链。")
        lines.append("局部变化、删尾和整链重算的实际结论及边界见审计链验证.csv。")
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        lines.append(f"原链检查证据无法读取：{error}。")
    try:
        operations = read_table(directory / "操作记录.csv")
        if "chain_status" not in operations:
            raise ValueError("操作记录表缺少链验证状态")
        states = operations.chain_status.map(lambda value: project_text(value)).drop_duplicates()
        if len(states):
            labels = VALUE_LABELS["chain_status"]
            lines.append("查询前检查状态：" + "；".join(labels.get(state, f"无法识别状态：{state}") for state in states) + "。")
        else:
            lines.append("操作明细无事件，不能据此确认查询前检查已经通过。")
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        lines.append(f"查询前检查状态无法读取：{error}。")
    try:
        queries = read_table(directory / "操作查询.csv")
        if not {"query", "scope", "query_status", "count", "matched_locations", "reason"} <= set(queries):
            raise ValueError("操作查询表缺少问题、范围、状态、次数、命中位置或说明")
        if queries.empty:
            raise ValueError("操作查询表没有查询行，不能确认任何查询已执行")
        counts = queries.query_status.map(lambda value: project_text(value)).value_counts(sort=False)
        labels = VALUE_LABELS["query_status"]
        lines.append("查询执行状态：" + "；".join(f"{labels.get(state, '无法识别状态：' + state)}{count}项" for state, count in counts.items()) + "。")
        for row in queries.itertuples():
            if row.query_status != "COMPLETED":
                lines.append(f"查询“{project_text(row.query)}”：{project_text(row.reason)}；统计范围：{project_text(row.scope)}。")
            elif pd.isna(row.count) or not isinstance(row.count, Real) or not math.isfinite(row.count) or row.count < 0 or row.count != int(row.count):
                lines.append(f"查询“{project_text(row.query)}”的次数缺失或无效，不能视为已完成统计。")
        lines.append("查询未执行或未完成时次数留空；实际零次只表示本次查询范围内未命中。")
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        lines.append(f"查询执行证据无法读取：{error}。")
    return lines


def read_stage_summary(folder, number):
    """读取本阶段的真实产出，不重新处理数据或判断学生答案。"""
    directory = folder / TASKS[number]
    lines = []
    try:
        if number == 1:
            counts = []
            for asset in ASSETS:
                path = directory / "清洗数据" / asset
                if path.is_file():
                    unit = "案例记录" if asset == "service_cases.csv" else "设备记录"
                    counts.append(f"{asset}：{len(read_table(path))}条{unit}")
                else:
                    counts.append(f"{asset}：未生成")
            lines.append("清洗表：" + "；".join(counts) + "。记录数不表示全部治理合格。")
            evidence = read_table(directory / "质量处理.csv")
            pending = evidence.needs_review.map(lambda value: value is True).sum()
            unknown = sum(not isinstance(value, bool) for value in evidence.needs_review)
            lines.append(f"质量处理表：{int(pending)}条明确待复核证据；同一设备或案例可对应多条。")
            if unknown:
                lines.append(f"另有{unknown}条证据的复核状态缺失或无法识别，不能计为无需复核。")
        elif number == 2:
            rules = read_table(directory / "流转与保护规则.csv")
            if not {"result_name", "ready"} <= set(rules) or rules.result_name.duplicated().any() or set(rules.result_name) != set(USES):
                raise ValueError("六项用途规则名称缺失、重复或不一致")
            ready = sum(value is True for value in rules.ready)
            unknown = sum(not isinstance(value, bool) for value in rules.ready)
            lines.append(f"规则明确就绪：{ready}/6项；仅表示规则配置状态，见流转与保护规则.csv。")
            if unknown:
                lines.append(f"另有{unknown}项规则就绪值缺失或无法识别。")
        elif number in {3, 4}:
            # 阶段摘要始终显示生成状态；最终核验结论在整体汇总中展示。
            index = collect_result_index(folder, [1, 2, number])
            rows = index.loc[index.task == number]
            lines.append("生成结果：" + "；".join(f"{row.result_name}：{project_status(row.status)}" for row in rows.itertuples()) + "。")
        else:
            evidence = read_result_evidence(folder, [1, 2, 3, 4, 5])
            checks = evidence["checks"]
            passed = sum(row["status"] == "CHECK_PASSED" for row in checks.values())
            lines.append(f"本次有效检查记录：{len(checks)}/6项；其中检查通过{passed}项。")
            counts = {}
            for check in checks.values():
                counts[check["status"]] = counts.get(check["status"], 0) + 1
            if counts:
                lines.append("检查记录状态：" + "；".join(f"{project_status(status)}{count}项" for status, count in counts.items()) + "。")
            if evidence["check_error"]:
                lines.append(evidence["check_error"])
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        lines.append(f"阶段摘要证据无法读取：{error}；查看 {TASKS[number]} 的原始文件和运行日志。")
    if number == 5:
        lines.extend(read_process_summary(directory))
        lines.append("六类业务结果通过只表示已执行的内容检查通过；还需分别核对模型聚合、原链检查及查询作答。")
    return lines


def project_execution(number, context, records):
    """说明本次是否选择、是否执行；与业务结果状态分开。"""
    if number not in context["planned_tasks"]:
        return "本次未选择"
    record = next((row for row in records if row["task"] == number), None)
    if record is None:
        return "此前运行异常，本阶段未启动" if context["failure"] else "本阶段尚未启动"
    return EXECUTION_LABELS.get(record["execution_status"], "执行状态无法识别")


def build_project_summary(context, records, index):
    """复用索引和真实检查记录构建一次汇总，供终端和Markdown共同展示。"""
    folder = context["folder"]
    executed = [row["task"] for row in records]
    evidence = read_result_evidence(folder, executed)
    indexed = {row["result_name"]: row for row in index.to_dict("records")}
    uses = []
    tips = []
    for name, spec in USES.items():
        chosen = spec["task"] in context["planned_tasks"]
        attempted = spec["task"] in executed
        row = indexed.get(name, {}) if attempted else {}
        check = evidence["checks"].get(name)
        if not attempted:
            verification = "未安排核验" if not chosen else "所属阶段尚未启动"
        elif not row:
            verification = "索引缺失，本次汇总无法确认结果状态"
        elif check:
            verification = "任务5已有本次有效检查记录"
        elif name in evidence.get("check_errors", {}):
            verification = evidence["check_errors"][name]
        elif evidence["check_error"]:
            verification = evidence["check_error"]
        else:
            verification = "尚未核验：任务5未执行" if 5 not in executed else "尚未核验：无本用途的本次检查记录"
        status = row.get("status")
        reason = project_text(row.get("reason"), "本次无生成记录")
        if attempted and not row:
            reason = "结果索引未能取得本用途记录，查看收尾问题"
        uses.append({
            "name": name, "task": spec["task"], "need": spec["need"], "sources": spec["sources"],
            "selected": chosen, "attempted": attempted,
            "execution": project_execution(spec["task"], context, records),
            "status": status, "reason": reason, "file": project_text(row.get("file"), ""),
            "method": project_text(row.get("method"), "无记录"),
            "verification": verification, "checked": bool(attempted and row and check),
        })
        if attempted:
            if status == "QUALITY_PENDING":
                tips.append(f"{name}：先回查任务1质量处理.csv和清洗对比.csv；{reason}")
            elif status == "BASELINE_ONLY":
                tips.append(f"{name}：回查任务2同名用途规则及任务{spec['task']}处理说明；{reason}")
            elif status in {"CHECK_FAILED", "NOT_RUN"}:
                tips.append(f"{name}：回查同名检查、结果登记和阶段日志；{reason}")
            elif not check:
                tips.append(f"{name}：{verification}；完成前序后运行 python main.py --task 5 进行完整核验。")
    return {
        "folder": folder, "context": context, "stages": records, "uses": uses, "index": index,
        "stage_summaries": {row["task"]: read_stage_summary(folder, row["task"]) for row in records},
        "tips": tips, "exit_code": context["exit_code"], "overview_saved": False, "index_saved": False,
    }


def project_markdown(value):
    """转义表格与Markdown控制字符，保持来源文字可读。"""
    text = project_text(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    for symbol in ("\\", "`", "*", "_", "[", "]", "|", "#"):
        text = text.replace(symbol, "\\" + symbol)
    return text.replace("\r", "").replace("\n", "<br>")


def project_file_reference(folder, path, label=None, source=False):
    """列出证据名称与文件路径，不生成超链接；不采信越界结果路径。"""
    path = Path(path)
    path = path if path.is_absolute() else folder / path
    resolved = path.resolve()
    label = project_markdown(label or path.name)
    allowed_source = source and resolved in {raw_path(asset).resolve() for asset in ASSETS}
    if not resolved.is_relative_to(folder.resolve()) and not allowed_source:
        return f"{label}（路径无效或超出本次运行目录）"
    if not path.is_file():
        expected = os.path.relpath(path, folder).replace(os.sep, "/")
        return f"{label}（未生成或文件缺失；预期：{project_markdown(expected)}）"
    try:
        relative = os.path.relpath(path, folder).replace(os.sep, "/")
    except ValueError:
        # Windows允许把结果保存在其他盘；跨盘原始来源不能构造相对路径。
        return f"{label}（原始输入：{project_markdown(resolved)}）"
    return f"{label}（路径：{project_markdown(relative)}）"


def save_project_overview(summary):
    """保存可回查的项目概览；事实来自本次运行，不代写学生报告分析。"""
    folder, context = summary["folder"], summary["context"]
    lines = [
        "# 项目运行概览", "", f"版本：{project_markdown(context['version'])}。本次目录：{project_markdown(folder)}。",
        f"请求任务：{context['requested_task'] or '全部'}；计划任务：{' → '.join(map(str, context['planned_tasks']))}。",
        f"实际尝试：{' → '.join(str(row['task']) for row in summary['stages']) or '无'}；是否启动及返回情况见各阶段执行状态。",
        f"开始：{context['started_at']}；结束：{context['finished_at']}；时间采用UTC。",
        f"业务观察期：{context['observation_month']}；预测时点：{context['prediction_time']}；标签结束日（不含）：{context['label_end_exclusive']}。",
        "", "华辰需要提前发现故障，德维需要案例关联、设备研究和服务统计；双方还需匹配共同设备、联合统计告警并比较故障模型。",
        "业务依赖：任务1 → 任务2 →〔任务3、任务4两条分支〕→ 任务5。任务3、4都直接读取任务1清洗表和任务2规则。", "",
        "本概览汇总实际记录并导航证据。规则就绪、文件保存、业务检查、原链检查和查询执行是不同阶段；这些状态不等于作业完成率或评分。",
    ]
    if context["failure"]:
        lines.extend(["", "运行异常：" + project_markdown(context["failure"])])
    if context["summary_errors"]:
        lines.extend(["", "收尾问题：", ""] + ["- " + project_markdown(error) for error in context["summary_errors"]])
    lines.extend(["", "## 任务交接与本次执行", ""])
    for number, info in PROJECT_STAGES.items():
        lines.extend([
            f"### 任务{number}：{TASKS[number][3:]}", "",
            f"本次执行：{project_execution(number, context, summary['stages'])}。{info['purpose']}",
            "交接：" + info["handoff"],
        ])
        for line in summary["stage_summaries"].get(number, []):
            lines.append(project_markdown(line))
        if number in summary["stage_summaries"]:
            lines.append("证据：" + "；".join(project_file_reference(folder, Path(TASKS[number]) / name) for name in info["evidence"]))
            if info.get("auxiliary_evidence"):
                lines.append("辅助材料（按需回查）：" + "；".join(project_file_reference(folder, Path(TASKS[number]) / name) for name in info["auxiliary_evidence"]))
                diagnostic = Path(TASKS[number]) / "辅助材料/失败过程记录.jsonl"
                if (folder / diagnostic).is_file():
                    lines.append("本次失败记录：" + project_file_reference(folder, diagnostic))
            record = next(row for row in summary["stages"] if row["task"] == number)
            lines.append("阶段日志：" + project_file_reference(folder, record["log"]))
        lines.append("")
    lines.extend(["## 六项用途与当前状态", ""])
    selected = [use for use in summary["uses"] if use["selected"]]
    checked = [use for use in selected if use["checked"]]
    if not selected:
        if any(number in context["planned_tasks"] for number in (3, 4)):
            lines.append("本次已安排业务结果生成，但项目结果无法汇总；请回查收尾问题与实际任务文件。")
        else:
            lines.append("本次尚未安排业务结果生成；优先查看上面的治理或用途规则证据。")
    elif any(use["attempted"] and use["status"] is None for use in selected):
        lines.append("结果索引缺失，无法汇总本次用途核验状态；请回查收尾问题和任务5已有检查证据。")
    elif checked:
        passed = sum(use["status"] == "CHECK_PASSED" for use in checked)
        lines.append(f"本次已核验结果中，检查通过 {passed}/{len(checked)} 项；本次范围内另有 {len(selected) - len(checked)} 项无有效检查记录。")
        lines.append("上述数量仅统计六类业务结果；任务5原链和查询状态见上方任务交接，不能据此认定作业全部完成。")
    else:
        lines.append(f"本次范围内有 {len(selected)} 项用途，尚无有效的任务5检查记录。")
    lines.extend(["", "| 用途 | 本次范围与执行情况 | 结果状态 | 主要原因 | 结果文件 |", "|---|---|---|---|---|"])
    for use in summary["uses"]:
        state = project_status(use["status"]) if use["status"] else "无本次结果状态"
        reason = use["reason"] if use["attempted"] else "本次未选择" if not use["selected"] else "所属阶段未启动"
        result_link = project_file_reference(folder, use["file"]) if use["file"] else "无本次已登记文件"
        lines.append("| " + " | ".join(project_markdown(item) for item in [use["name"], use["execution"], state, reason]) + " | " + result_link + " |")
    for use in summary["uses"]:
        lines.extend(["", "### " + use["name"], "", "用途：" + use["need"]])
        if not use["attempted"]:
            lines.append("本次情况：" + use["execution"] + "。不计入本次结果核验分母。")
            continue
        method = VALUE_LABELS["method"].get(use["method"], use["method"])
        lines.append("实际方法：" + project_markdown(method) + "。核验情况：" + project_markdown(use["verification"]) + "。")
        lines.append("结果文件：" + (project_file_reference(folder, use["file"]) if use["file"] else "无已登记结果文件"))
        lines.append("原始来源：" + "；".join(project_file_reference(folder, raw_path(asset), asset, source=True) for asset in use["sources"]))
        lines.append("清洗来源：" + "；".join(project_file_reference(folder, Path(TASKS[1]) / "清洗数据" / asset, asset) for asset in use["sources"]))
        paths = [(1, "质量处理.csv"), (1, "字段分级.csv"), (2, "流转与保护规则.csv"), (use["task"], "结果登记.csv")]
        paths += [(use["task"], name) for name in USE_EVIDENCE_FILES[use["name"]]]
        if use["name"] in {"联合告警统计", "联合故障模型"}:
            paths.append((4, "结果文件/共同设备匹配.csv"))
        if 5 in [row["task"] for row in summary["stages"]]:
            paths += [(5, "结果检查.csv"), (5, "操作记录.csv")]
        lines.append("规则、处理与核验：" + "；".join(project_file_reference(folder, Path(TASKS[number]) / name) for number, name in paths))
        lines.append("在规则、登记和结果检查表按“结果名称”定位本用途；在质量表按“来源表、记录编号”回查。列出路径只表示文件存在，来源对应结论以实际核验为准。")
    lines.extend(["", "## 优先回查", ""])
    lines.extend("- " + project_markdown(tip) for tip in summary["tips"])
    if not summary["tips"]:
        lines.append("按本次任务范围查看上面的实际证据，并在报告中解释方法、结果和局限。")
    lines.extend(["", "修改前序程序后，重新运行受影响流程；综合报告引用最后一次完整运行，不拼接不同运行的记录。", "",
                  "根索引：" + (project_file_reference(folder, "结果索引.csv") if summary["index_saved"] else "本次未能保存，查看收尾问题"),
                  "运行信息：" + project_file_reference(folder, "run.json"),
                  "实际业务事件：" + project_file_reference(folder, "过程记录.jsonl"), ""])
    path = folder / "项目运行概览.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def finish_project_run(context, records, failure):
    """独立尝试保存索引、概览和运行记录，原始失败优先于收尾失败。"""
    context["finished_at"] = project_time()
    context["stages"] = records
    context["failure"] = ("用户中断运行" if isinstance(failure, KeyboardInterrupt) else f"{type(failure).__name__}: {failure}") if failure else None
    original_code = 130 if isinstance(failure, KeyboardInterrupt) else getattr(failure, "exit_code", 1) if failure else 0
    context["exit_code"] = original_code
    errors = context["summary_errors"]
    columns = ["task", "result_name", "file", "method", "status", "reason"]
    index = pd.DataFrame(columns=columns)
    index_available = False
    try:
        index = collect_result_index(context["folder"], [row["task"] for row in records])
        index_available = True
    except Exception as error:
        errors.append(f"结果索引无法汇总：{error}")
    try:
        summary = build_project_summary(context, records, index)
    except Exception as error:
        errors.append(f"项目概览无法汇总：{error}")
        summary = {"folder": context["folder"], "context": context, "stages": records,
                   "uses": [], "index": index, "stage_summaries": {}, "tips": [],
                   "exit_code": original_code, "overview_saved": False, "index_saved": False}
    if index_available:
        try:
            save_csv(context["folder"] / "结果索引.csv", index)
            summary["index_saved"] = True
        except Exception as error:
            errors.append(f"结果索引保存失败：{error}")
    if failure:
        try:
            (context["folder"] / "错误日志.txt").write_text(
                "".join(traceback.format_exception(type(failure), failure, failure.__traceback__)), encoding="utf-8"
            )
        except Exception as error:
            errors.append(f"错误日志保存失败：{error}")
    context["exit_code"] = original_code or (1 if errors else 0)
    summary["exit_code"] = context["exit_code"]
    try:
        save_project_overview(summary)
        summary["overview_saved"] = True
    except Exception as error:
        errors.append(f"项目运行概览保存失败：{error}")
    context["exit_code"] = original_code or (1 if errors else 0)
    summary["exit_code"] = context["exit_code"]
    try:
        save_project_run(context)
    except Exception as error:
        errors.append(f"run.json保存失败：{error}")
        context["exit_code"] = original_code or 1
        summary["exit_code"] = context["exit_code"]
        if summary["overview_saved"]:
            try:
                save_project_overview(summary)
            except Exception as overview_error:
                errors.append(f"概览收尾问题补记失败：{overview_error}")
    return summary


def show_project_summary(summary):
    """展示本次范围内的真实结果；完整原因和证据导航在概览中。"""
    context = summary["context"]
    print("\n本次项目结果", flush=True)
    selected = [use for use in summary["uses"] if use["selected"]]
    if not selected:
        message = "本次已安排业务结果生成，但项目结果无法汇总；回查收尾问题与实际任务文件。" if any(number in context["planned_tasks"] for number in (3, 4)) else "本次尚未安排业务结果生成；查看已执行任务的治理或用途规则证据。"
        print(message, flush=True)
    for use in selected:
        state = project_status(use["status"]) if use["status"] else "结果状态无法汇总" if use["attempted"] else use["execution"]
        print(f"  {use['name']}：{state}（{use['verification']}）", flush=True)
    checked = [use for use in selected if use["checked"]]
    if any(use["attempted"] and use["status"] is None for use in selected):
        print("结果索引缺失，无法汇总本次用途核验状态；请回查收尾问题和已有检查证据。", flush=True)
    elif checked:
        passed = sum(use["status"] == "CHECK_PASSED" for use in checked)
        print(f"本次已核验结果中，检查通过 {passed}/{len(checked)} 项；另有 {len(selected) - len(checked)} 项无有效检查记录。", flush=True)
    if any(row["task"] == 5 for row in summary["stages"]):
        for line in read_process_summary(summary["folder"] / TASKS[5]):
            print(line, flush=True)
        print("业务检查通过数量不代表作答全部完成；模型聚合、原链和查询状态须分别核对。", flush=True)
    pending = [use["name"] for use in selected if use["status"] == "QUALITY_PENDING"]
    if pending:
        print("等待任务1数据治理：" + "、".join(pending), flush=True)
        print("先回查质量处理.csv与清洗对比.csv，完成治理后重新运行受影响流程。", flush=True)
    if any(use["status"] == "BASELINE_ONLY" for use in selected):
        print("基线用途：回查任务2同名规则与对应处理任务；具体原因见项目运行概览。", flush=True)
    if any(use["status"] in {"CHECK_FAILED", "NOT_RUN"} for use in selected):
        print("失败或未生成用途：回查同名检查、结果登记和阶段日志。", flush=True)
    if context["failure"]:
        print("运行异常：" + context["failure"], flush=True)
    for error in context["summary_errors"]:
        print("收尾问题：" + error, flush=True)
    if summary["overview_saved"]:
        print(f"项目运行概览：{summary['folder'] / '项目运行概览.md'}", flush=True)
    if summary["index_saved"]:
        print(f"结果索引：{summary['folder'] / '结果索引.csv'}", flush=True)
    print(f"结果目录：{summary['folder']}", flush=True)
