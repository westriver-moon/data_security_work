"""项目入口：治理数据、明确用途、生成共享与联合分析结果，最后核查追踪。

华辰需要提前发现故障，德维需要关联维修案例、研究设备和复核服务统计；
双方还需要匹配共同设备、汇总告警并比较故障模型，共形成六项用途。
三张原表先经过治理和用途决策，再进入共享发布、联合分析两条分支。

python main.py：完整运行；python main.py --task 4：运行任务1、2、4。
每次建立新目录，先看项目运行概览.md，再按其中的文件路径查看本次证据。
运行、日志与汇总由utils.py提供；学生在各任务main.py的作答框中填写。
"""

import argparse
from pathlib import Path
from utils import (
    TASKS, create_project_run, save_project_run, show_project_context,
    run_stage, finish_project_run, show_project_summary,
)

# 任务3、4都直接依赖任务1、2，两者没有先后依赖；任务5核查两条分支。
PREREQUISITES = {1: [], 2: [1], 3: [1, 2], 4: [1, 2], 5: [1, 2, 3, 4]}


def main():
    """显式组织五个业务阶段，并汇总本次实际结果与未完成原因。

    任务之间通过同一次运行中的清洗表、用途规则、结果登记及事件交接。
    程序正常结束只说明流程结束，具体完成情况须结合结果、代码和报告。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, choices=TASKS, help="运行指定任务及必要前序")
    parser.add_argument("--output-dir", type=Path, help="指定新结果目录，不覆盖已有目录")
    parser.add_argument("--verbose", action="store_true", help="各阶段结束后回显完整日志")
    args = parser.parse_args()
    sequence = list(TASKS) if args.task is None else PREREQUISITES[args.task] + [args.task]
    try:
        context = create_project_run(args, sequence)
    except OSError as error:
        parser.exit(1, f"无法创建新的结果目录：{error}\n")
    records = context["stages"]
    failure = None
    try:
        save_project_run(context)
        show_project_context(context)

        if 1 in sequence:
            # 治理双方设备表和服务案例表：清洗表供后续计算，字段分级供任务2。
            run_stage(1, context, records)

        if 2 in sequence:
            # 结合清洗数据与字段分级，形成六项用途的输入、输出和保护规则。
            run_stage(2, context, records)

        if 3 in sequence:
            # 共享分支：按任务2规则处理任务1数据，生成德维需要的三类结果。
            run_stage(3, context, records)

        if 4 in sequence:
            # 联合分支：同样使用任务1、2；先匹配与确定归属，再统计告警、比较模型。
            run_stage(4, context, records)

        if 5 in sequence:
            # 汇合两条分支：核对六类结果与来源，读回验链后查询操作。
            run_stage(5, context, records)
    except (Exception, KeyboardInterrupt) as error:
        failure = error
    finally:
        # 异常时也收集已产生的事实；汇总问题不能掩盖原来的失败原因。
        summary = finish_project_run(context, records, failure)
    show_project_summary(summary)
    raise SystemExit(summary["exit_code"])


if __name__ == "__main__":
    main()
