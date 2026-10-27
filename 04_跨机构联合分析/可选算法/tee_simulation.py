"""可信执行环境（TEE）的接口与证明流程模拟。

算法背景：真实TEE依靠处理器和固件建立与普通操作系统隔离的执行区域，使宿主系统不能直接
读取受保护内存。数据提供方通常先验证远程证明中的代码度量、平台状态和新鲜随机数，再通过
受保护信道提交数据；受信代码只返回约定结果。

本示例的作用：用普通Python展示四个概念边界：代码度量、模拟证明、度量核对和最小结果
输出。它适合比较“把数据交给普通进程”和“只允许已批准程序输出汇总”的接口差异。

输入与输出：输入是各参与方的数值列表、预期代码度量和证明随机数；输出只有记录数量、合计、
均值以及一份模拟证明记录。关键参数是被批准程序的稳定描述、数据类型和允许公开的结果字段。

安全假设：真实方案必须信任具体TEE硬件、固件、证明根和密钥配置，并验证证明的新鲜性、平台
补丁状态和代码度量。数据进入TEE前后的加密信道和密钥供应也必须单独设计。

能力边界：这个文件没有硬件隔离。Python解释器、宿主操作系统和调试者都能看到所有输入、
修改代码和伪造下面的“证明”；模拟证据只是无密钥SHA-256摘要，不是硬件签名。它也没有处理
侧信道、回滚、拒绝服务、恶意输入、输出推断、证明服务、密钥封装或多租户隔离。因此运行本
示例不能证明数据受到TEE保护，也不能替代SGX、TDX、SEV-SNP等真实平台及其安全评估。
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Iterable

# 独立演示数据：不读取项目业务CSV，也不使用项目字段或参考参数。
DEMO_PARTY_PAYLOADS = {
    "party_a": [12.0, 18.0, 21.0],
    "party_b": [15.0, 24.0],
}
APPROVED_PROGRAM_DESCRIPTION = "aggregate-v1:count,sum,mean"
TAMPERED_PROGRAM_DESCRIPTION = "aggregate-v2:count,sum,mean,max"
DEMO_ATTESTATION_NONCE = "demo-request-001"


def measure_program(program_description: str) -> str:
    """对稳定的程序描述计算教学度量；真实TEE由受信硬件测量实际加载代码。"""
    return hashlib.sha256(program_description.encode("utf-8")).hexdigest()


def flatten_numeric_payloads(
    party_payloads: dict[str, Iterable[float]],
) -> list[float]:
    """验证演示输入并汇总到同一进程；这一步本身不提供机密性。"""
    values: list[float] = []
    if not party_payloads:
        raise ValueError("至少需要一个参与方")
    for party, payload in party_payloads.items():
        party_values = list(payload)
        if not party_values:
            raise ValueError(f"{party}没有输入记录")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in party_values
        ):
            raise TypeError(f"{party}只允许提供数值")
        if not all(math.isfinite(value) for value in party_values):
            raise ValueError(f"{party}只允许提供有限数值")
        values.extend(float(value) for value in party_values)
    return values


class MockTEE:
    """只模拟TEE调用接口；对象内部数据仍完全暴露给Python宿主。"""

    def __init__(self, program_description: str) -> None:
        """保存演示程序描述并计算其度量值，用于后续模拟证明核验。
        该对象属于本地接口演示，不提供真实硬件隔离。
        """
        self.program_description = program_description
        self.measurement = measure_program(program_description)

    def attest(self, nonce: str) -> dict[str, object]:
        """返回可观察的模拟证明记录；evidence不是签名，任何人都能重新计算。"""
        if not nonce:
            raise ValueError("证明随机数不能为空")
        public_claims = {
            "measurement": self.measurement,
            "nonce": nonce,
            "evidence_type": "SIMULATED_NOT_HARDWARE_BACKED",
        }
        encoded = json.dumps(
            public_claims,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return {
            **public_claims,
            "evidence": hashlib.sha256(encoded).hexdigest(),
        }

    def execute_approved_aggregate(
        self,
        party_payloads: dict[str, Iterable[float]],
        expected_measurement: str,
        nonce: str,
    ) -> dict[str, object]:
        """度量相符时只返回批准的汇总字段和模拟证明，不返回逐条输入。"""
        attestation = self.attest(nonce)
        if attestation["measurement"] != expected_measurement:
            raise ValueError("代码度量与数据提供方批准的度量不一致")
        values = flatten_numeric_payloads(party_payloads)
        total = sum(values)
        if not math.isfinite(total):
            raise ValueError("输入总和超出有限浮点数范围")
        return {
            "public_result": {
                "record_count": len(values),
                "sum": total,
                "mean": total / len(values),
            },
            "attestation": attestation,
        }


if __name__ == "__main__":
    approved_measurement = measure_program(APPROVED_PROGRAM_DESCRIPTION)
    approved_tee = MockTEE(APPROVED_PROGRAM_DESCRIPTION)
    result = approved_tee.execute_approved_aggregate(
        DEMO_PARTY_PAYLOADS,
        expected_measurement=approved_measurement,
        nonce=DEMO_ATTESTATION_NONCE,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))

    tampered_tee = MockTEE(TAMPERED_PROGRAM_DESCRIPTION)
    try:
        tampered_tee.execute_approved_aggregate(
            DEMO_PARTY_PAYLOADS,
            expected_measurement=approved_measurement,
            nonce=DEMO_ATTESTATION_NONCE,
        )
    except ValueError as error:
        print("未批准程序被模拟度量核对拒绝：", error)

    print("注意：普通Python没有硬件隔离，以上证明可以被宿主伪造。")
