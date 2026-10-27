"""最小哈希链示例。

背景：区块链最基础的结构是按顺序追加区块。每个区块保存自己的交易、前一区块的摘要和
当前内容摘要。历史区块的内容发生变化后，当前摘要会失效，后续区块保存的前序摘要也不再
匹配，因此可以发现未重算摘要的内容修改、删除中间区块或调换区块顺序。
删除末尾区块会留下结构一致的前缀；没有可信的期望末端时，不能识别这种截断。

核心数据结构：区块编号、时间、交易列表、前序哈希和当前哈希。第0号创世区块使用64个0
作为前序哈希。交易是任意JSON兼容字典。

输入与输出：输入是交易列表和前序哈希；输出是新区块。验证函数输入整条链，输出当前内容摘要
和链接是否一致，以及发现问题的位置；不能据此证明历史无缺失。查询函数按一个交易字段返回匹配记录。

基本步骤：①稳定序列化区块内容；②计算SHA-256；③把前一区块摘要写入新区块；④从创世块
开始逐块重新计算摘要并检查链接；⑤按字段遍历交易。

作用：该示例只展示追加、哈希链接、内容摘要与链接校验和查询，不提供业务访问控制。

局限：它没有挖矿、共识、节点网络、数字签名、智能合约、可信时间戳和持久化节点。能够
控制唯一副本的一方可以重写整条链，因此这只是区块链数据结构的最小教学模拟。
"""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json

DEMO_TRANSACTIONS = [
    {"transaction_id": "TX-001", "operation": "CREATE", "object": "document-A"},
    {"transaction_id": "TX-002", "operation": "APPROVE", "object": "document-A"},
]


def calculate_block_hash(block: dict[str, object]) -> str:
    """排除block_hash字段后计算稳定的SHA-256摘要。"""
    content = {key: value for key, value in block.items() if key != "block_hash"}
    encoded = json.dumps(
        content, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def create_block(
    block_index: int,
    transactions: list[dict[str, object]],
    previous_hash: str,
    timestamp: str | None = None,
) -> dict[str, object]:
    """创建一个最小区块并计算当前哈希。"""
    if type(block_index) is not int or block_index < 0:
        raise ValueError("区块编号必须是非负整数")
    if (not isinstance(previous_hash, str) or len(previous_hash) != 64
            or any(ch not in "0123456789abcdef" for ch in previous_hash)):
        raise ValueError("前序摘要必须为64个小写十六进制字符")
    if not isinstance(transactions, list) or any(not isinstance(item, dict) for item in transactions):
        raise ValueError("交易必须是JSON对象列表")
    if timestamp is not None and (not isinstance(timestamp, str) or not timestamp):
        raise ValueError("时间必须是非空字符串")
    block: dict[str, object] = {
        "block_index": block_index,
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        "transactions": deepcopy(transactions),
        "previous_hash": previous_hash,
    }
    block["block_hash"] = calculate_block_hash(block)
    return block


def create_genesis_block() -> dict[str, object]:
    """创建第0号创世区块。"""
    return create_block(
        0, [{"transaction_id": "GENESIS", "operation": "CREATE_CHAIN"}], "0" * 64
    )


def verify_blockchain(blockchain: list[dict[str, object]]) -> tuple[bool, str]:
    """检查编号、前序链接和区块内容摘要。"""
    if not isinstance(blockchain, list) or not blockchain:
        return False, "空链或无效链缺少创世区块"
    for position, block in enumerate(blockchain):
        if not isinstance(block, dict):
            return False, f"区块位置{position}不是JSON对象"
        if (type(block.get("block_index")) is not int
                or not isinstance(block.get("timestamp"), str) or not block["timestamp"]
                or not isinstance(block.get("transactions"), list)
                or any(not isinstance(item, dict) for item in block["transactions"])):
            return False, f"区块 #{position} 的字段类型或交易结构无效"
        expected_previous = (
            "0" * 64 if position == 0 else str(blockchain[position - 1]["block_hash"])
        )
        if block.get("block_index") != position:
            return False, f"区块位置{position}的编号不连续"
        if block.get("previous_hash") != expected_previous:
            return False, f"区块 #{position} 的前序哈希不匹配"
        try:
            computed = calculate_block_hash(block)
        except (ValueError, TypeError):
            return False, f"区块 #{position} 含有不能稳定序列化的内容"
        if block.get("block_hash") != computed:
            return False, f"区块 #{position} 的内容摘要不一致"
    return True, f"结构一致，共{len(blockchain)}个区块；未证明历史无缺失"


def query_transactions(
    blockchain: list[dict[str, object]],
    field: str,
    value: object,
) -> list[dict[str, object]]:
    """返回指定交易字段等于目标值的链上记录。"""
    valid, reason = verify_blockchain(blockchain)
    if not valid:
        raise ValueError(f"不能查询结构无效的链：{reason}")
    matches: list[dict[str, object]] = []
    for block in blockchain:
        for transaction in block["transactions"]:
            if field in transaction and transaction[field] == value:
                matches.append(
                    {
                        **transaction,
                        "block_index": block["block_index"],
                        "block_hash": block["block_hash"],
                    }
                )
    return matches


if __name__ == "__main__":
    demo_chain = [create_genesis_block()]
    demo_chain.append(create_block(1, DEMO_TRANSACTIONS, str(demo_chain[-1]["block_hash"])))
    print(json.dumps(demo_chain, ensure_ascii=False, indent=2))
    print(verify_blockchain(demo_chain))
    print(query_transactions(demo_chain, "transaction_id", "TX-002"))
