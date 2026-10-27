"""使用AES-GCM进行带认证的对称加密。

算法背景：AES是分组密码，同一把秘密密钥用于加密和解密。GCM是一种认证加密工作模式，
除密文外还生成认证标签，使接收方能够发现密文或附加认证数据是否被修改。AES-GCM常用于
保护存储中的数据、网络消息和备份文件。

核心步骤：①生成128、192或256位AES密钥；②为每次加密生成在该密钥下唯一的12字节随机数
nonce；③将明文和可选附加认证数据交给AES-GCM；④接收方使用相同密钥、nonce和附加数据
解密并验证认证标签。

输入与输出：encrypt_aes_gcm接收明文字节、密钥和可选附加认证数据，内部生成nonce，
返回(nonce, ciphertext)，其中ciphertext包含认证标签。decrypt_aes_gcm接收nonce、
密文、相同密钥及附加认证数据，验证后返回原始明文。示例使用Base64只为便于打印，
Base64本身不提供加密保护。

作用与局限：AES-GCM保护机密性和完整性，但合法接收方解密后仍能看到完整明文。它不会让
密文直接支持普通统计分析、集合匹配或模型训练。相同密钥下绝不能重复使用nonce，否则会
严重破坏安全性；密钥还需要独立的生成、分发、轮换和撤销机制。
"""

import base64
import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# 独立演示数据：附加认证数据不会被加密，但会受到认证标签保护。
DEMO_MESSAGE = b"Confidential sample message"
DEMO_ASSOCIATED_DATA = b"record-type:demo"
DEMO_KEY_BITS = 128


def encrypt_aes_gcm(
    plaintext: bytes,
    key: bytes,
    associated_data: bytes | None = None,
) -> tuple[bytes, bytes]:
    """生成12字节随机nonce，并返回nonce和带认证标签的密文。"""
    nonce = os.urandom(12)
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, associated_data)
    return nonce, ciphertext


def decrypt_aes_gcm(
    nonce: bytes,
    ciphertext: bytes,
    key: bytes,
    associated_data: bytes | None = None,
) -> bytes:
    """验证认证标签并解密；密文被修改时会抛出异常。"""
    return AESGCM(key).decrypt(nonce, ciphertext, associated_data)


if __name__ == "__main__":
    demo_key = AESGCM.generate_key(bit_length=DEMO_KEY_BITS)
    demo_nonce, demo_ciphertext = encrypt_aes_gcm(DEMO_MESSAGE, demo_key, DEMO_ASSOCIATED_DATA)
    recovered_message = decrypt_aes_gcm(
        demo_nonce, demo_ciphertext, demo_key, DEMO_ASSOCIATED_DATA
    )
    print("nonce(base64):", base64.b64encode(demo_nonce).decode("ascii"))
    print("ciphertext(base64):", base64.b64encode(demo_ciphertext).decode("ascii"))
    print("recovered:", recovered_message.decode("utf-8"))
