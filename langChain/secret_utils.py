import base64

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


def _fernet_key(password: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=390000,
    )
    return base64.urlsafe_b64encode(kdf.derive(password.encode("utf-8")))


def decrypt_env_value(encrypted_value: str, password: str) -> str:
    salt_text, token = encrypted_value.split(".", 1)
    salt = base64.urlsafe_b64decode(salt_text.encode("utf-8"))
    return Fernet(_fernet_key(password, salt)).decrypt(token.encode("utf-8")).decode("utf-8")
