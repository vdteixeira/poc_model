"""Cenários de treino para o subagente `scanner`.

Cada cenário é um trecho de código com vulnerabilidades plantadas. Com o
RULER não precisamos rotular as vulnerabilidades — o juiz compara as N
tentativas do modelo entre si e ranqueia as melhores. Os campos
`expected_hints` existem só para você validar manualmente o progresso.

Em produção, substitua por trechos reais do SEU codebase (ex.: funções
extraídas de commits que corrigiram vulnerabilidades — o "antes" do fix é
um cenário perfeito).
"""

from dataclasses import dataclass, field


@dataclass
class ScanScenario:
    id: str
    filename: str
    code: str
    # Apenas para inspeção humana; RULER não usa isto.
    expected_hints: list[str] = field(default_factory=list)


SCENARIOS: list[ScanScenario] = [
    ScanScenario(
        id="sql-injection",
        filename="db/users.py",
        code='''\
import sqlite3

def get_user(conn: sqlite3.Connection, username: str):
    cursor = conn.cursor()
    query = f"SELECT * FROM users WHERE username = '{username}'"
    cursor.execute(query)
    return cursor.fetchone()

def update_email(conn: sqlite3.Connection, user_id: str, email: str):
    conn.execute(
        "UPDATE users SET email = '" + email + "' WHERE id = " + user_id
    )
    conn.commit()
''',
        expected_hints=["SQL injection em get_user", "SQL injection em update_email"],
    ),
    ScanScenario(
        id="command-injection",
        filename="ops/backup.py",
        code='''\
import os
import subprocess

def backup_directory(path: str):
    os.system(f"tar czf /backups/backup.tar.gz {path}")

def ping_host(host: str) -> str:
    result = subprocess.run(
        f"ping -c 1 {host}", shell=True, capture_output=True, text=True
    )
    return result.stdout
''',
        expected_hints=["command injection via os.system", "shell=True com input do usuário"],
    ),
    ScanScenario(
        id="path-traversal",
        filename="api/files.py",
        code='''\
from flask import Flask, request, send_file

app = Flask(__name__)

UPLOAD_DIR = "/var/app/uploads"

@app.route("/download")
def download():
    filename = request.args.get("file")
    return send_file(f"{UPLOAD_DIR}/{filename}")

@app.route("/read")
def read():
    name = request.args.get("name")
    with open(UPLOAD_DIR + "/" + name) as f:
        return f.read()
''',
        expected_hints=["path traversal em /download", "path traversal em /read"],
    ),
    ScanScenario(
        id="secrets-and-weak-crypto",
        filename="auth/tokens.py",
        code='''\
import hashlib
import jwt

SECRET_KEY = "super-secret-key-123"
API_TOKEN = "sk-prod-9f8e7d6c5b4a"

def hash_password(password: str) -> str:
    return hashlib.md5(password.encode()).hexdigest()

def decode_token(token: str):
    return jwt.decode(token, SECRET_KEY, algorithms=["HS256", "none"])
''',
        expected_hints=[
            "segredos hardcoded",
            "MD5 para senha",
            "algoritmo 'none' aceito no JWT",
        ],
    ),
    ScanScenario(
        id="ssrf-and-deserialization",
        filename="services/webhooks.py",
        code='''\
import pickle
import requests
from flask import Flask, request

app = Flask(__name__)

@app.route("/fetch")
def fetch():
    url = request.args.get("url")
    return requests.get(url, timeout=5).text

@app.route("/import", methods=["POST"])
def import_data():
    data = pickle.loads(request.data)
    return {"imported": len(data)}
''',
        expected_hints=["SSRF em /fetch", "deserialização insegura com pickle"],
    ),
    ScanScenario(
        id="clean-file",
        filename="utils/formatting.py",
        code='''\
def truncate(text: str, max_len: int = 80) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."

def slugify(title: str) -> str:
    return "-".join(
        "".join(c for c in word if c.isalnum()) for word in title.lower().split()
    )
''',
        # Arquivo limpo de propósito: ensina o modelo a NÃO alucinar
        # vulnerabilidades — o RULER pune falsos positivos ao comparar.
        expected_hints=["nenhuma vulnerabilidade — resposta correta é lista vazia"],
    ),
]
