"""
Modulo de servicos S3 e PostgreSQL para upload de imagens com marca d'agua.
Gerencia conexao com banco de dados (PecasReposicao) e upload para AWS S3.
"""

import os
import uuid
from io import BytesIO
from pathlib import Path
from datetime import datetime

try:
    import boto3
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

try:
    import psycopg2
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False


def extract_sap_code(filename: str) -> str:
    """
    Extrai codigo SAP do nome do arquivo.
    Exemplos:
        '000014.jpg'              -> '000014'
        '007522_1200.jpg'         -> '007522'
        '001560-2_001935-2_1200.jpg' -> '001560-2'
    """
    stem = Path(filename).stem
    return stem.split('_')[0]


class DatabaseService:
    """Gerencia conexao e operacoes no PostgreSQL (tabela PecasReposicao)."""

    def __init__(self, host: str, port: int, dbname: str, user: str, password: str, **kwargs):
        self._conn_params = {
            'host': host,
            'port': port,
            'dbname': dbname,
            'user': user,
            'password': password,
            'sslmode': 'require',
        }
        self._conn = None

    def connect(self):
        """Abre conexao com o banco de dados."""
        if not PSYCOPG2_AVAILABLE:
            raise RuntimeError("psycopg2 nao esta instalado. Instale com: pip install psycopg2-binary")
        self._conn = psycopg2.connect(**self._conn_params)

    def disconnect(self):
        """Fecha conexao com o banco de dados."""
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def test_connection(self) -> bool:
        """Testa conectividade com o banco."""
        try:
            self.connect()
            cur = self._conn.cursor()
            cur.execute("SELECT 1")
            cur.close()
            self.disconnect()
            return True
        except Exception:
            self.disconnect()
            return False

    def validate_sap_codes(self, sap_codes: list) -> tuple:
        """
        Valida que todos os codigos SAP existem na tabela PecasReposicao.

        Retorna:
            (found_codes, missing_codes) - listas de codigos encontrados e faltantes
        """
        if not self._conn:
            raise RuntimeError("Nao conectado ao banco de dados. Chame connect() primeiro.")

        if not sap_codes:
            return [], []

        cur = self._conn.cursor()
        cur.execute(
            'SELECT "codigoSap" FROM "PecasReposicao" WHERE "codigoSap" = ANY(%s)',
            (list(sap_codes),)
        )
        found = {row[0] for row in cur.fetchall()}
        cur.close()

        found_codes = [c for c in sap_codes if c in found]
        missing_codes = [c for c in sap_codes if c not in found]

        return found_codes, missing_codes

    def update_image_url(self, sap_code: str, url: str) -> bool:
        """
        Atualiza o campo 'iamgem' na tabela PecasReposicao para o codigo SAP informado.

        Retorna True em caso de sucesso.
        """
        if not self._conn:
            raise RuntimeError("Nao conectado ao banco de dados. Chame connect() primeiro.")

        cur = self._conn.cursor()
        cur.execute(
            'UPDATE "PecasReposicao" SET "imagem" = %s WHERE "codigoSap" = %s',
            (url, sap_code)
        )
        self._conn.commit()
        affected = cur.rowcount
        cur.close()
        return affected > 0


class S3Service:
    """Gerencia upload de imagens para AWS S3."""

    BUCKET = "rotolinecustomersaplication"
    REGION = "sa-east-1"

    def __init__(self, access_key_id: str, secret_access_key: str):
        if not BOTO3_AVAILABLE:
            raise RuntimeError("boto3 nao esta instalado. Instale com: pip install boto3")
        self._client = boto3.client(
            's3',
            region_name=self.REGION,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
        )

    def test_connection(self) -> bool:
        """Testa conectividade com o bucket S3."""
        try:
            self._client.head_bucket(Bucket=self.BUCKET)
            return True
        except Exception:
            return False

    def generate_s3_key(self, directory: str, original_filename: str) -> str:
        """
        Gera chave unica para o objeto no S3.
        Formato: media/{directory}/{uuid8}_{timestamp}_{nome_original}
        """
        unique_id = uuid.uuid4().hex[:8]
        timestamp = int(datetime.now().timestamp())
        safe_name = Path(original_filename).name
        return f"media/{directory}/{unique_id}_{timestamp}_{safe_name}"

    def upload_image(self, pil_image, s3_key: str, fmt: str = "jpg") -> str:
        """
        Faz upload de uma imagem PIL para o S3.

        Args:
            pil_image: Imagem PIL (Image)
            s3_key: Chave do objeto no S3
            fmt: Formato da imagem ('jpg' ou 'png')

        Retorna:
            URL publica do objeto no S3
        """
        buffer = BytesIO()

        if fmt.lower() in ("jpg", "jpeg"):
            pil_image.convert("RGB").save(buffer, "JPEG", quality=95, optimize=True)
            content_type = "image/jpeg"
        else:
            pil_image.save(buffer, "PNG", compress_level=6)
            content_type = "image/png"

        buffer.seek(0)

        self._client.put_object(
            Bucket=self.BUCKET,
            Key=s3_key,
            Body=buffer,
            ContentType=content_type,
        )

        url = f"https://{self.BUCKET}.s3.{self.REGION}.amazonaws.com/{s3_key}"
        return url


def load_upload_config() -> dict:
    """
    Carrega configuracao de upload de variaveis de ambiente ou config.json.

    Prioridade: variaveis de ambiente > config.json > valores default.

    Retorna dict com chaves: aws, database, s3_directory
    """
    config = {
        "aws": {
            "access_key_id": os.environ.get("AWS_ACCESS_KEY_ID", ""),
            "secret_access_key": os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
            "bucket": "rotolinecustomersaplication",
            "region": "sa-east-1",
        },
        "database": {
            "host": os.environ.get("DB_HOST", "dbapi.c446hggeg6nd.sa-east-1.rds.amazonaws.com"),
            "port": int(os.environ.get("DB_PORT", "5432")),
            "dbname": os.environ.get("DB_NAME", "new_customers"),
            "user": os.environ.get("DB_USER", "RotoAPI"),
            "password": os.environ.get("DB_PASSWORD", ""),
        },
        "s3_directory": "pecas",
    }

    # Tentar carregar de config.json como fallback
    config_file = Path(__file__).parent / "config.json"
    if config_file.exists():
        try:
            import json
            with open(config_file, "r", encoding="utf-8") as f:
                file_config = json.load(f)

            s3_cfg = file_config.get("s3_upload", {})

            # AWS - so usa config.json se env var estiver vazia
            aws_file = s3_cfg.get("aws", {})
            if not config["aws"]["access_key_id"] and aws_file.get("access_key_id"):
                config["aws"]["access_key_id"] = aws_file["access_key_id"]
            if not config["aws"]["secret_access_key"] and aws_file.get("secret_access_key"):
                config["aws"]["secret_access_key"] = aws_file["secret_access_key"]

            # Database - so usa config.json se env var estiver vazia
            db_file = s3_cfg.get("database", {})
            if not config["database"]["password"] and db_file.get("password"):
                config["database"]["password"] = db_file["password"]

            # S3 directory
            if s3_cfg.get("s3_directory"):
                config["s3_directory"] = s3_cfg["s3_directory"]

        except Exception:
            pass

    return config
