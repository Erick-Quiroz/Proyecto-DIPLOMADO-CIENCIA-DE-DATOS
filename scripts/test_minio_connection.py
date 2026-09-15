"""Script para probar la conectividad y credenciales con el servidor MinIO S3."""

import os
import sys
from dotenv import dotenv_values


def test_connection():
    print("=" * 60)
    print("TEST DE CONEXIÓN A MINIO S3")
    print("=" * 60)

    env = dotenv_values(".env")
    endpoint = env.get("DVC_S3_ENDPOINT") or env.get("MINIO_ENDPOINT")
    access_key = env.get("AWS_ACCESS_KEY_ID") or env.get("MINIO_ROOT_USER")
    secret_key = env.get("AWS_SECRET_ACCESS_KEY") or env.get("MINIO_ROOT_PASSWORD")
    bucket = env.get("MINIO_BUCKET", "dvc-storage")
    region = env.get("AWS_DEFAULT_REGION", "us-east-1")

    print(f"Endpoint configurado : {endpoint}")
    print(f"Bucket objetivo      : {bucket}")
    print(f"Usuario / Access Key : {'[CONFIGURADO]' if access_key else '[NO ENCONTRADO]'}")
    print(f"Secret Key           : {'[CONFIGURADO]' if secret_key else '[NO ENCONTRADO]'}")
    print("-" * 60)

    if not endpoint or not access_key or not secret_key:
        print("❌ ERROR: Faltan variables en .env (ENDPOINT, ACCESS_KEY o SECRET_KEY).")
        return

    # 1. Probar conectividad HTTP básica
    import urllib.request

    print("1. Probando alcance de red al Endpoint...")
    try:
        req = urllib.request.Request(
            endpoint,
            headers={"User-Agent": "MinIO-Connection-Tester/1.0"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            print(f"   ✓ Conectividad HTTP exitosa (HTTP {resp.status})")
    except urllib.error.HTTPError as e:
        # MinIO retorna 403 Forbidden en la raíz para solicitudes anónimas, lo cual confirma que el servidor responde
        print(f"   ✓ Servidor MinIO alcanzable y respondiendo (HTTP {e.code})")
    except Exception as e:
        print(f"   ❌ No se pudo conectar al endpoint HTTP: {e}")
        return

    # 2. Probar autenticación y operaciones S3
    print("\n2. Probando autenticación S3 (Boto3 / S3 Client)...")
    try:
        import boto3
        from botocore.client import Config

        s3_client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
            config=Config(signature_version="s3v4"),
        )

        response = s3_client.list_buckets()
        buckets = [b["Name"] for b in response.get("Buckets", [])]
        print("   ✓ Autenticación S3 EXITOSA!")
        print(f"   ✓ Buckets detectados en MinIO: {buckets}")

        # 3. Comprobar si existe el bucket configurado
        if bucket in buckets:
            print(f"   ✓ El bucket objetivo '{bucket}' ya existe.")
        else:
            print(f"   ℹ️ El bucket objetivo '{bucket}' no existe aún. Intentando crearlo...")
            s3_client.create_bucket(Bucket=bucket)
            print(f"   ✓ Bucket '{bucket}' creado exitosamente.")

        # 4. Probar escritura y lectura de un objeto de prueba
        test_key = ".test_connection_ping.txt"
        test_content = b"ping minio connection test ok"
        s3_client.put_object(Bucket=bucket, Key=test_key, Body=test_content)
        print(f"   ✓ Escritura en bucket '{bucket}' verificada.")

        obj = s3_client.get_object(Bucket=bucket, Key=test_key)
        content = obj["Body"].read()
        assert content == test_content
        print(f"   ✓ Lectura desde bucket '{bucket}' verificada.")

        s3_client.delete_object(Bucket=bucket, Key=test_key)
        print(f"   ✓ Limpieza de objeto de prueba realizada.")

        print("\n" + "=" * 60)
        print("✓ LA CONEXIÓN A MINIO ESTÁ FUNCIONANDO PERFECTAMENTE.")
        print("=" * 60)

    except Exception as e:
        print(f"   ❌ Error en operación S3: {e}")


if __name__ == "__main__":
    test_connection()
