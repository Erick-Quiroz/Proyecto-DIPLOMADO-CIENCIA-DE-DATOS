"""Gestor de entornos virtuales unificados o desacoplados por servicio."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]


def run_cmd(cmd, cwd=ROOT_DIR):
    print(f"-> Ejecutando: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    res = subprocess.run(cmd, shell=True if isinstance(cmd, str) else False, cwd=cwd)
    if res.returncode != 0:
        print(f"Error al ejecutar: {cmd}")
        sys.exit(res.returncode)


def setup_unified_env():
    print("\n--- Configurando Entorno Virtual Principal (.venv) ---")
    venv_dir = ROOT_DIR / ".venv"
    if not venv_dir.exists():
        run_cmd([sys.executable, "-m", "venv", str(venv_dir)])
    
    # Path al pip del venv
    if sys.platform == "win32":
        pip_path = venv_dir / "Scripts" / "pip.exe"
        python_path = venv_dir / "Scripts" / "python.exe"
    else:
        pip_path = venv_dir / "bin" / "pip"
        python_path = venv_dir / "bin" / "python"
        
    run_cmd([str(pip_path), "install", "--upgrade", "pip", "setuptools", "wheel"])
    req_file = ROOT_DIR / "requirements.txt"
    if req_file.exists():
        run_cmd([str(pip_path), "install", "-r", str(req_file)])
    
    # Instalar src como paquete editable
    run_cmd([str(pip_path), "install", "-e", "."])
    print("\n[OK] Entorno virtual principal configurado exitosamente en .venv")


def setup_service_env(service_name: str):
    print(f"\n--- Configurando Entorno Aislado para Servicio: {service_name} ---")
    service_dir = ROOT_DIR / "services" / service_name
    if not service_dir.exists():
        print(f"Servicio {service_name} no encontrado en {service_dir}")
        return
    
    venv_dir = service_dir / ".venv"
    if not venv_dir.exists():
        run_cmd([sys.executable, "-m", "venv", str(venv_dir)])
        
    if sys.platform == "win32":
        pip_path = venv_dir / "Scripts" / "pip.exe"
    else:
        pip_path = venv_dir / "bin" / "pip"
        
    run_cmd([str(pip_path), "install", "--upgrade", "pip", "setuptools", "wheel"])
    req_file = service_dir / "requirements.txt"
    if req_file.exists():
        run_cmd([str(pip_path), "install", "-r", str(req_file)])
        
    print(f"\n[OK] Entorno para {service_name} configurado en {venv_dir}")


def main():
    parser = argparse.ArgumentParser(description="Gestión de entornos virtuales")
    parser.add_argument(
        "--mode",
        choices=["unified", "service", "all-services"],
        default="unified",
        help="Modo de instalación: unificado (default), servicio individual o todos los servicios",
    )
    parser.add_argument("--service", type=str, help="Nombre del servicio específico")
    args = parser.parse_args()

    if args.mode == "unified":
        setup_unified_env()
    elif args.mode == "service":
        if not args.service:
            print("Debe especificar --service <nombre_servicio>")
            sys.exit(1)
        setup_service_env(args.service)
    elif args.mode == "all-services":
        services_dir = ROOT_DIR / "services"
        for s in services_dir.iterdir():
            if s.is_dir() and (s / "requirements.txt").exists():
                setup_service_env(s.name)


if __name__ == "__main__":
    main()
