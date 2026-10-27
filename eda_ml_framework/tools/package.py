"""Empaquetar fuentes y documentación sin entornos, cachés ni modelos privados."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def main() -> None:
    """Crear ZIP portable en la carpeta superior del proyecto."""
    project = Path(__file__).resolve().parents[1]
    output = project.parent / "atlas-data-lab.zip"
    excluded = {".venv", "__pycache__", ".pytest_cache", ".ruff_cache", ".git", "artifacts"}
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for source in sorted(project.rglob("*")):
            relative = source.relative_to(project)
            if not source.is_file() or any(
                p in excluded or p.endswith(".egg-info") for p in relative.parts
            ):
                continue
            if source.suffix in {".pyc", ".joblib"}:
                continue
            if (
                source.name.startswith(".env") and source.name != ".env.example"
            ) or source.name == "secrets.toml":
                continue
            archive.write(source, Path(project.name) / relative)
    print(f"Paquete creado: {output.name}")


if __name__ == "__main__":
    main()
