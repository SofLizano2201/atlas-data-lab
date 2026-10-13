"""Generar ejemplos CSV locales reproducibles: python datasets/examples/generate.py."""

from pathlib import Path

from sklearn.datasets import load_diabetes, load_iris


def main() -> None:
    """Guardar ejemplos en la carpeta del generador."""
    destination = Path(__file__).resolve().parent
    for name, loader in {"iris": load_iris, "diabetes": load_diabetes}.items():
        loader(as_frame=True).frame.to_csv(destination / f"{name}.csv", index=False)


if __name__ == "__main__":
    main()
