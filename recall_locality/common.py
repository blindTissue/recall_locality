import csv
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def model_dir(model_name: str) -> str:
    return model_name.replace("/", "_")


def load_json(path: str, limit: int | None = None) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data[:limit] if limit else data


class ResultWriter:
    """Stream result rows to ``<out_dir>/results.csv``."""

    def __init__(self, out_dir: str | Path, fieldnames: list[str]):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.path = out_dir / "results.csv"
        self._file = open(self.path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        self._writer.writeheader()

    def write(self, row: dict) -> None:
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()
        logging.getLogger(__name__).info("Wrote %s", self.path)
