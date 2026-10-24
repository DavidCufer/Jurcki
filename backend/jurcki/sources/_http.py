"""Tiny download helper shared by the source modules."""
from pathlib import Path

import requests


def download(url: str, dest: Path, timeout: int = 60) -> bool:
    """Download url to dest unless it exists. Returns False on HTTP 404."""
    dest = Path(dest)
    if dest.exists():
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=timeout) as r:
        if r.status_code == 404:
            return False
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    tmp.rename(dest)
    return True
