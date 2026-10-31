"""Read dataset images directly from data_new without using target annotations."""

from pathlib import Path
import zipfile

from PIL import Image


def load_dataset_image(data_dir: str | Path, image_id: str) -> Image.Image:
    """Load a local image or one matching member in a dataset ZIP archive.

    Args:
        data_dir: Dataset folder, such as data/data_new, containing images or ZIPs.
        image_id: Filename stem such as 0063. Numeric IDs are padded to four digits.

    Returns:
        A detached RGB image. JSON/CSV answers, points, and target boxes are never
        read by this function and cannot influence the search.
    """
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    stem = Path(image_id).stem
    if stem.isdigit():
        stem = stem.zfill(4)
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.stem == stem and path.suffix.lower() in extensions
    ]
    if len(files) > 1:
        raise ValueError(f"Multiple dataset images match ID {stem}.")
    if files:
        with Image.open(files[0]) as source:
            return source.convert("RGB")
    matches: list[tuple[Path, str]] = []
    for archive_path in sorted(root.glob("*.zip")):
        with zipfile.ZipFile(archive_path) as archive:
            matches.extend(
                (archive_path, name)
                for name in archive.namelist()
                if Path(name).stem == stem and Path(name).suffix.lower() in extensions
            )
    if len(matches) != 1:
        raise ValueError(
            f"Expected one image for ID {stem}; found {len(matches)} ZIP members."
        )
    archive_path, name = matches[0]
    with zipfile.ZipFile(archive_path) as archive, archive.open(name) as stream:
        with Image.open(stream) as source:
            return source.convert("RGB")
