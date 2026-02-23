"""Batch OCR utility using PaddleOCR.

Recursively scans an input directory for PNG images, runs PaddleOCR on each image,
prints structured SAOMC metadata when detected, and shows an interactive preview
of the OCR overlay image.
"""

import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image

from mncdb import MCC_MNC_TO_OPERATOR

# Initialize PaddleOCR instance

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "1"
ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=True, use_textline_orientation=False)


@dataclass(frozen=True, slots=True)
class SaomcRecord:
    """Structured SAOMC record extracted from OCR.

    - `imei` is derived from the image filename stem.
    - `saomc` stores the OCR text immediately following the matched SAOMC token
      in the same `rec_texts` list (or `None` if not available).
    """

    imei: str
    saomc: str | None
    model: str
    omc: str
    csc: str
    version: str
    subversion: str


@dataclass(frozen=True, slots=True)
class DevconInfo:
    """Device connectivity info extracted from AT command data in csv file."""

    imei: str
    mn: str
    mnc: str
    mcc: str
    prd: str
    aid: str
    cc: str


SAOMC_RE = re.compile(
    r"^SAOMC_(?P<model>[^_]+)_(?P<omc>[^_]+)_(?P<csc>[^_]+)_(?P<version>[^_]+)_(?P<subversion>[^_]+)$"
)


def parse_saomc_text(text: str) -> dict[str, str] | None:
    """Parse a SAOMC string into its named fields.

    Returns a dict with keys: model, omc, csc, version, subversion; or `None` if
    the input does not match the expected SAOMC format.
    """
    m = SAOMC_RE.match(text.strip())
    if not m:
        return None
    return m.groupdict()


def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    """Convert a PIL.Image to an OpenCV image (numpy array).

    OpenCV uses BGR channel order; PIL commonly uses RGB/RGBA.
    """
    if pil_img.mode == "RGB":
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    if pil_img.mode == "RGBA":
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGR)
    if pil_img.mode == "L":
        return np.array(pil_img)

    # Fallback: convert uncommon modes (e.g., P, CMYK) to RGB first.
    pil_rgb = pil_img.convert("RGB")
    return cv2.cvtColor(np.array(pil_rgb), cv2.COLOR_RGB2BGR)


def _get_rec_texts(result_item):
    """Return `rec_texts` from a PaddleOCR result item.

    Supports both object-style items (attribute `rec_texts`) and dict-style items
    (either top-level `rec_texts` or nested under `res`).
    """
    recognized_texts = getattr(result_item, "rec_texts", None)
    if recognized_texts is not None:
        return recognized_texts

    if isinstance(result_item, dict):
        if "rec_texts" in result_item:
            return result_item.get("rec_texts")
        res = result_item.get("res")
        if isinstance(res, dict) and "rec_texts" in res:
            return res.get("rec_texts")

    return None


def find_in_results_startswith(results, prefix: str) -> tuple[int, int] | None:
    """Search OCR output for a recognized text entry starting with `prefix`.

    PaddleOCR result items can be either:
    - an object with attributes like `.rec_texts`, or
    - a dict-like payload (often shaped like `{'res': {..., 'rec_texts': [...]}}`).

    Returns `(result_item_index, rec_text_index)` for the first match, otherwise `None`.
    """

    for loop_item_i, item in enumerate(results):
        item_rec_texts = _get_rec_texts(item)
        if not item_rec_texts:
            continue
        for loop_text_i, text in enumerate(item_rec_texts):
            if isinstance(text, str) and text.startswith(prefix):
                return (loop_item_i, loop_text_i)

    return None


def read_devconinfo_csv(input_dir: Path, imei: str, *, strict: bool = False) -> DevconInfo:
    """Read `<imei>.csv` from `input_dir` and return a `DevconInfo`.

    The CSV is expected to have a single header row followed by at least one data row.

    Some files have 7 headers: `imei,mn,mnc,mcc,prd,aid,cc`.
    Some files have 6 headers and `mn` is entirely missing: `imei,mnc,mcc,prd,aid,cc`.
    In that case, `mn` is returned as `"-"`.

    If `strict=False` (default) and the CSV cannot be found/read, returns a `DevconInfo`
    with empty fields (except `imei`). If `strict=True`, raises an exception.
    """

    csv_path = input_dir / f"{imei}.csv"
    if not csv_path.exists():
        # Fallback: if PNGs are in subfolders, their CSV might be too.
        matches = list(input_dir.rglob(f"{imei}.csv"))
        if matches:
            csv_path = matches[0]

    if not csv_path.exists():
        if strict:
            raise FileNotFoundError(f"Devcon CSV not found for IMEI {imei!r} under {str(input_dir)!r}")
        return DevconInfo(imei=imei, mn="-", mnc="", mcc="", prd="", aid="", cc="")

    try:
        with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            row = next(reader, None)
            if row is None:
                raise ValueError("CSV has header but no data rows")

            def _get(name: str) -> str:
                val = row.get(name, "")
                return val.strip() if isinstance(val, str) else ""

            mn_val = "-"
            if reader.fieldnames and "mn" in reader.fieldnames:
                mn_val = _get("mn") or "-"

            return DevconInfo(
                imei=_get("imei") or imei,
                mn=mn_val,
                mnc=_get("mnc"),
                mcc=_get("mcc"),
                prd=_get("prd"),
                aid=_get("aid"),
                cc=_get("cc"),
            )
    except (OSError, ValueError, csv.Error):
        if strict:
            raise
        return DevconInfo(imei=imei, mn="-", mnc="", mcc="", prd="", aid="", cc="")


# Run OCR inference on each PNG in a directory (recursive)

INPUT_DIR = Path(
    "C:\\Users\\frlocy00\\OneDrive - Ingram Micro\\Shared with R&D Beauvais\\Documents\\BVS_MOBILE\\Analyse Carrier Tag"
)
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_PATH = OUTPUT_DIR / "result.csv"
RESULT_FIELDS = [
    "imei",
    "carrier",
    "mn",
    "mnc",
    "mcc",
    "sim_operator",
    "prd",
    "aid",
    "cc",
    "saomc",
    "model",
    "omc",
    "csc",
    "version",
    "subversion",
]

with RESULT_PATH.open("w", newline="", encoding="utf-8") as result_file:
    writer = csv.DictWriter(result_file, fieldnames=RESULT_FIELDS)
    writer.writeheader()

    for img_path in INPUT_DIR.rglob("*.png"):
        devcon = read_devconinfo_csv(INPUT_DIR, img_path.stem)
        if devcon.imei != img_path.stem:
            raise ValueError(
                f"IMEI mismatch for {str(img_path)!r}: png stem={img_path.stem!r} but csv imei={devcon.imei!r}"
            )
        sim_operator = MCC_MNC_TO_OPERATOR.get((devcon.mcc, devcon.mnc), "")
        result = ocr.predict(input=str(img_path))

        saomc_record: SaomcRecord | None = None
        saomc_match = find_in_results_startswith(result, "SAOMC")
        if saomc_match is not None:
            match_item_i, match_text_i = saomc_match

            # Handle both object-style and dict-style results.
            if hasattr(result[match_item_i], "rec_texts"):
                matched_text = result[match_item_i].rec_texts[match_text_i]
            else:
                payload = result[match_item_i].get("res", result[match_item_i])
                matched_text = payload["rec_texts"][match_text_i]

            print(
                f"{img_path}: found SAOMC at rec_texts[{match_text_i}] (result item {match_item_i}) = {matched_text!r}"
            )

            next_text: str | None = None
            matched_item_rec_texts = _get_rec_texts(result[match_item_i])
            if matched_item_rec_texts and (match_text_i + 1) < len(matched_item_rec_texts):
                candidate = matched_item_rec_texts[match_text_i + 1]
                if isinstance(candidate, str):
                    next_text = candidate

            saomc_fields = parse_saomc_text(matched_text)
            if saomc_fields is not None:
                saomc_record = SaomcRecord(imei=img_path.stem, saomc=next_text, **saomc_fields)
                print(saomc_record)
        print(devcon)
        writer.writerow(
            {
                "imei": devcon.imei,
                "carrier": img_path.parent.name,
                "mn": devcon.mn,
                "mnc": devcon.mnc,
                "mcc": devcon.mcc,
                "sim_operator": sim_operator,
                "prd": devcon.prd,
                "aid": devcon.aid,
                "cc": devcon.cc,
                "saomc": saomc_record.saomc if saomc_record else "",
                "model": saomc_record.model if saomc_record else "",
                "omc": saomc_record.omc if saomc_record else "",
                "csc": saomc_record.csc if saomc_record else "",
                "version": saomc_record.version if saomc_record else "",
                "subversion": saomc_record.subversion if saomc_record else "",
            }
        )

        img = result[0].img["ocr_res_img"]  # PIL.Image
        cv_img = pil_to_cv2(img)

        # Example: show via OpenCV (press any key to advance)
        cv2.imshow("ocr_res_img", cv_img)
        # key = cv2.waitKey(0) & 0xFF
        # if key in (ord("q"), ord("Q")):
        #     break
        cv2.waitKey(100)

cv2.destroyAllWindows()
