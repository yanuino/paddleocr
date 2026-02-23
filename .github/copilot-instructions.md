# Copilot instructions (this repo)

## Big picture
- This is a small, script-first OCR workspace using `paddleocr` + `paddlepaddle-gpu` and OpenCV.
- Primary workflow today is batch-processing images from a folder and previewing OCR results interactively.

## Key files
- `batchocr.py`: main script; recursively scans an input folder for `*.png`, runs `PaddleOCR.predict(...)`, and previews `ocr_res_img`.
- `pyproject.toml`: Python 3.10 pin + runtime deps and tool configs (Black/isort/pylint).
- `output/`: created at runtime by `batchocr.py` (currently not used for writing files).

## Run / debug (Windows)
- Python version is pinned to 3.10 (`requires-python = "=3.10"`).
- Typical local run:
  - Create/activate venv: `python -m venv .venv` then `./.venv/Scripts/Activate.ps1`
  - Install deps: `pip install -e .` (uses `pyproject.toml`)
  - Run: `python .\batchocr.py`

## Project conventions and patterns
- Paddle model source checks are disabled via `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=1` in `batchocr.py`.
- OCR call pattern:
  - `result = ocr.predict(input=str(img_path))`
  - `result` is a list of result items; for a single image this is typically length 1.
  - A result item may be either:
    - object-style (attributes like `.rec_texts`, `.rec_boxes`, `.img`), or
    - dict-style like `{"res": { ... "rec_texts": [...] ... }}` (this is what `print(result)` can look like).
  - OCR visualization image is read as a `PIL.Image` at `result[0].img["ocr_res_img"]` (object-style).

### Access patterns used in this repo
- Recognized text list:
  - object-style: `item.rec_texts`
  - dict-style: `item["res"]["rec_texts"]`
- Common search: find the index in `rec_texts` where a string starts with `"SAOMC"`.
- SAOMC format: `SAOMC_<model>_<omc>_<csc>_<version>_<subversion>` (e.g. `SAOMC_SM-G991B_OXM_VOD_15_0002`).
- SAOMC extraction regex (Python, named groups):
  - `^SAOMC_(?P<model>[^_]+)_(?P<omc>[^_]+)_(?P<csc>[^_]+)_(?P<version>\d+)_(?P<subversion>\d+)$`

### `predict()` result struct (fields on a result item)
- **Input metadata**
  - `input_path` (str): input path of the image to be predicted.
  - `page_index` (int | None): `None` for images.
  - `model_settings` (Dict[str, bool]): model parameters toggles for the pipeline.
  - `use_doc_preprocessor` (bool): whether the document preprocessing sub-pipeline is enabled.
  - `use_textline_orientation` (bool): whether Text Line Orientation Classification is enabled.

- **Document preprocessor results** (exists when `use_doc_preprocessor=True`)
  - `doc_preprocessor_res` (Dict[str, Union[str, Dict[str, bool], int]]): outputs from the preprocessing sub-pipeline.
  - `doc_preprocessor_res.input_path` (str | None): image path accepted by preprocessing; when input is a `numpy.ndarray`, saved as `None`.
  - `doc_preprocessor_res.model_settings` (Dict): model configuration parameters of the preprocessing sub-pipeline.
  - `doc_preprocessor_res.use_doc_orientation_classify` (bool): enable document orientation classification.
  - `doc_preprocessor_res.use_doc_unwarping` (bool): enable text image correction (unwarping).
  - `doc_preprocessor_res.angle` (int): orientation prediction; `[0,1,2,3]` maps to `[0°,90°,180°,270°]`; `-1` if not enabled.

- **Detection / recognition outputs**
  - `dt_polys` (List[numpy.ndarray]): detected polygon boxes; each is shape `(4, 2)`, dtype `int16`.
  - `dt_scores` (List[float]): confidence per detection box.
  - `text_det_params` (Dict[str, Dict[str, int | float]]): text detection config params.
    - `limit_side_len` (int): edge length limit during preprocessing.
    - `limit_type` (str): how to deal with side length restriction.
    - `thresh` (float): confidence threshold for text pixel classification.
    - `box_thresh` (float): confidence threshold for text detection boxes.
    - `unclip_ratio` (float): expansion coefficient of detected boxes.
    - `text_type` (str): fixed to `"general"`.
  - `textline_orientation_angles` (List[int]): actual angles when enabled (e.g. `[0,0,1]`); else `[-1,-1,-1]`.
  - `text_rec_score_thresh` (float): filtering threshold for recognition results.
  - `rec_texts` (List[str]): recognized text filtered by `text_rec_score_thresh`.
  - `rec_scores` (List[float]): recognition confidences filtered by `text_rec_score_thresh`.
  - `rec_polys` (List[numpy.ndarray]): detection polygons filtered by confidence (same format as `dt_polys`).
  - `rec_boxes` (numpy.ndarray): rectangular boxes, shape `(n, 4)`, dtype `int16`, rows are `[x_min, y_min, x_max, y_max]`.
- Image interop:
  - Convert PIL → OpenCV using `pil_to_cv2(...)` (handles `RGB`, `RGBA`, and grayscale `L`; converts to OpenCV BGR ordering).
- Interactive preview:
  - Uses `cv2.imshow(...)` + `cv2.waitKey(0)`; pressing `q`/`Q` exits the loop; always call `cv2.destroyAllWindows()` after the loop.

## Docstrings (required)
- Always add a module docstring at the top of any Python module.
- Always add docstrings for new public classes and for non-trivial helper functions.
- When changing behavior, update docstrings to match (no stale docs).

## Formatting / linting
- Black: line length 120 (see `pyproject.toml`).
- isort: Black profile + first-party modules list already configured.
- pylint: configured for OpenCV (`cv2`) and ignores `.venv`.

## Dependency notes
- Runtime deps include `paddlepaddle-gpu`; expect CUDA/GPU runtime requirements on the machine.
- OpenCV dependency is `opencv-python`.
