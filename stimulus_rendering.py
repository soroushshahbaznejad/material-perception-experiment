"""Focus rendering for material-perception stimuli.

Each stimulus is a photograph plus a LabelMe-style polygon annotation of the
target material. Two views are rendered on a fixed canvas:

* ``normal`` - the full scene, letterboxed to the output size.
* ``blur``   - the same scene with everything outside the target mask blurred
  and dimmed, used for the press-and-hold focus interaction.
"""
from __future__ import annotations

import json
from functools import lru_cache
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_SIZE = (640, 640)
FOCUS_DIM_FACTOR = 0.48        # non-target areas dimmed to 48% brightness
FOCUS_BLUR_RADIUS = 28
CANVAS_COLOR = (18, 18, 18)
MIN_OUTPUT_EDGE = 320
MAX_OUTPUT_EDGE = 2400


def normalize_label(label: str) -> str:
    return (label or "").strip().lower()


def resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = BASE_DIR / path
    resolved = path.resolve()
    resolved.relative_to(BASE_DIR.resolve())
    return resolved


def load_annotation_record(annotation_path: str) -> dict:
    with resolve_repo_path(annotation_path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    image_width = data.get("imageWidth")
    image_height = data.get("imageHeight")
    annotation_size: tuple[int, int] | None = None
    if image_width and image_height:
        annotation_size = (int(image_width), int(image_height))

    shapes: list[dict] = []
    for shape in data.get("shapes", []):
        points: list[tuple[float, float]] = []
        raw_points = shape.get("points") or shape.get("vertices") or []
        for point in raw_points:
            if len(point) < 2:
                continue
            points.append((float(point[0]), float(point[1])))
        if len(points) < 3:
            continue

        triangles: list[tuple[int, int, int]] = []
        for triangle in shape.get("triangles", []):
            if len(triangle) < 3:
                continue
            triangles.append((int(triangle[0]), int(triangle[1]), int(triangle[2])))

        label = (shape.get("label") or "").strip()
        if not label:
            continue
        attributes = shape.get("attributes") or {}
        orig_label = (
            attributes.get("orig_label")
            or attributes.get("source_object_name")
            or ""
        ).strip()
        shapes.append(
            {
                "label": label,
                "points": points,
                "triangles": triangles,
                "orig_label": orig_label,
                "attributes": attributes,
            }
        )
    return {
        "image_size": annotation_size,
        "shapes": shapes,
    }


def filter_shapes_for_label(shapes: list[dict], target_label: str) -> list[dict]:
    target = normalize_label(target_label)
    return [shape for shape in shapes if normalize_label(shape["label"]) == target]


def scale_shapes_to_image(
    shapes: list[dict],
    annotation_size: tuple[int, int] | None,
    image_size: tuple[int, int],
) -> list[dict]:
    if not annotation_size:
        return shapes

    ann_w, ann_h = annotation_size
    img_w, img_h = image_size
    if ann_w <= 0 or ann_h <= 0:
        return shapes
    if ann_w == img_w and ann_h == img_h:
        return shapes

    scale_x = img_w / ann_w
    scale_y = img_h / ann_h
    scaled_shapes: list[dict] = []
    for shape in shapes:
        scaled_shapes.append(
            {
                "label": shape["label"],
                "points": [
                    (x * scale_x, y * scale_y)
                    for x, y in shape["points"]
                ],
                "triangles": [tuple(triangle) for triangle in shape.get("triangles", [])],
                "orig_label": shape.get("orig_label", ""),
                "attributes": dict(shape.get("attributes") or {}),
            }
        )
    return scaled_shapes


def combined_bbox(shapes: list[dict]) -> tuple[float, float, float, float]:
    xs = [x for shape in shapes for x, _ in shape["points"]]
    ys = [y for shape in shapes for _, y in shape["points"]]
    return min(xs), min(ys), max(xs), max(ys)


def create_mask(size: tuple[int, int], shapes: list[dict]) -> Image.Image:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for shape in shapes:
        triangles = shape.get("triangles") or []
        if triangles:
            for triangle in triangles:
                try:
                    tri_points = [shape["points"][index] for index in triangle]
                except IndexError:
                    continue
                draw.polygon(tri_points, fill=255)
            continue
        draw.polygon(shape["points"], fill=255)
    return mask


def fit_image_and_shapes(
    image: Image.Image,
    shapes: list[dict],
    output_size: tuple[int, int] = OUTPUT_SIZE,
) -> tuple[Image.Image, list[dict]]:
    src_w, src_h = image.size
    out_w, out_h = output_size
    scale = min(out_w / max(src_w, 1), out_h / max(src_h, 1))
    scaled_w = max(1, int(round(src_w * scale)))
    scaled_h = max(1, int(round(src_h * scale)))
    offset_x = (out_w - scaled_w) / 2
    offset_y = (out_h - scaled_h) / 2

    resampling = getattr(Image, "Resampling", Image)
    resized = image.resize((scaled_w, scaled_h), resampling.LANCZOS)
    canvas = Image.new("RGB", output_size, CANVAS_COLOR)
    canvas.paste(resized, (int(round(offset_x)), int(round(offset_y))))

    scaled_shapes: list[dict] = []
    for shape in shapes:
        scaled_shapes.append(
            {
                "label": shape["label"],
                "points": [
                    (x * scale + offset_x, y * scale + offset_y)
                    for x, y in shape["points"]
                ],
                "triangles": [tuple(triangle) for triangle in shape.get("triangles", [])],
                "orig_label": shape.get("orig_label", ""),
            }
        )
    return canvas, scaled_shapes


def render_focus_image(
    source_image_path: str,
    annotation_path: str,
    target_label: str,
    *,
    mode: str = "normal",
    output_size: tuple[int, int] = OUTPUT_SIZE,
) -> Image.Image:
    annotation = load_annotation_record(annotation_path)

    with Image.open(resolve_repo_path(source_image_path)) as raw_image:
        image = raw_image.convert("RGB")

    shapes = scale_shapes_to_image(annotation["shapes"], annotation.get("image_size"), image.size)
    target_shapes = filter_shapes_for_label(shapes, target_label)
    if not target_shapes:
        raise ValueError(
            f"No polygons for class '{target_label}' in annotation '{annotation_path}'."
        )

    fitted_image, scaled_shapes = fit_image_and_shapes(image, target_shapes, output_size=output_size)
    mask = create_mask(output_size, scaled_shapes)
    soft_mask = mask.filter(ImageFilter.GaussianBlur(radius=1.2))

    if mode == "normal":
        return fitted_image.copy()
    if mode == "blur":
        blurred = fitted_image.filter(ImageFilter.GaussianBlur(radius=FOCUS_BLUR_RADIUS))
        dimmed = ImageEnhance.Brightness(blurred).enhance(FOCUS_DIM_FACTOR)
        return Image.composite(fitted_image, dimmed, soft_mask)
    raise ValueError(f"Unsupported render mode: {mode}")


def clamp_output_size(width: int | None, height: int | None) -> tuple[int, int]:
    if width is None or height is None:
        return OUTPUT_SIZE
    safe_width = max(MIN_OUTPUT_EDGE, min(int(width), MAX_OUTPUT_EDGE))
    safe_height = max(MIN_OUTPUT_EDGE, min(int(height), MAX_OUTPUT_EDGE))
    return safe_width, safe_height


@lru_cache(maxsize=128)
def render_focus_encoded_bytes(
    source_image_path: str,
    annotation_path: str,
    target_label: str,
    mode: str = "normal",
    output_width: int = OUTPUT_SIZE[0],
    output_height: int = OUTPUT_SIZE[1],
    image_format: str = "PNG",
    quality: int = 90,
) -> bytes:
    image = render_focus_image(
        source_image_path,
        annotation_path,
        target_label,
        mode=mode,
        output_size=(int(output_width), int(output_height)),
    )
    buffer = BytesIO()
    fmt = (image_format or "PNG").upper()
    save_kwargs: dict = {"format": fmt}
    if fmt == "JPEG":
        save_kwargs["quality"] = max(60, min(int(quality), 95))
        save_kwargs["optimize"] = True
        image = image.convert("RGB")
    elif fmt == "WEBP":
        save_kwargs["quality"] = max(60, min(int(quality), 95))
        save_kwargs["method"] = 6
    else:
        save_kwargs["optimize"] = True
    image.save(buffer, **save_kwargs)
    return buffer.getvalue()
