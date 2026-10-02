import io
import zipfile
from typing import Callable, Dict, Tuple

import numpy as np
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

st.set_page_config(
    page_title="Image Data Augmentation Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧪 Image Data Augmentation Lab")
st.caption(
    "Advanced interactive companion for the Shorten & Khoshgoftaar (2019) "
    "image data augmentation survey."
)

st.markdown(
    """
    <style>
    .metric-card {padding: 0.7rem 1rem; border: 1px solid #ddd; border-radius: 10px;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_image(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def image_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    img.save(buffer, format=fmt)
    return buffer.getvalue()


def resize_for_display(img: Image.Image, max_side: int = 1200) -> Image.Image:
    copy = img.copy()
    copy.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return copy


def rotate(img: Image.Image, degrees: float) -> Image.Image:
    return img.rotate(
        degrees,
        expand=True,
        fillcolor=(255, 255, 255),
        resample=Image.Resampling.BICUBIC,
    )


def translate(img: Image.Image, x: int, y: int) -> Image.Image:
    return img.transform(
        img.size,
        Image.Transform.AFFINE,
        (1, 0, -x, 0, 1, -y),
        resample=Image.Resampling.BICUBIC,
        fillcolor=(255, 255, 255),
    )


def shear(img: Image.Image, x_shear: float, y_shear: float) -> Image.Image:
    return img.transform(
        img.size,
        Image.Transform.AFFINE,
        (1, x_shear, 0, y_shear, 1, 0),
        resample=Image.Resampling.BICUBIC,
        fillcolor=(255, 255, 255),
    )


def random_erasing(
    img: Image.Image,
    area_percent: float,
    seed: int,
    fill_mode: str = "black",
) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = np.array(img).copy()
    h, w, _ = arr.shape
    target = max(1, int(h * w * area_percent / 100))
    ratio = float(rng.uniform(0.5, 2.0))
    eh = max(1, min(h, int(np.sqrt(target * ratio))))
    ew = max(1, min(w, int(np.sqrt(target / ratio))))
    y = int(rng.integers(0, max(1, h - eh + 1)))
    x = int(rng.integers(0, max(1, w - ew + 1)))

    if fill_mode == "random":
        arr[y : y + eh, x : x + ew] = rng.integers(
            0, 256, size=(eh, ew, 3), dtype=np.uint8
        )
    elif fill_mode == "mean":
        arr[y : y + eh, x : x + ew] = np.mean(arr, axis=(0, 1)).astype(np.uint8)
    else:
        arr[y : y + eh, x : x + ew] = 0

    return Image.fromarray(arr)


def center_crop(img: Image.Image, fraction: float) -> Image.Image:
    w, h = img.size
    nw, nh = int(w * fraction), int(h * fraction)
    left, top = (w - nw) // 2, (h - nh) // 2
    return img.crop((left, top, left + nw, top + nh)).resize(
        (w, h), Image.Resampling.LANCZOS
    )


def apply_operation(img: Image.Image, operation: str, params: Dict) -> Image.Image:
    if operation == "Horizontal flip":
        return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if operation == "Vertical flip":
        return img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    if operation == "Rotation":
        return rotate(img, params["degrees"])
    if operation == "Translation":
        return translate(img, params["x"], params["y"])
    if operation == "Shear":
        return shear(img, params["x"], params["y"])
    if operation == "Center crop + resize":
        return center_crop(img, params["fraction"])

    if operation == "Brightness":
        return ImageEnhance.Brightness(img).enhance(params["value"])
    if operation == "Contrast":
        return ImageEnhance.Contrast(img).enhance(params["value"])
    if operation == "Colour saturation":
        return ImageEnhance.Color(img).enhance(params["value"])
    if operation == "Sharpness":
        return ImageEnhance.Sharpness(img).enhance(params["value"])
    if operation == "Grayscale":
        return ImageOps.grayscale(img).convert("RGB")
    if operation == "Invert":
        return ImageOps.invert(img)
    if operation == "Solarize":
        return ImageOps.solarize(img, threshold=params["threshold"])
    if operation == "Posterize":
        return ImageOps.posterize(img, bits=params["bits"])

    if operation == "Blur":
        return img.filter(ImageFilter.BLUR)
    if operation == "Gaussian blur":
        return img.filter(ImageFilter.GaussianBlur(radius=params["radius"]))
    if operation == "Sharpen":
        return img.filter(ImageFilter.SHARPEN)
    if operation == "Edge enhance":
        return img.filter(ImageFilter.EDGE_ENHANCE)
    if operation == "Emboss":
        return img.filter(ImageFilter.EMBOSS)
    if operation == "Custom edge kernel":
        return img.filter(
            ImageFilter.Kernel(
                (3, 3),
                [-1, -1, -1, -1, 8, -1, -1, -1, -1],
                scale=1,
                offset=128,
            )
        )

    if operation == "Random erasing":
        return random_erasing(
            img,
            params["area"],
            params["seed"],
            params["fill_mode"],
        )

    return img.copy()


def operation_parameters(operation: str, prefix: str = "") -> Dict:
    key = prefix or operation
    if operation == "Rotation":
        return {"degrees": st.slider(f"{key} — degrees", -180, 180, 25, key=f"{key}_deg")}
    if operation == "Translation":
        return {
            "x": st.slider(f"{key} — X", -100, 100, 20, key=f"{key}_x"),
            "y": st.slider(f"{key} — Y", -100, 100, 20, key=f"{key}_y"),
        }
    if operation == "Shear":
        return {
            "x": st.slider(f"{key} — X shear", -0.5, 0.5, 0.0, 0.01, key=f"{key}_xs"),
            "y": st.slider(f"{key} — Y shear", -0.5, 0.5, 0.0, 0.01, key=f"{key}_ys"),
        }
    if operation == "Center crop + resize":
        return {"fraction": st.slider(f"{key} — crop fraction", 0.5, 1.0, 0.8, 0.05, key=f"{key}_crop")}
    if operation in {"Brightness", "Contrast", "Colour saturation", "Sharpness"}:
        return {"value": st.slider(f"{key} — intensity", 0.0, 2.0, 1.0, 0.05, key=f"{key}_int")}
    if operation == "Gaussian blur":
        return {"radius": st.slider(f"{key} — radius", 0.1, 10.0, 2.0, 0.1, key=f"{key}_radius")}
    if operation == "Solarize":
        return {"threshold": st.slider(f"{key} — threshold", 0, 255, 128, key=f"{key}_threshold")}
    if operation == "Posterize":
        return {"bits": st.slider(f"{key} — bits/channel", 1, 8, 4, key=f"{key}_bits")}
    if operation == "Random erasing":
        return {
            "area": st.slider(f"{key} — erased area (%)", 1, 40, 10, key=f"{key}_area"),
            "seed": st.number_input(f"{key} — seed", 0, 999999, 42, key=f"{key}_seed"),
            "fill_mode": st.selectbox(
                f"{key} — fill",
                ["black", "mean", "random"],
                key=f"{key}_fill",
            ),
        }
    return {}


def mse(a: Image.Image, b: Image.Image) -> float:
    x = np.asarray(a.resize(b.size), dtype=np.float32)
    y = np.asarray(b, dtype=np.float32)
    return float(np.mean((x - y) ** 2))


def mae(a: Image.Image, b: Image.Image) -> float:
    x = np.asarray(a.resize(b.size), dtype=np.float32)
    y = np.asarray(b, dtype=np.float32)
    return float(np.mean(np.abs(x - y)))


def psnr(a: Image.Image, b: Image.Image) -> float:
    value = mse(a, b)
    if value == 0:
        return float("inf")
    return float(10 * np.log10((255.0**2) / value))


def histogram_frame(img: Image.Image):
    arr = np.asarray(img)
    hist = {}
    for i, channel in enumerate(["Red", "Green", "Blue"]):
        counts, _ = np.histogram(arr[:, :, i], bins=32, range=(0, 256))
        hist[channel] = counts
    return hist


uploaded = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
    help="Use a non-sensitive image. The demo processes the image in the app session.",
)

if not uploaded:
    st.info("Upload a JPG or PNG image to open the augmentation laboratory.")
    st.stop()

original = load_image(uploaded.getvalue())
st.session_state.setdefault("original_image", original)

tab_playground, tab_pipeline, tab_batch, tab_analysis = st.tabs(
    ["🎛️ Playground", "🔗 Pipeline", "📦 Batch generation", "📊 Analysis"]
)

# ---------------------------------------------------------------------------
# Playground
# ---------------------------------------------------------------------------
with tab_playground:
    st.subheader("Single-technique laboratory")

    categories = {
        "Geometric": [
            "Horizontal flip",
            "Vertical flip",
            "Rotation",
            "Translation",
            "Shear",
            "Center crop + resize",
        ],
        "Colour-space": [
            "Brightness",
            "Contrast",
            "Colour saturation",
            "Sharpness",
            "Grayscale",
            "Invert",
            "Solarize",
            "Posterize",
        ],
        "Kernel filter": [
            "Blur",
            "Gaussian blur",
            "Sharpen",
            "Edge enhance",
            "Emboss",
            "Custom edge kernel",
        ],
        "Occlusion / erasing": ["Random erasing"],
    }

    c1, c2 = st.columns([1, 2])
    with c1:
        category = st.selectbox("Category", list(categories))
        operation = st.selectbox("Technique", categories[category])
        params = operation_parameters(operation, "playground")
        result = apply_operation(original, operation, params)

    with c2:
        left, right = st.columns(2)
        with left:
            st.markdown("**Original**")
            st.image(resize_for_display(original), use_container_width=True)
        with right:
            st.markdown("**Augmented**")
            st.image(resize_for_display(result), use_container_width=True)

    metric_cols = st.columns(3)
    metric_cols[0].metric("MSE", f"{mse(original, result):.2f}")
    metric_cols[1].metric("MAE", f"{mae(original, result):.2f}")
    metric_cols[2].metric("PSNR", "∞" if np.isinf(psnr(original, result)) else f"{psnr(original, result):.2f} dB")

    st.download_button(
        "⬇️ Download augmented PNG",
        image_bytes(result),
        "augmented_image.png",
        "image/png",
    )

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
with tab_pipeline:
    st.subheader("Multi-stage augmentation pipeline")
    st.write("Apply several image-space operations sequentially and inspect the cumulative result.")

    available = [
        "Horizontal flip",
        "Vertical flip",
        "Rotation",
        "Translation",
        "Shear",
        "Center crop + resize",
        "Brightness",
        "Contrast",
        "Colour saturation",
        "Sharpness",
        "Grayscale",
        "Invert",
        "Solarize",
        "Posterize",
        "Blur",
        "Gaussian blur",
        "Sharpen",
        "Edge enhance",
        "Emboss",
        "Custom edge kernel",
        "Random erasing",
    ]

    selected = st.multiselect(
        "Pipeline operations (order matters)",
        available,
        default=["Horizontal flip", "Rotation", "Contrast"],
        max_selections=6,
    )

    pipeline_result = original.copy()
    pipeline_log = []

    for index, operation in enumerate(selected, start=1):
        with st.expander(f"Stage {index}: {operation}", expanded=index <= 2):
            params = operation_parameters(operation, f"stage{index}")
            pipeline_result = apply_operation(pipeline_result, operation, params)
            pipeline_log.append(operation)

    if selected:
        a, b = st.columns(2)
        with a:
            st.markdown("**Input**")
            st.image(resize_for_display(original), use_container_width=True)
        with b:
            st.markdown("**Pipeline output**")
            st.image(resize_for_display(pipeline_result), use_container_width=True)

        st.success(" → ".join(pipeline_log))
        st.download_button(
            "⬇️ Download pipeline output",
            image_bytes(pipeline_result),
            "pipeline_augmented.png",
            "image/png",
        )
    else:
        st.warning("Select at least one operation.")

# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------
with tab_batch:
    st.subheader("Synthetic variant generator")
    st.write(
        "Generate multiple deterministic variants from the uploaded image. "
        "This is a demonstration of augmentation diversity, not model training."
    )

    batch_size = st.slider("Number of variants", 2, 20, 6)
    seed = st.number_input("Base random seed", 0, 999999, 2026)

    random_ops = [
        "Horizontal flip",
        "Rotation",
        "Brightness",
        "Contrast",
        "Colour saturation",
        "Gaussian blur",
        "Random erasing",
    ]

    if st.button("Generate variants", type="primary"):
        rng = np.random.default_rng(seed)
        variants = []

        for i in range(batch_size):
            variant = original.copy()
            op = random_ops[int(rng.integers(0, len(random_ops)))]

            if op == "Horizontal flip":
                variant = apply_operation(variant, op, {})
            elif op == "Rotation":
                variant = rotate(variant, float(rng.integers(-35, 36)))
            elif op == "Brightness":
                variant = ImageEnhance.Brightness(variant).enhance(float(rng.uniform(0.7, 1.3)))
            elif op == "Contrast":
                variant = ImageEnhance.Contrast(variant).enhance(float(rng.uniform(0.7, 1.3)))
            elif op == "Colour saturation":
                variant = ImageEnhance.Color(variant).enhance(float(rng.uniform(0.6, 1.4)))
            elif op == "Gaussian blur":
                variant = variant.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.2, 3.0))))
            elif op == "Random erasing":
                variant = random_erasing(variant, float(rng.uniform(5, 20)), int(seed + i), "random")

            variants.append((op, variant))

        st.session_state["variants"] = variants

    variants = st.session_state.get("variants", [])
    if variants:
        cols = st.columns(3)
        for i, (op, variant) in enumerate(variants):
            with cols[i % 3]:
                st.caption(f"Variant {i + 1}: {op}")
                st.image(resize_for_display(variant, 600), use_container_width=True)

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for i, (op, variant) in enumerate(variants, start=1):
                archive.writestr(
                    f"variant_{i:02d}_{op.lower().replace(' ', '_')}.png",
                    image_bytes(variant),
                )

        st.download_button(
            "📦 Download all variants as ZIP",
            zip_buffer.getvalue(),
            "augmentation_variants.zip",
            "application/zip",
        )

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
with tab_analysis:
    st.subheader("Visual and numerical comparison")

    analysis_result = st.session_state.get("variants", [(None, original)])[0][1]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Original RGB histogram**")
        st.line_chart(histogram_frame(original))
    with c2:
        st.markdown("**Selected variant RGB histogram**")
        st.line_chart(histogram_frame(analysis_result))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Original size", f"{original.width} × {original.height}")
    m2.metric("Output size", f"{analysis_result.width} × {analysis_result.height}")
    m3.metric("Mean absolute error", f"{mae(original, analysis_result):.2f}")
    m4.metric(
        "PSNR",
        "∞" if np.isinf(psnr(original, analysis_result)) else f"{psnr(original, analysis_result):.2f} dB",
    )

    st.info(
        "These numerical metrics describe pixel-level difference from the input. "
        "They do not establish whether an augmentation improves a machine-learning model."
    )

with st.expander("📚 Academic scope and limitations"):
    st.markdown(
        """
        **Covered in this demonstration:** image-space transformations, colour-space
        transformations, kernel/filter operations, random erasing and simple image mixing.

        **Discussed in the survey but not implemented as claims of reproduction here:**
        feature-space augmentation, adversarial training, GAN-based augmentation,
        neural style transfer and meta-learning.

        The application is therefore a practical visual demonstration rather than a
        reproduction of the survey's experiments. Any reported accuracy/error values
        from the paper should be cited as literature results, not as results produced
        by this application.
        """
    )
