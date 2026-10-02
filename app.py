import io
import json
import zipfile
import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

st.set_page_config(
    page_title="Image Augmentation Research Lab",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧬 Image Data Augmentation Research Lab")
st.caption(
    "Research-oriented interactive laboratory based on the augmentation taxonomy "
    "discussed by Shorten & Khoshgoftaar (2019)."
)

# ---------------------------------------------------------------------------
# Core image operators
# ---------------------------------------------------------------------------

@st.cache_data
def load_image(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def image_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    img.save(buffer, format=fmt)
    return buffer.getvalue()


def resize_for_display(img: Image.Image, max_side: int = 1100) -> Image.Image:
    out = img.copy()
    out.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return out


def rotate(img, degrees):
    return img.rotate(
        degrees, expand=True, fillcolor=(255, 255, 255),
        resample=Image.Resampling.BICUBIC
    )


def translate(img, x, y):
    return img.transform(
        img.size, Image.Transform.AFFINE,
        (1, 0, -x, 0, 1, -y),
        resample=Image.Resampling.BICUBIC,
        fillcolor=(255, 255, 255)
    )


def shear(img, x_shear, y_shear):
    return img.transform(
        img.size, Image.Transform.AFFINE,
        (1, x_shear, 0, y_shear, 1, 0),
        resample=Image.Resampling.BICUBIC,
        fillcolor=(255, 255, 255)
    )


def random_erasing(img, area_percent, seed, fill_mode="black"):
    rng = np.random.default_rng(int(seed))
    arr = np.asarray(img).copy()
    h, w, _ = arr.shape
    target = max(1, int(h * w * area_percent / 100))
    ratio = float(rng.uniform(0.5, 2.0))
    eh = max(1, min(h, int(np.sqrt(target * ratio))))
    ew = max(1, min(w, int(np.sqrt(target / ratio))))
    y = int(rng.integers(0, max(1, h - eh + 1)))
    x = int(rng.integers(0, max(1, w - ew + 1)))

    if fill_mode == "random":
        arr[y:y + eh, x:x + ew] = rng.integers(
            0, 256, size=(eh, ew, 3), dtype=np.uint8
        )
    elif fill_mode == "mean":
        arr[y:y + eh, x:x + ew] = np.mean(arr, axis=(0, 1)).astype(np.uint8)
    else:
        arr[y:y + eh, x:x + ew] = 0

    return Image.fromarray(arr)


def apply_operation(img, operation, p):
    if operation == "Horizontal flip":
        return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if operation == "Vertical flip":
        return img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    if operation == "Rotation":
        return rotate(img, p["degrees"])
    if operation == "Translation":
        return translate(img, p["x"], p["y"])
    if operation == "Shear":
        return shear(img, p["x"], p["y"])
    if operation == "Crop + resize":
        w, h = img.size
        frac = p["fraction"]
        nw, nh = int(w * frac), int(h * frac)
        left, top = (w - nw) // 2, (h - nh) // 2
        return img.crop((left, top, left + nw, top + nh)).resize(
            (w, h), Image.Resampling.LANCZOS
        )

    if operation == "Brightness":
        return ImageEnhance.Brightness(img).enhance(p["value"])
    if operation == "Contrast":
        return ImageEnhance.Contrast(img).enhance(p["value"])
    if operation == "Saturation":
        return ImageEnhance.Color(img).enhance(p["value"])
    if operation == "Sharpness":
        return ImageEnhance.Sharpness(img).enhance(p["value"])
    if operation == "Grayscale":
        return ImageOps.grayscale(img).convert("RGB")
    if operation == "Invert":
        return ImageOps.invert(img)
    if operation == "Solarize":
        return ImageOps.solarize(img, threshold=p["threshold"])
    if operation == "Posterize":
        return ImageOps.posterize(img, bits=p["bits"])

    if operation == "Blur":
        return img.filter(ImageFilter.BLUR)
    if operation == "Gaussian blur":
        return img.filter(ImageFilter.GaussianBlur(p["radius"]))
    if operation == "Sharpen":
        return img.filter(ImageFilter.SHARPEN)
    if operation == "Edge enhance":
        return img.filter(ImageFilter.EDGE_ENHANCE)
    if operation == "Emboss":
        return img.filter(ImageFilter.EMBOSS)
    if operation == "Edge kernel":
        return img.filter(
            ImageFilter.Kernel(
                (3, 3), [-1, -1, -1, -1, 8, -1, -1, -1, -1],
                scale=1, offset=128
            )
        )
    if operation == "Random erasing":
        return random_erasing(
            img, p["area"], p["seed"], p["fill_mode"]
        )

    return img.copy()


OPERATIONS = {
    "Geometric": [
        "Horizontal flip", "Vertical flip", "Rotation",
        "Translation", "Shear", "Crop + resize"
    ],
    "Colour-space": [
        "Brightness", "Contrast", "Saturation", "Sharpness",
        "Grayscale", "Invert", "Solarize", "Posterize"
    ],
    "Kernel filters": [
        "Blur", "Gaussian blur", "Sharpen", "Edge enhance",
        "Emboss", "Edge kernel"
    ],
    "Occlusion": ["Random erasing"],
}

ALL_OPERATIONS = [x for group in OPERATIONS.values() for x in group]


# ---------------------------------------------------------------------------
# Research diagnostics
# ---------------------------------------------------------------------------

def rgb_histogram(img, bins=32):
    arr = np.asarray(img.resize((256, 256)))
    output = []
    for c in range(3):
        h, _ = np.histogram(arr[:, :, c], bins=bins, range=(0, 256), density=True)
        output.append(h)
    return np.concatenate(output)


def grayscale_array(img):
    arr = np.asarray(img.resize((256, 256)), dtype=np.float32)
    return 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]


def edge_density(img):
    g = grayscale_array(img)
    gx = np.abs(np.diff(g, axis=1))
    gy = np.abs(np.diff(g, axis=0))
    return float((np.mean(gx) + np.mean(gy)) / 2.0 / 255.0)


def mse(a, b):
    x = np.asarray(a.resize(b.size), dtype=np.float32)
    y = np.asarray(b, dtype=np.float32)
    return float(np.mean((x - y) ** 2))


def mae(a, b):
    x = np.asarray(a.resize(b.size), dtype=np.float32)
    y = np.asarray(b, dtype=np.float32)
    return float(np.mean(np.abs(x - y)))


def psnr(a, b):
    value = mse(a, b)
    return float("inf") if value == 0 else float(10 * np.log10((255.0 ** 2) / value))


def histogram_distance(a, b):
    ha = rgb_histogram(a)
    hb = rgb_histogram(b)
    denom = np.linalg.norm(ha) * np.linalg.norm(hb)
    return float(1 - np.dot(ha, hb) / denom) if denom else 0.0


def diagnostics(original, augmented):
    return {
        "MSE": mse(original, augmented),
        "MAE": mae(original, augmented),
        "PSNR_dB": psnr(original, augmented),
        "Histogram_distance": histogram_distance(original, augmented),
        "Edge_density_original": edge_density(original),
        "Edge_density_augmented": edge_density(augmented),
        "Edge_density_delta": edge_density(augmented) - edge_density(original),
    }


# ---------------------------------------------------------------------------
# Stochastic research policy
# ---------------------------------------------------------------------------

@dataclass
class PolicyStep:
    operation: str
    probability: float
    magnitude: float


def stochastic_policy(img, steps: List[PolicyStep], seed: int):
    rng = np.random.default_rng(int(seed))
    result = img.copy()
    trace = []

    for step in steps:
        applied = bool(rng.random() <= step.probability)
        if not applied:
            trace.append(f"SKIP:{step.operation}")
            continue

        m = step.magnitude
        op = step.operation

        if op == "Horizontal flip":
            params = {}
        elif op == "Rotation":
            params = {"degrees": float(rng.uniform(-30, 30) * m)}
        elif op == "Translation":
            params = {
                "x": int(rng.uniform(-80, 80) * m),
                "y": int(rng.uniform(-80, 80) * m),
            }
        elif op == "Shear":
            params = {
                "x": float(rng.uniform(-0.35, 0.35) * m),
                "y": float(rng.uniform(-0.35, 0.35) * m),
            }
        elif op == "Brightness":
            params = {"value": float(1 + rng.uniform(-0.4, 0.4) * m)}
        elif op == "Contrast":
            params = {"value": float(1 + rng.uniform(-0.4, 0.4) * m)}
        elif op == "Saturation":
            params = {"value": float(1 + rng.uniform(-0.5, 0.5) * m)}
        elif op == "Gaussian blur":
            params = {"radius": float(max(0.05, rng.uniform(0.2, 4.0) * m))}
        elif op == "Random erasing":
            params = {
                "area": float(rng.uniform(5, 25) * m),
                "seed": int(rng.integers(0, 10_000_000)),
                "fill_mode": "random",
            }
        elif op == "Crop + resize":
            params = {"fraction": float(1 - rng.uniform(0.05, 0.35) * m)}
        else:
            params = {}

        result = apply_operation(result, op, params)
        trace.append(f"APPLY:{op}")

    return result, trace


def random_policy(rng, length):
    choices = [
        "Horizontal flip", "Rotation", "Translation", "Brightness",
        "Contrast", "Saturation", "Gaussian blur", "Random erasing",
        "Crop + resize"
    ]
    return [
        PolicyStep(
            operation=str(rng.choice(choices)),
            probability=float(rng.uniform(0.25, 1.0)),
            magnitude=float(rng.uniform(0.25, 1.0)),
        )
        for _ in range(length)
    ]


def policy_diversity_score(original, augmented):
    d = diagnostics(original, augmented)
    # A diagnostic score, not a model-performance metric.
    return float(
        0.40 * np.clip(d["MAE"] / 80.0, 0, 1)
        + 0.30 * np.clip(d["Histogram_distance"] * 5, 0, 1)
        + 0.30 * np.clip(abs(d["Edge_density_delta"]) * 10, 0, 1)
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

uploaded = st.file_uploader(
    "Upload a JPG or PNG image",
    type=["jpg", "jpeg", "png"],
    help="The image is used by the current Streamlit session for experimentation."
)

if not uploaded:
    st.info("Upload an image to start the research laboratory.")
    st.stop()

original = load_image(uploaded.getvalue())

tabs = st.tabs([
    "🎛️ Playground",
    "🔗 Policy pipeline",
    "🧬 Policy search",
    "📈 Sensitivity",
    "📦 Dataset generator",
    "🧪 Experiment matrix",
    "🔬 Diagnostics",
])

# ---------------------------------------------------------------------------
# Playground
# ---------------------------------------------------------------------------
with tabs[0]:
    c1, c2 = st.columns([1, 2])

    with c1:
        category = st.selectbox("Technique family", list(OPERATIONS), key="pg_category")
        operation = st.selectbox("Operation", OPERATIONS[category], key="pg_operation")

        p = {}
        if operation == "Rotation":
            p["degrees"] = st.slider("Angle", -180, 180, 25)
        elif operation == "Translation":
            p["x"] = st.slider("X", -100, 100, 20)
            p["y"] = st.slider("Y", -100, 100, 20)
        elif operation == "Shear":
            p["x"] = st.slider("X shear", -0.5, 0.5, 0.0, 0.01)
            p["y"] = st.slider("Y shear", -0.5, 0.5, 0.0, 0.01)
        elif operation == "Crop + resize":
            p["fraction"] = st.slider("Crop fraction", 0.5, 1.0, 0.8, 0.05)
        elif operation in {"Brightness", "Contrast", "Saturation", "Sharpness"}:
            p["value"] = st.slider("Intensity", 0.0, 2.0, 1.0, 0.05)
        elif operation == "Gaussian blur":
            p["radius"] = st.slider("Radius", 0.1, 10.0, 2.0, 0.1)
        elif operation == "Solarize":
            p["threshold"] = st.slider("Threshold", 0, 255, 128)
        elif operation == "Posterize":
            p["bits"] = st.slider("Bits/channel", 1, 8, 4)
        elif operation == "Random erasing":
            p["area"] = st.slider("Area (%)", 1, 40, 10)
            p["seed"] = st.number_input("Seed", 0, 999999, 42)
            p["fill_mode"] = st.selectbox("Fill", ["black", "mean", "random"])

        result = apply_operation(original, operation, p)

    with c2:
        a, b = st.columns(2)
        with a:
            st.markdown("**Original**")
            st.image(resize_for_display(original), use_container_width=True)
        with b:
            st.markdown("**Augmented**")
            st.image(resize_for_display(result), use_container_width=True)

    d = diagnostics(original, result)
    cols = st.columns(5)
    cols[0].metric("MSE", f"{d['MSE']:.2f}")
    cols[1].metric("MAE", f"{d['MAE']:.2f}")
    cols[2].metric("PSNR", "∞" if np.isinf(d["PSNR_dB"]) else f"{d['PSNR_dB']:.2f} dB")
    cols[3].metric("Histogram Δ", f"{d['Histogram_distance']:.4f}")
    cols[4].metric("Edge Δ", f"{d['Edge_density_delta']:.4f}")

    st.download_button(
        "⬇️ Download result",
        image_bytes(result),
        "augmentation_result.png",
        "image/png",
    )

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Explicit augmentation policy")
    st.write(
        "Compose operations in a controlled order. Each stage has a probability and "
        "magnitude, making the pipeline closer to a research experiment than a fixed demo."
    )

    n = st.slider("Number of policy stages", 1, 6, 3)
    steps = []

    for i in range(n):
        with st.expander(f"Stage {i + 1}", expanded=i < 2):
            op = st.selectbox(
                "Operation",
                ALL_OPERATIONS,
                index=min(i, len(ALL_OPERATIONS) - 1),
                key=f"policy_op_{i}",
            )
            probability = st.slider(
                "Application probability", 0.0, 1.0, 0.7,
                0.05, key=f"policy_prob_{i}"
            )
            magnitude = st.slider(
                "Magnitude", 0.0, 1.0, 0.7,
                0.05, key=f"policy_mag_{i}"
            )
            steps.append(PolicyStep(op, probability, magnitude))

    seed = st.number_input("Policy seed", 0, 999999, 2026, key="policy_seed")
    output, trace = stochastic_policy(original, steps, seed)

    left, right = st.columns(2)
    with left:
        st.image(resize_for_display(original), caption="Input", use_container_width=True)
    with right:
        st.image(resize_for_display(output), caption="Policy output", use_container_width=True)

    st.code("\n".join(trace), language="text")

    policy_json = json.dumps([asdict(x) for x in steps], indent=2)
    st.download_button(
        "🧾 Export policy JSON",
        policy_json,
        "augmentation_policy.json",
        "application/json",
    )

# ---------------------------------------------------------------------------
# Policy search
# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Monte-Carlo augmentation policy search")
    st.write(
        "This module samples stochastic policies and ranks them using an explicit "
        "image-diversity diagnostic. It does **not** claim to optimise neural-network accuracy."
    )

    trials = st.slider("Number of sampled policies", 5, 100, 20)
    stages = st.slider("Stages per policy", 1, 5, 3)
    search_seed = st.number_input("Search seed", 0, 999999, 1234)

    if st.button("Run policy search", type="primary"):
        rng = np.random.default_rng(search_seed)
        rows = []

        progress = st.progress(0)
        for i in range(trials):
            policy = random_policy(rng, stages)
            output, trace = stochastic_policy(
                original, policy, int(rng.integers(0, 10_000_000))
            )
            d = diagnostics(original, output)

            rows.append({
                "trial": i + 1,
                "diversity_score": policy_diversity_score(original, output),
                "MAE": d["MAE"],
                "PSNR_dB": d["PSNR_dB"],
                "histogram_distance": d["Histogram_distance"],
                "edge_delta": d["Edge_density_delta"],
                "policy": " | ".join(x.operation for x in policy),
            })
            progress.progress((i + 1) / trials)

        df = pd.DataFrame(rows).sort_values(
            "diversity_score", ascending=False
        ).reset_index(drop=True)

        st.session_state["search_results"] = df

    if "search_results" in st.session_state:
        df = st.session_state["search_results"]
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.line_chart(df.set_index("trial")["diversity_score"])

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📄 Download policy-search results CSV",
            csv,
            "policy_search_results.csv",
            "text/csv",
        )

# ---------------------------------------------------------------------------
# Sensitivity
# ---------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Augmentation magnitude sensitivity")
    operation = st.selectbox(
        "Sensitivity operation",
        ["Rotation", "Brightness", "Contrast", "Saturation", "Gaussian blur", "Random erasing"],
    )
    points = st.slider("Evaluation points", 5, 25, 11)

    rows = []
    magnitudes = np.linspace(0.0, 1.0, points)

    for m in magnitudes:
        if operation == "Rotation":
            p = {"degrees": float(-45 + 90 * m)}
        elif operation == "Brightness":
            p = {"value": float(0.6 + 0.8 * m)}
        elif operation == "Contrast":
            p = {"value": float(0.6 + 0.8 * m)}
        elif operation == "Saturation":
            p = {"value": float(0.5 + 1.0 * m)}
        elif operation == "Gaussian blur":
            p = {"radius": float(0.1 + 5.0 * m)}
        else:
            p = {
                "area": float(1 + 35 * m),
                "seed": 2026,
                "fill_mode": "random",
            }

        out = apply_operation(original, operation, p)
        d = diagnostics(original, out)
        rows.append({
            "magnitude": m,
            "MAE": d["MAE"],
            "PSNR_dB": d["PSNR_dB"],
            "histogram_distance": d["Histogram_distance"],
            "edge_density_delta": d["Edge_density_delta"],
        })

    sensitivity_df = pd.DataFrame(rows)
    st.dataframe(sensitivity_df, use_container_width=True, hide_index=True)
    st.line_chart(
        sensitivity_df.set_index("magnitude")[["MAE", "histogram_distance"]]
    )

# ---------------------------------------------------------------------------
# Dataset generator
# ---------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Controlled augmentation dataset generation")

    extra = st.file_uploader(
        "Optional additional source images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    source_images = [original]
    if extra:
        source_images.extend(load_image(x.getvalue()) for x in extra)

    variants_per_image = st.slider("Variants per source", 1, 10, 3)
    dataset_seed = st.number_input("Dataset seed", 0, 999999, 2026)

    if st.button("Generate dataset package", type="primary"):
        rng = np.random.default_rng(dataset_seed)
        generated = []

        for source_idx, source in enumerate(source_images):
            for variant_idx in range(variants_per_image):
                policy = random_policy(rng, int(rng.integers(1, 4)))
                out, trace = stochastic_policy(
                    source, policy, int(rng.integers(0, 10_000_000))
                )
                generated.append(
                    (source_idx, variant_idx, out, " | ".join(trace))
                )

        st.session_state["generated_dataset"] = generated

    generated = st.session_state.get("generated_dataset", [])
    if generated:
        preview = st.columns(4)
        for i, (_, _, img, trace) in enumerate(generated[:12]):
            with preview[i % 4]:
                st.image(resize_for_display(img, 500), use_container_width=True)
                st.caption(trace)

        zip_buffer = io.BytesIO()
        manifest = []

        with zipfile.ZipFile(
            zip_buffer, "w", zipfile.ZIP_DEFLATED
        ) as archive:
            for source_idx, variant_idx, img, trace in generated:
                filename = f"source_{source_idx:03d}_variant_{variant_idx:03d}.png"
                archive.writestr(filename, image_bytes(img))
                manifest.append({
                    "filename": filename,
                    "source_index": source_idx,
                    "variant_index": variant_idx,
                    "trace": trace,
                })

            archive.writestr(
                "manifest.csv",
                pd.DataFrame(manifest).to_csv(index=False)
            )

        st.download_button(
            "📦 Download generated dataset + manifest",
            zip_buffer.getvalue(),
            "augmentation_dataset.zip",
            "application/zip",
        )

# ---------------------------------------------------------------------------
# Experiment matrix
# ---------------------------------------------------------------------------
with tabs[5]:
    st.subheader("🧪 Controlled experiment matrix")
    st.write(
        "Run the same image through multiple augmentation families under a fixed "
        "parameter protocol. Results are image-level diagnostics, not model-accuracy evidence."
    )

    selected_ops = st.multiselect(
        "Operations to compare",
        ALL_OPERATIONS,
        default=[
            "Horizontal flip", "Rotation", "Brightness",
            "Contrast", "Gaussian blur", "Random erasing",
        ],
    )
    repetitions = st.slider("Repetitions per operation", 1, 20, 5)
    matrix_seed = st.number_input("Experiment seed", 0, 999999, 2026, key="matrix_seed")

    if st.button("Run controlled comparison", type="primary"):
        rows = []
        base_rng = np.random.default_rng(matrix_seed)

        for operation in selected_ops:
            for repetition in range(repetitions):
                seed = int(base_rng.integers(0, 10_000_000))

                if operation == "Horizontal flip" or operation == "Vertical flip":
                    params = {}
                elif operation == "Rotation":
                    params = {"degrees": 30.0}
                elif operation == "Translation":
                    params = {"x": 25, "y": 25}
                elif operation == "Shear":
                    params = {"x": 0.15, "y": 0.0}
                elif operation == "Crop + resize":
                    params = {"fraction": 0.8}
                elif operation in {"Brightness", "Contrast", "Saturation"}:
                    params = {"value": 1.25}
                elif operation == "Sharpness":
                    params = {"value": 1.5}
                elif operation == "Grayscale" or operation == "Invert":
                    params = {}
                elif operation == "Solarize":
                    params = {"threshold": 128}
                elif operation == "Posterize":
                    params = {"bits": 4}
                elif operation in {"Blur", "Sharpen", "Edge enhance", "Emboss", "Edge kernel"}:
                    params = {}
                elif operation == "Gaussian blur":
                    params = {"radius": 2.0}
                elif operation == "Random erasing":
                    params = {"area": 10.0, "seed": seed, "fill_mode": "random"}
                else:
                    params = {}

                output = apply_operation(original, operation, params)
                rows.append({
                    "operation": operation,
                    "repetition": repetition + 1,
                    "seed": seed,
                    **diagnostics(original, output),
                })

        st.session_state["experiment_matrix"] = pd.DataFrame(rows)

    if "experiment_matrix" in st.session_state:
        matrix_df = st.session_state["experiment_matrix"]
        summary = (
            matrix_df.groupby("operation")
            .agg(
                MAE_mean=("MAE", "mean"),
                MAE_std=("MAE", "std"),
                PSNR_mean=("PSNR_dB", "mean"),
                histogram_mean=("Histogram_distance", "mean"),
                edge_delta_mean=("Edge_density_delta", "mean"),
            )
            .reset_index()
            .fillna(0)
        )

        st.markdown("### Summary statistics")
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.markdown("### Mean image-change magnitude")
        st.bar_chart(summary.set_index("operation")["MAE_mean"])
        st.markdown("### Repeated-run observations")
        st.dataframe(matrix_df, use_container_width=True, hide_index=True)

        experiment_id = hashlib.sha256(
            matrix_df.to_csv(index=False).encode("utf-8")
        ).hexdigest()[:12]
        metadata = {
            "experiment_id": experiment_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "seed": int(matrix_seed),
            "repetitions": int(repetitions),
            "image_size": list(original.size),
            "note": "Image-level diagnostics only; no model-performance claim.",
        }

        st.code(json.dumps(metadata, indent=2), language="json")
        st.download_button(
            "📊 Download experiment matrix CSV",
            matrix_df.to_csv(index=False).encode("utf-8"),
            "experiment_matrix.csv",
            "text/csv",
        )
        st.download_button(
            "🧾 Download experiment metadata JSON",
            json.dumps(metadata, indent=2),
            "experiment_metadata.json",
            "application/json",
        )

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
with tabs[6]:
    st.subheader("Research diagnostics and interpretation")

    arr = np.asarray(original.resize((256, 256)))
    channel_df = pd.DataFrame({
        "channel": ["Red", "Green", "Blue"],
        "mean": arr.mean(axis=(0, 1)),
        "std": arr.std(axis=(0, 1)),
        "min": arr.min(axis=(0, 1)),
        "max": arr.max(axis=(0, 1)),
    })
    st.dataframe(channel_df, use_container_width=True, hide_index=True)

    hist = rgb_histogram(original).reshape(3, 32)
    hist_df = pd.DataFrame(
        hist.T,
        columns=["Red", "Green", "Blue"]
    )
    st.line_chart(hist_df)

    st.markdown(
        """
        ### Interpretation boundary

        Pixel-level diagnostics quantify **how much an image changed**. They do not
        establish whether the transformed image is semantically valid, whether a
        class label remains correct, or whether a CNN will generalise better.

        For a genuine machine-learning performance study, the generated data should
        be evaluated with a fixed model architecture, controlled train/validation/test
        split, repeated random seeds, confidence intervals and task-specific metrics.
        This distinction is important because augmentation quality is dataset- and
        task-dependent.
        """
    )

with st.expander("📚 Research scope and relation to the survey"):
    st.markdown(
        """
        The source survey organises image augmentation around **data warping** and
        **oversampling** and discusses geometric transformations, colour-space
        transformations, kernel filters, image mixing, random erasing, feature-space
        augmentation, adversarial training, GAN-based augmentation, neural style
        transfer and meta-learning.

        This laboratory implements a substantial subset of image-space techniques and
        adds experimental infrastructure for reproducibility, stochastic policies,
        sensitivity analysis and controlled dataset generation.

        The policy-search module is an **experimental engineering extension**. It is
        not presented as a reproduction of AutoAugment, GAN training, adversarial
        training, or the numerical experiments reported in the survey.
        """
    )
