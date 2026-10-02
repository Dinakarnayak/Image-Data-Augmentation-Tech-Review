import io
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import streamlit as st

st.set_page_config(page_title="Image Data Augmentation Demo", page_icon="🧪", layout="wide")

st.title("🧪 Image Data Augmentation Demo")
st.caption("Interactive demonstration of image augmentation techniques discussed in the Shorten & Khoshgoftaar (2019) survey.")

@st.cache_data
def load_image(data):
    return Image.open(io.BytesIO(data)).convert("RGB")

def rotate(img, degrees):
    return img.rotate(degrees, expand=True, fillcolor=(255, 255, 255))

def translate(img, x, y):
    return img.transform(img.size, Image.Transform.AFFINE, (1, 0, -x, 0, 1, -y), fillcolor=(255, 255, 255))

def random_erasing(img, area_percent, seed):
    rng = np.random.default_rng(seed)
    arr = np.array(img).copy()
    h, w, _ = arr.shape
    area = h * w
    target = max(1, int(area * area_percent / 100))
    ratio = float(rng.uniform(0.5, 2.0))
    eh = max(1, min(h, int(np.sqrt(target * ratio))))
    ew = max(1, min(w, int(np.sqrt(target / ratio))))
    y = int(rng.integers(0, max(1, h - eh + 1)))
    x = int(rng.integers(0, max(1, w - ew + 1)))
    arr[y:y+eh, x:x+ew] = 0
    return Image.fromarray(arr)

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    original = load_image(uploaded.getvalue())

    st.sidebar.header("Augmentation controls")
    category = st.sidebar.selectbox(
        "Technique category",
        ["Geometric", "Colour-space", "Kernel filter", "Random erasing", "Mixing"]
    )

    result = original.copy()

    if category == "Geometric":
        technique = st.sidebar.selectbox("Technique", ["Horizontal flip", "Vertical flip", "Rotation", "Translation"])
        if technique == "Horizontal flip":
            result = original.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        elif technique == "Vertical flip":
            result = original.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        elif technique == "Rotation":
            result = rotate(original, st.sidebar.slider("Degrees", -180, 180, 25))
        else:
            x = st.sidebar.slider("X translation", -100, 100, 20)
            y = st.sidebar.slider("Y translation", -100, 100, 20)
            result = translate(original, x, y)

    elif category == "Colour-space":
        technique = st.sidebar.selectbox("Technique", ["Brightness", "Contrast", "Colour saturation", "Sharpness"])
        value = st.sidebar.slider("Intensity", 0.0, 2.0, 1.0, 0.05)
        enhancer = {
            "Brightness": ImageEnhance.Brightness,
            "Contrast": ImageEnhance.Contrast,
            "Colour saturation": ImageEnhance.Color,
            "Sharpness": ImageEnhance.Sharpness,
        }[technique]
        result = enhancer(original).enhance(value)

    elif category == "Kernel filter":
        technique = st.sidebar.selectbox("Filter", ["Blur", "Gaussian blur", "Sharpen", "Edge enhance"])
        result = {
            "Blur": original.filter(ImageFilter.BLUR),
            "Gaussian blur": original.filter(ImageFilter.GaussianBlur(radius=st.sidebar.slider("Radius", 0.1, 10.0, 2.0))),
            "Sharpen": original.filter(ImageFilter.SHARPEN),
            "Edge enhance": original.filter(ImageFilter.EDGE_ENHANCE),
        }[technique]

    elif category == "Random erasing":
        area = st.sidebar.slider("Erased area (%)", 1, 40, 10)
        seed = st.sidebar.number_input("Seed", 0, 9999, 42)
        result = random_erasing(original, area, seed)

    elif category == "Mixing":
        st.info("Mixing methods combine information from multiple images. Upload a second image to demonstrate a simple linear image blend.")
        second_upload = st.sidebar.file_uploader("Upload second image", type=["jpg", "jpeg", "png"])
        alpha = st.sidebar.slider("Blend weight for first image", 0.0, 1.0, 0.5, 0.05)
        if second_upload:
            second = load_image(second_upload.getvalue()).resize(original.size)
            result = Image.blend(original, second, 1 - alpha)
        else:
            result = original

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(original, use_container_width=True)
    with col2:
        st.subheader("Augmented")
        st.image(result, use_container_width=True)

    buffer = io.BytesIO()
    result.save(buffer, format="PNG")
    st.download_button(
        "⬇️ Download augmented image",
        data=buffer.getvalue(),
        file_name="augmented_image.png",
        mime="image/png",
    )

    with st.expander("About this demonstration"):
        st.write(
            "This application demonstrates practical image-space augmentation. "
            "The survey also discusses feature-space augmentation, adversarial training, "
            "GAN-based augmentation, neural style transfer and meta-learning; those "
            "methods are not claimed to be reproduced by this simple interface."
        )
else:
    st.info("Upload a JPG or PNG image to begin.")
