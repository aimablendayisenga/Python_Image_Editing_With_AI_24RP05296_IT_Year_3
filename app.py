"""Smart AI Photo Editor. Run: streamlit run app.py"""
import base64
import hashlib
import io, json, os, time
import pandas as pd
import streamlit as st
from PIL import Image, ImageFilter, ImageOps
from classifier import CLASSES, load_classifier, predict
from editor import FIXES, FIX_NAMES, LOOK_PRESETS, apply_look, auto_fix
from enhancer import MODEL_DIR, degrade, enhance, load_model

st.set_page_config(
    page_title="Photo Studio Image AI",
    page_icon=":material/auto_awesome:",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
def get_clf():
    return load_classifier()


@st.cache_resource
def get_sr():
    return load_model()


@st.cache_data(show_spinner=False)
def get_demo_pair():
    before_path = "assets/AIMABLE BEFORE.png"
    after_path = "assets/AIMABLE AFTER.jpg"
    if os.path.exists(before_path) and os.path.exists(after_path):
        before = Image.open(before_path).convert("RGB")
        after = Image.open(after_path).convert("RGB")
        after = ImageOps.fit(
            after,
            before.size,
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
        return before, after, True

    image = Image.open("data/train/100075.jpg").convert("RGB")
    width, height = image.size
    image = image.crop((0, 0, width - width % 2, height - height % 2))
    before = degrade(image)
    return before, enhance(get_sr(), before), False


@st.cache_data(show_spinner=False)
def get_studio_background():
    background_path = os.path.join(
        os.path.dirname(__file__), "assets", "AIMABLE AFTER.jpg"
    )
    with Image.open(background_path) as source:
        background = source.convert("RGB")
    background.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    background = background.filter(ImageFilter.GaussianBlur(radius=8))
    output = io.BytesIO()
    background.save(output, format="JPEG", quality=76, optimize=True)
    return base64.b64encode(output.getvalue()).decode("ascii")


st.html(
    f"""<style>
    .stApp {{
        background:
            linear-gradient(115deg, rgba(10, 12, 18, 0.88), rgba(10, 12, 18, 0.72)),
            url("data:image/jpeg;base64,{get_studio_background()}") center / cover fixed;
    }}
    </style>"""
)


if not os.path.exists(f"{MODEL_DIR}/classifier.pt"):
    st.error("No trained classifier. Run `python classifier.py train --data data/train` first.")
    st.stop()
has_sr = os.path.exists(f"{MODEL_DIR}/srnet_x2.pt")

# ---- sidebar: evaluation results
with st.sidebar:
    st.markdown("### Photo Studio Image AI")
    st.caption("AI photo studio")
    with st.expander("Model evaluation"):
        metrics_path = f"{MODEL_DIR}/clf_metrics.json"
        if os.path.exists(metrics_path):
            with open(metrics_path, encoding="utf-8") as metrics_file:
                metrics = json.load(metrics_file)
            st.metric("Classifier accuracy", f"{metrics['accuracy']:.0%}")
            st.metric("Macro F1", f"{metrics['f1']:.3f}")
            st.dataframe(
                pd.DataFrame(metrics["confusion_matrix"], index=CLASSES, columns=CLASSES),
                width="stretch",
            )

st.title("Photo Studio Image AI", icon=":material/photo_camera:")
st.caption("Color, light, and the moments worth keeping.")

preview_column, editor_column = st.columns([1.4, 1], gap="medium")
image_file = None
image = None
source_image = None
edited_image = None
image_label = None
image_probs = None
image_key = None

with editor_column:
    with st.container(border=True):
        st.subheader("Photo editor", icon=":material/tune:")
        image_file = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])

        if image_file is not None:
            image_bytes = image_file.getvalue()
            image_key = hashlib.sha256(image_bytes).hexdigest()
            source_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image = source_image.copy()
            if max(image.size) > 800:
                image.thumbnail((800, 800), Image.Resampling.LANCZOS)
                st.caption(
                    "Large photo resized to 800 px for editing; recognition uses the original."
                )

            if st.session_state.get("image_key") != image_key:
                st.session_state.image_key = image_key
                st.session_state.pop("edited_image", None)
                st.session_state.pop("edited_key", None)
                st.session_state.pop("edited_caption", None)
                with st.spinner("Analyzing photo..."):
                    classifier = get_clf()
                    st.session_state.image_label, st.session_state.image_probs = predict(
                        classifier, source_image
                    )
                    if st.session_state.image_label == "noisy":
                        with st.spinner("Noise detected. Creating a denoised preview..."):
                            edited_image = auto_fix(image, "noisy")
                        st.session_state.edited_image = edited_image
                        st.session_state.edited_key = image_key
                        st.session_state.edited_caption = "Auto · Noise reduced"
                        st.session_state.setdefault("log", []).append(
                            f"{image_file.name}: noisy -> automatic noise removal -> {after_label}"
                        )

            image_label = st.session_state.image_label
            image_probs = st.session_state.image_probs
            st.badge(
                f"Detected: {image_label} · {image_probs[image_label]:.0%}",
                icon=":material/center_focus_strong:",
            )
            if image_label == "noisy":
                st.caption(
                    "Noise detected: a denoised preview is ready. Your original stays unchanged."
                )

            with st.form("edit_controls"):
                choice = st.selectbox(
                    "Correction / style preset",
                    ["Auto (recommended)"]
                    + [FIX_NAMES[key] for key in FIXES]
                    + list(LOOK_PRESETS),
                )
                upscale = st.selectbox(
                    "Output size",
                    ["Original", "HD (2x)", "Ultra HD (4x)"],
                    index=1 if has_sr else 0,
                )
                apply_edits = st.form_submit_button(
                    "Apply edits",
                    type="primary",
                    icon=":material/auto_fix_high:",
                    width="stretch",
                )

            if upscale != "Original" and not has_sr:
                st.caption("Train the upscaler with `python enhancer.py train` to enable HD output.")

            if apply_edits:
                if choice.startswith("Auto"):
                    preset_name = f"Auto · {FIX_NAMES[image_label]}"
                elif choice in FIX_NAMES.values():
                    fix_key = next(key for key, name in FIX_NAMES.items() if name == choice)
                    preset_name = choice
                else:
                    preset_name = choice

                with st.spinner("Applying edits..."):
                    if choice.startswith("Auto"):
                        edited_image = auto_fix(image, image_label)
                    elif choice in FIX_NAMES.values():
                        edited_image = auto_fix(image, fix_key)
                    else:
                        edited_image = apply_look(image, choice)
                    if upscale != "Original" and has_sr:
                        edited_image = enhance(
                            get_sr(), edited_image, 1 if upscale == "HD (2x)" else 2
                        )
                    after_label, _ = predict(get_clf(), edited_image)
                st.session_state.edited_image = edited_image
                st.session_state.edited_key = image_key
                st.session_state.edited_caption = (
                    f"{preset_name} · {upscale} · detected {after_label}"
                )
                st.session_state.setdefault("log", []).append(
                    f"{image_file.name}: {image_label} -> {preset_name} + {upscale} -> {after_label}"
                )

            if st.session_state.get("edited_key") == image_key:
                edited_image = st.session_state.edited_image
                output = io.BytesIO()
                edited_image.save(output, "PNG")
                st.download_button(
                    "Download edited photo",
                    output.getvalue(),
                    "edited.png",
                    "image/png",
                    icon=":material/download:",
                    width="stretch",
                )
        else:
            for key in (
                "image_key",
                "image_label",
                "image_probs",
                "edited_image",
                "edited_key",
                "edited_caption",
            ):
                st.session_state.pop(key, None)
            st.caption("JPG or PNG · your photo stays on this device.")

with preview_column:
    with st.container(border=True):
        st.caption("BEFORE / AFTER")
        if image is None:
            before_image, after_image, is_real_pair = get_demo_pair()
            if is_real_pair:
                before_caption = "Before · original color"
                after_caption = "After · black & white"
                preview_note = ""
            else:
                before_caption = f"Before · {before_image.width} × {before_image.height}"
                after_caption = f"HD preview · {after_image.width} × {after_image.height}"
                preview_note = "BSDS500 example · trained 2× reconstruction"
        else:
            before_image = source_image
            has_edited_image = st.session_state.get("edited_key") == image_key
            after_image = st.session_state.edited_image if has_edited_image else None
            before_caption = (
                f"Before · {source_image.width} × {source_image.height} · {image_label}"
            )
            after_caption = st.session_state.get("edited_caption", "After · apply edits to preview")
            preview_note = "Your uploaded photo remains unchanged in the before panel."

        before_column, after_column = st.columns(2, gap="small", wrap=False)
        with before_column:
            st.image(
                before_image,
                caption=before_caption,
                width="stretch",
                alt="Photo before editing",
            )
        with after_column:
            if after_image is None:
                st.container(height=300, border=True).markdown("**Your edited preview will appear here.**")
            else:
                st.image(
                    after_image,
                    caption=after_caption,
                    width="stretch",
                    alt="Photo after editing",
                )
        if preview_note:
            st.caption(preview_note)

if image_probs is not None:
    with st.expander("Recognition details"):
        st.bar_chart(pd.Series(image_probs, name="probability"))

if st.session_state.get("log"):
    with st.expander("Recent edits", icon=":material/history:"):
        st.write("\n".join(st.session_state["log"]))

st.caption("AI edits can invent detail. Compare with the original, especially faces and text.")
