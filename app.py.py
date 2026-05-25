import streamlit as st
from PIL import Image
import numpy as np
import os
import sys
import zipfile
import tempfile
import matplotlib.pyplot as plt
import hmac
import importlib

import Model as model_module
import Model_micro as model_micro_module
import Model_gradcam as gradcam_micro_module
import Model_gradcam_big_new as gradcam_big_module

for module in (model_module, model_micro_module, gradcam_micro_module, gradcam_big_module):
    importlib.reload(module)

predict = model_module.predict
predict_micro = model_micro_module.predict_micro
predict_gradcam_micro = gradcam_micro_module.predict_gradcam
predict_gradcam_micro_score = gradcam_micro_module.predict_gradcam_score
predict_gradcam_big = gradcam_big_module.predict_gradcam
predict_gradcam_big_score = gradcam_big_module.predict_gradcam_score

ALLOWED_SIZES = [(1280, 960), (4248, 2832)]

# Password check
def check_password():
    def password_entered():
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False
    if st.session_state.get("password_correct", False):
        return True
    st.text_input("Password", type="password", on_change=password_entered, key="password")
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False

if not check_password():
    st.stop()


# Utility functions
def display_result_box(text):
    box = f"""
    <div style="border:2px solid black; background-color:white; color:black; padding:10px; border-radius:10px;
                text-align:center; margin-bottom:10px; font-size:18px; font-weight:bold;">
        {text}
    </div>
    """
    st.markdown(box, unsafe_allow_html=True)

def plot_gradation_bar(damage):
    fig, ax = plt.subplots(figsize=(8, 1))
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    gradient = np.vstack((gradient, gradient))
    ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap('Reds'), extent=[0, 100, 1, 2])
    ax.axvline(damage, color='red', linestyle='-', linewidth=3)
    ax.text(damage, 1.5, f'{damage}%', fontsize=12, color='black', ha='center', fontweight='bold')
    ax.text(0, 2.2, 'Good', color='green', fontweight='bold')
    ax.text(100, 2.2, 'Bad', color='red', fontweight='bold', ha='right')
    ax.set_yticks([]); ax.set_xticks([])
    st.pyplot(fig)

def get_health_label(damage):
    if damage < 20:
        return "Good"
    elif damage < 60:
        return "Normal"
    elif damage < 80:
        return "Warning"
    else:
        return "Bad"

def extract_zip(zip_file):
    # temp_dir = tempfile.mkdtemp()
    # zip_path = os.path.join(temp_dir, "uploaded.zip")
    # with open(zip_path, "wb") as f:
    #     f.write(zip_file.getbuffer())
    # with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    #     zip_ref.extractall(temp_dir)
    # images = []
    # for root, _, files in os.walk(temp_dir):
    #     for file in files:
    #         if file.lower().endswith((".jpg", ".jpeg", ".png")):
    #             images.append(os.path.join(root, file))
    # return images

    # Save the uploaded ZIP file to a temporary directory
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, "uploaded.zip")
    with open(zip_path, "wb") as f:
        f.write(zip_file.getbuffer())
    
    # Extract the ZIP contents
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    
    # Scan for image files (including subdirectories)
    image_extensions = (".jpg", ".jpeg", ".png")
    image_files = []
    for root, _, files in os.walk(temp_dir):
        if "__MACOSX" in root:  # Ignore macOS metadata folder
            continue 
        for file in files:
            if file.lower().endswith(image_extensions):
                image_files.append(os.path.join(root, file))
    
    if not image_files:
        st.error("No valid image files found in the ZIP")
    return image_files
    

def run_gradcam(image_np, is_micro):
    if is_micro:
        return predict_gradcam_micro(image_np)
    return predict_gradcam_big(image_np)

def apply_patch_overlay(image, coords, probs, top, is_micro):
    img = np.array(image)
    for (x, y), p in zip(coords, probs):
        w = int(p * (15 if is_micro else 10))
        img[top + y:top + y + 224, x:x + 224, 0] = np.clip(img[top + y:top + y + 224, x:x + 224, 0] + w, 0, 255)
    return img

def insert(lst, item, max=True):
    if max:
        if lst == []:
            lst = item
        else:
            if item[0] > lst[0]:
                lst = item
    else:
        if lst == []:
            lst = item
        else:
            if item[0] < lst[0]:
                lst = item
    return lst


def resize_with_aspect_ratio(image, target_size=(1280, 960)):
    # """Resize image to target_size keeping aspect ratio by padding."""
    if image.size == target_size:
        return image

    target_w, target_h = target_size
    src_w, src_h = image.size
    scale = min(target_w / src_w, target_h / src_h)
    new_size = (
        max(1, int(round(src_w * scale))),
        max(1, int(round(src_h * scale)))
    )

    resample_filter = getattr(Image, "Resampling", Image).LANCZOS
    resized = image.resize(new_size, resample_filter)
    canvas = Image.new(image.mode, target_size)
    offset = ((target_w - new_size[0]) // 2, (target_h - new_size[1]) // 2)
    canvas.paste(resized, offset)
    return canvas

def ensure_supported_size(image):
    # """
    # 업로드 이미지가 허용 크기가 아니면 1280x960으로 패딩 리사이즈해 추론이 가능하도록 만든다.
    # """
    if image.size not in ALLOWED_SIZES:
        return resize_with_aspect_ratio(image, (1280, 960))
    return image

def analyze_images(folder1_images, folder2_images):
    max1 = []
    min1 = []
    max2 = []
    min2 = []

    for img_path in folder1_images:
        img = Image.open(img_path).convert("RGB")
        # ZIP 이미지도 단일 업로드와 동일하게 크기를 표준화한다.
        img = ensure_supported_size(img)
        if img.size == (1280, 960):
            result, coorinates_damage, top, bottom, probs = predict(np.array(img))
        else:
            result, coorinates_damage, top, bottom, probs = predict_micro(np.array(img))
        max1 = insert(max1, [result, coorinates_damage, top, bottom, probs, img_path], max=True)
        min1 = insert(min1, [result, coorinates_damage, top, bottom, probs, img_path], max=False)
        
    for img_path in folder2_images:
        img = Image.open(img_path).convert("RGB")
        # ZIP 이미지도 단일 업로드와 동일하게 크기를 표준화한다.
        img = ensure_supported_size(img)
        if img.size == (1280, 960):
            result, coorinates_damage, top, bottom, probs = predict(np.array(img))
        else:
            result, coorinates_damage, top, bottom, probs = predict_micro(np.array(img))
        max2 = insert(max2, [result, coorinates_damage, top, bottom, probs, img_path], max=True)
        min2 = insert(min2, [result, coorinates_damage, top, bottom, probs, img_path], max=False)

    if max1[0] - min2[0] > max2[0] - min1[0]:
        return max1, min2
    else:
        return min1, max2

def analyze_images_gradcam(folder1, folder2):
    max1 = [0, 0]
    min1 = [1, 0]
    max2 = [0, 0]
    min2 = [1, 0]

    for img_path in folder1:
        img = Image.open(img_path).convert("RGB")
        # Grad-CAM 비교에서도 이미지 크기를 정규화해 모델 선택을 일관되게 유지한다.
        img = ensure_supported_size(img)
        if img.size == (1280, 960):
            result = predict_gradcam_big_score(np.array(img))
        else:
            result = predict_gradcam_micro_score(np.array(img))
        max1 = [max(max1[0], result), img_path]
        min1 = [min(min1[0], result), img_path]

    for img_path in folder2:
        img = Image.open(img_path).convert("RGB")
        # Grad-CAM 비교에서도 이미지 크기를 정규화해 모델 선택을 일관되게 유지한다.
        img = ensure_supported_size(img)
        if img.size == (1280, 960):
            result = predict_gradcam_big_score(np.array(img))
        else:
            result = predict_gradcam_micro_score(np.array(img))
        max2 = [max(max2[0], result), img_path]
        min2 = [min(min2[0], result), img_path]

    if max1[0] - min2[0] > max2[0] - min1[0]:
        x1 = max1[1]
        x2 = min2[1]
    else:
        x1 = min1[1]
        x2 = max2[1]
    
    x1_img = Image.open(x1).convert("RGB")
    # """결과 프리뷰도 추론 당시와 동일한 크기를 사용한다."""
    x1_img = ensure_supported_size(x1_img)
    if x1_img.size == (1280, 960):
        result1, pred, visualization1 = predict_gradcam_big(np.array(x1_img))
    else:
        result1, pred, visualization1 = predict_gradcam_micro(np.array(x1_img))
    x2_img = Image.open(x2).convert("RGB")
    # """결과 프리뷰도 추론 당시와 동일한 크기를 사용한다."""
    x2_img = ensure_supported_size(x2_img)
    if x2_img.size == (1280, 960):
        result2, pred, visualization2 = predict_gradcam_big(np.array(x2_img))
    else:
        result2, pred, visualization2 = predict_gradcam_micro(np.array(x2_img))

    return [result1, visualization1, x1], [result2, visualization2, x2]


# UI header
st.markdown("<h1 style='text-align:center; background-color:black; color:white; padding:10px; border-radius:10px;'>Hair Damage Determination</h1>", unsafe_allow_html=True)
mode = st.radio("Select Mode", ["Single Image Diagnosis", "Folder Comparison"])


# Single Image Diagnosis
if mode == "Single Image Diagnosis":
    file = st.file_uploader("Upload your Hair Image", type=['jpg', 'png'])
    if file:
        # """단일 업로드도 공용 리사이즈 규칙을 따른다."""
        uploaded_image = Image.open(file).convert("RGB")
        image = ensure_supported_size(uploaded_image)

        st.image(image, caption=f"{file.name.split('/')[-1]}", use_container_width=True)
        if image.size not in ALLOWED_SIZES:
            display_result_box("헤어 이미지 아닙니다.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                patch_btn = st.button("Start Diagnosis [Patch]")
            with col2:
                grad_btn = st.button("Start Diagnosis [Grad-CAM]")

            is_micro = image.size != (1280, 960)

            if patch_btn:
                if is_micro:
                    result, coords, top, _, probs = predict_micro(np.array(image))
                else:
                    result, coords, top, _, probs = predict(np.array(image))

                img = np.array(image)
                if coords:
                    for coord, weight in zip(coords, probs):
                        x, y = coord
                        w = int(weight * (15 if is_micro else 10))
                        img[top + y:top + y + 224, x: x+224, 0] = np.where(
                            img[top + y:top + y + 224, x: x+224, 0] > 255 - w,
                            255 - w,
                            img[top + y:top + y + 224, x: x+224, 0]
                        )
                        img[top + y:top + y + 224, x: x+224, 0] += w

                st.image(img, caption="Damage Highlighted [Patch]", use_container_width=True)
                display_result_box(f"Damage Level: {result}%")
                display_result_box(f"Hair Health: {get_health_label(result)}")
                plot_gradation_bar(result)

            if grad_btn:
                img_np = np.array(image)
                if is_micro:
                    result, _, heatmap = predict_gradcam_micro(img_np)
                else:
                    result, _, heatmap = predict_gradcam_big(img_np)

                st.image(heatmap, caption="Grad-CAM Result", use_container_width=True)
                display_result_box(f"Damage Level: {result}%")
                display_result_box(f"Hair Health: {get_health_label(result)}")
                plot_gradation_bar(result)


# Folder Comparison
if mode == "Folder Comparison":
    # folder1 = st.file_uploader("Upload Folder 1", type=["zip"], key="folder1")
    # folder2 = st.file_uploader("Upload Folder 2", type=["zip"], key="folder2")

    # if folder1 and folder2:
    #     st.subheader("Processing folders...")
    #     imgs1 = extract_zip(folder1)
    #     imgs2 = extract_zip(folder2)

    #     def analyze_and_cache(img_paths, folder_label):
    #         results = []
    #         for path in img_paths:
    #             with Image.open(path) as img:
    #                 img = img.convert("RGB")
    #                 is_micro = img.size != (1280, 960)
    #                 name = os.path.basename(path)
    #                 img_np = np.array(img)
    #                 if is_micro:
    #                     dmg, coords, top, _, probs = predict_micro(img_np)
    #                 else:
    #                     dmg, coords, top, _, probs = predict(img_np)
    #                 results.append({
    #                     "name": name,
    #                     "damage": dmg,
    #                     "image": img_np,
    #                     "path": path,
    #                     "folder": folder_label,
    #                     "is_micro": is_micro,
    #                     "coords": coords,
    #                     "probs": probs,
    #                     "top": top
    #                 })
    #         if len(results) < 2:
    #             return results
    #         sorted_results = sorted(results, key=lambda x: x["damage"])
    #         return [sorted_results[0], sorted_results[-1]]

    #     preview1 = analyze_and_cache(imgs1, folder1.name)
    #     preview2 = analyze_and_cache(imgs2, folder2.name)

    #     col1, col2 = st.columns(2)
    #     with col1:
    #         st.subheader(f"Folder 1 - Preview ({folder1.name})")
    #         for item in preview1:
    #             st.image(item["image"], caption=f'{item["name"]} - {item["damage"]}%', use_container_width=True)
    #     with col2:
    #         st.subheader(f"Folder 2 - Preview ({folder2.name})")
    #         for item in preview2:
    #             st.image(item["image"], caption=f'{item["name"]} - {item["damage"]}%', use_container_width=True)

    #     patch_btn = st.button("Start Diagnosis [Patch]", key="patch_folder")
    #     grad_btn = st.button("Start Diagnosis [Grad-CAM]", key="grad_folder")

    #     if patch_btn or grad_btn:
    #         method = "patch" if patch_btn else "grad"

    #         def run_visualization(preview_data):
    #             results = []
    #             for item in preview_data:
    #                 if method == "patch":
    #                     viz = apply_patch_overlay(item["image"], item["coords"], item["probs"], item["top"], item["is_micro"])
    #                     dmg = item["damage"]
    #                 else:
    #                     dmg, _, viz = run_gradcam(item["image"], item["is_micro"])
    #                 results.append((item["name"], dmg, viz, item["folder"]))
    #             return results

    #         results1 = run_visualization(preview1)
    #         results2 = run_visualization(preview2)

    #         col1, col2 = st.columns(2)
    #         with col1:
    #             st.subheader(f"Folder 1 Results - {folder1.name}")
    #             for name, dmg, img, src in results1:
    #                 st.image(img, caption=f"{name} [{method.upper()}]\nFrom Folder: {src}", use_container_width=True)
    #                 display_result_box(f"Damage Level: {dmg}%")
    #                 display_result_box(f"Hair Health: {get_health_label(dmg)}")
    #                 plot_gradation_bar(dmg)

    #         with col2:
    #             st.subheader(f"Folder 2 Results - {folder2.name}")
    #             for name, dmg, img, src in results2:
    #                 st.image(img, caption=f"{name} [{method.upper()}]\nFrom Folder: {src}", use_container_width=True)
    #                 display_result_box(f"Damage Level: {dmg}%")
    #                 display_result_box(f"Hair Health: {get_health_label(dmg)}")
    #                 plot_gradation_bar(dmg)
    folder1_zip = st.file_uploader("Upload Folder1 (ZIP)", type=['zip'])
    folder2_zip = st.file_uploader("Upload Folder2 (ZIP)", type=['zip'])


    if folder1_zip and folder2_zip:
        st.subheader("Processing Images...")

        # Extract and filter images from ZIP
        folder1_images = extract_zip(folder1_zip)
        folder2_images = extract_zip(folder2_zip)

        if folder1_images and folder2_images:
            if st.button("Diagnosis [Patch]"):
                x1, x2 = analyze_images(folder1_images, folder2_images)
                st.subheader("Most Significant Difference")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("<h3 style='text-align: center; background-color: lightgray; padding: 5px;'>Folder 1 Results</h3>", unsafe_allow_html=True)
                    result, coordinates_damage, top, bottom, probs, img_path = x1
                    base_img = Image.open(img_path).convert("RGB")
                    # """패치 시각화도 표준화된 이미지에 좌표를 적용한다."""
                    base_img = ensure_supported_size(base_img)
                    img = np.array(base_img)

                    for coord, weight in zip(coordinates_damage, probs):
                        x, y = coord
                        w = int(weight * 10)
                        # for red dimension, if pixel more than 245, set to 245
                        img[top + y:top + y + 224, x: x+224, 0] = np.where(img[top + y:top + y + 224, x: x+224, 0] > 255-w, 255-w, img[top + y:top + y + 224, x: x+224, 0])
                        img[top + y:top + y + 224, x: x+224, 0] += w
                    
                    st.image(img, caption=f"{img_path.split('/')[-1]} - {result}%", use_container_width=True)
                    display_result_box(f"Damage Level: {result}%")
                    display_result_box(f"Hair Health: {get_health_label(result)}")
                    plot_gradation_bar(result)
                
                with col2:
                    st.markdown("<h3 style='text-align: center; background-color: lightgray; padding: 5px;'>Folder 2 Results</h3>", unsafe_allow_html=True)
                    result, coordinates_damage, top, bottom, probs, img_path = x2
                    base_img = Image.open(img_path).convert("RGB")
                    # """패치 시각화도 표준화된 이미지에 좌표를 적용한다."""
                    base_img = ensure_supported_size(base_img)
                    img = np.array(base_img)

                    for coord, weight in zip(coordinates_damage, probs):
                        x, y = coord
                        w = int(weight * 10)
                        # for red dimension, if pixel more than 245, set to 245
                        img[top + y:top + y + 224, x: x+224, 0] = np.where(img[top + y:top + y + 224, x: x+224, 0] > 255-w, 255-w, img[top + y:top + y + 224, x: x+224, 0])
                        img[top + y:top + y + 224, x: x+224, 0] += w

                    st.image(img, caption=f"{img_path.split('/')[-1]} - {result}%", use_container_width=True)
                    display_result_box(f"Damage Level: {result}%")
                    display_result_box(f"Hair Health: {get_health_label(result)}")
                    plot_gradation_bar(result)

            if st.button("Diagnosis [Grad-CAM]"):
                x1, x2 = analyze_images_gradcam(folder1_images, folder2_images)

                st.subheader("Most Significant Difference")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("<h3 style='text-align: center; background-color: lightgray; padding: 5px;'>Folder 1 Results</h3>", unsafe_allow_html=True)
                    result, visualization, img_path = x1
                    
                    st.image(visualization, caption=f"{img_path.split('/')[-1]} - {result}%", use_container_width=True)
                    display_result_box(f"Damage Level: {result}%")
                    display_result_box(f"Hair Health: {get_health_label(result)}")
                    plot_gradation_bar(result)

                with col2:
                    st.markdown("<h3 style='text-align: center; background-color: lightgray; padding: 5px;'>Folder 2 Results</h3>", unsafe_allow_html=True)
                    result, visualization, img_path = x2
                    
                    st.image(visualization, caption=f"{img_path.split('/')[-1]} - {result}%", use_container_width=True)
                    display_result_box(f"Damage Level: {result}%")
                    display_result_box(f"Hair Health: {get_health_label(result)}")
                    plot_gradation_bar(result)
