import cv2
import pytesseract
from PIL import Image
import gradio as gr
import numpy as np
import pandas as pd
from pdf2image import convert_from_path
import os

# If Tesseract is not in PATH, uncomment and set the path
# pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"


def ocr_detect_image(img, lang='eng'):
    """
    Detect text from a single image.
    Returns detected text and confidence table.
    """
    if img is None:
        return "No image provided!", None

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Denoise
    gray = cv2.medianBlur(gray, 3)

    # Resize for better OCR
    scale_factor = 2
    gray_resized = cv2.resize(
        gray, None,
        fx=scale_factor,
        fy=scale_factor,
        interpolation=cv2.INTER_CUBIC
    )

    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray_resized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    # OCR using pytesseract
    data = pytesseract.image_to_data(
        thresh,
        lang=lang,
        output_type=pytesseract.Output.DICT
    )

    words, confs = [], []

    for i, word in enumerate(data['text']):
        if word.strip() != "":
            words.append(word)
            confs.append(int(data['conf'][i]))

    detected_text = " ".join(words) if words else "No text detected in the image."
    conf_table = (
        pd.DataFrame({"Word": words, "Confidence": confs})
        if words else None
    )

    return detected_text, conf_table


def ocr_detect_file(file, lang='eng', poppler_path=None):
    """
    Detect text from uploaded images or multi-page PDFs.
    Returns detected text and confidence table.
    """
    if file is None:
        return "No file provided!", None

    detected_text = ""
    all_words = []
    all_confs = []

    # Check if PDF
    if file.name.lower().endswith('.pdf'):
        pages = convert_from_path(
            file.name,
            dpi=300,
            poppler_path=poppler_path
        )

        for page in pages:
            img = np.array(page)
            page_text, conf_table = ocr_detect_image(img, lang)

            detected_text += page_text + "\n\n"

            if conf_table is not None:
                all_words.extend(conf_table["Word"].tolist())
                all_confs.extend(conf_table["Confidence"].tolist())

        conf_table_all = (
            pd.DataFrame({"Word": all_words, "Confidence": all_confs})
            if all_words else None
        )

        return detected_text.strip(), conf_table_all

    else:
        # Assume image file
        img = Image.open(file.name).convert("RGB")
        return ocr_detect_image(np.array(img), lang)


# Gradio Interface
iface = gr.Interface(
    fn=ocr_detect_file,
    inputs=[
        gr.File(
            file_types=['.png', '.jpg', '.jpeg', '.pdf'],
            label="Upload Image or PDF"
        ),
        gr.Dropdown(
            ['eng', 'tel', 'hin'],
            label="Language",
            value='eng'
        )
    ],
    outputs=[
        gr.Textbox(label="Detected Text"),
        gr.Dataframe(label="Confidence Scores")
    ],
    title="OCR Detection Chatbot (Text + Confidence Only)",
    description=(
        "Upload an image or multi-page PDF to detect text using OCR. "
        "Supports English, Telugu, and Hindi. "
        "Confidence scores are provided for each detected word."
    ),
    flagging_mode="never",
    live=False
)

if __name__ == "__main__":
    iface.launch(share=True)
