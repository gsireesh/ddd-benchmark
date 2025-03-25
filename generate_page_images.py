import logging
import os

import fire
import pandas as pd
import pdfplumber
from pypdf import PdfReader, PdfWriter
from tqdm.auto import tqdm

from data.dataset_metadata import metadata_by_dataset


def generate_page_images(dataset: str) -> None:

    dataset_directory = metadata_by_dataset[dataset]["data_directory"]

    if not os.path.exists(os.path.join(dataset_directory, "page_images")):
        os.mkdir(os.path.join(dataset_directory, "page_images"))

    for pdf_file in tqdm(
        os.listdir(os.path.join(dataset_directory, "pdf")), "Generating page " "images."
    ):
        try:
            pdf_path = f"{dataset_directory}/pdf/{pdf_file}"
            if not pdf_path.endswith(".pdf"):
                continue
            image_path_template = (
                f"{dataset_directory}/page_images/{pdf_file.replace('.pdf','')}_{{page_number}}.png"
            )
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    image_path = image_path_template.format(page_number=i)
                    if not os.path.exists(image_path):
                        page = pdf.pages[i]
                        page_image = page.to_image(resolution=300)
                        page_image.save(image_path)
        except Exception as e:
            logging.error(f"Failed to process {pdf_file}. Error: {e}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s:[%(filename)-25s:%(lineno)-4d] %(message)s",
    )
    fire.Fire(generate_page_images)
