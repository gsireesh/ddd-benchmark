import os

import fire
import pandas as pd
import pdfplumber
from pypdf import PdfReader, PdfWriter
from tqdm.auto import tqdm


def generate_page_images(
    meta_df: pd.DataFrame,
    input_directory: str,
    output_pdf_directory: str,
    output_image_directory: str,
) -> None:

    if not all([thing in meta_df.columns for thing in ["From?", "doi", "Page in paper"]]):
        raise AssertionError(
            "Not all required columns found! Are you using the correct data df?"
            'Required columns are "From?", "doi", and "Page in paper"'
        )

    for index, doi, page in tqdm(
        meta_df[meta_df["From?"] == "Table"][["doi", "Page in paper"]].itertuples()
    ):
        page_number = int(page) - 1

        pdf_path = f"{input_directory}/{doi.replace('/', '_')}.pdf"
        image_path = f"{output_image_directory}/{doi.replace('/', '_')}.png"
        page_pdf_path = f"{output_image_directory}/{doi.replace('/', '_')}.pdf"

        if not os.path.exists(pdf_path):
            raise AssertionError(f"PDF with doi {doi} not found! Exiting.")

        if not os.path.exists(image_path):
            with pdfplumber.open(pdf_path) as pdf:
                page = pdf.pages[page_number]
                page_image = page.to_image(resolution=300)
                page_image.save(image_path)

        if not os.path.exists(page_pdf_path):
            reader = PdfReader(pdf_path)
            writer = PdfWriter()
            writer.add_page(reader.pages[page_number])
            writer.write(page_pdf_path)


def generate_page_images_wrapper(
    meta_df_path: str = "data/zeolite_data_location_annotated.csv",
    input_directory: str = "data/pdf",
    output_pdf_directory: str = "data/zeolite_page_pdfs",
    output_image_directory: str = "data/page_images",
) -> None:
    """Generate page images and standalone page PDFs from the original PDF documents.

    This script will not overwrite images/PDFs if they already exist.

    :param meta_df_path: The path to the annotations of the dataset.
    :param input_directory: A directory of PDFs, whose filenames are expected to be the {doi}.pdf,
    "/" replaced by "_". If a listed paper in th meta_df is not present, the script will fail.
    :param output_pdf_directory: The directory to which to write single-page PDFs.
    :param output_image_directory: The directory to which to write single page PNG images.
    """
    meta_df = pd.read_csv(meta_df_path)
    generate_page_images(
        input_directory=input_directory,
        output_pdf_directory=output_pdf_directory,
        output_image_directory=output_image_directory,
    )


if __name__ == "__main__":
    fire.Fire(generate_page_images_wrapper)
