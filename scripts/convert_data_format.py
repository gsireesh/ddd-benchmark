from tqdm.auto import tqdm
from pathlib import Path
import fire

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TesseractOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.document import ConversionResult
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

def formatting_convert_result(result: ConversionResult, output_format: str="md") -> str:
    """
    Format the conversion result based on the specified output format.
    
    :param result: The conversion result containing the document.
    :param output_format: The desired output format (e.g., "md", "html", "txt").
    :return: The formatted document as a string.
    """
    match output_format:
        case "md":
            return result.document.export_to_markdown()
        case "html":
            return result.document.export_to_html()
        case "txt":
            return result.document.export_to_text()
        case _:
            raise ValueError(f"Unsupported output format: {output_format}")


def convert_document(filepath: str) -> ConversionResult:
    """
    Convert a document from a source to the specified output format.
    
    :param source: The local file path of the document.
    :param output_format: The desired output format (e.g., "md", "html", "json").
    :return: The converted document in the specified format.
    """

    # The following sections contain a combination of PipelineOptions
    # and PDF Backends for various configurations.
    # Uncomment one section at the time to see the differences in the output.

    # PyPdfium without EasyOCR
    # --------------------
    # pipeline_options = PdfPipelineOptions()
    # pipeline_options.do_ocr = False
    # pipeline_options.do_table_structure = True
    # pipeline_options.table_structure_options.do_cell_matching = False

    # doc_converter = DocumentConverter(
    #     format_options={
    #         InputFormat.PDF: PdfFormatOption(
    #             pipeline_options=pipeline_options, backend=PyPdfiumDocumentBackend
    #         )
    #     }
    # )

    # PyPdfium with EasyOCR
    # -----------------
    # pipeline_options = PdfPipelineOptions()
    # pipeline_options.do_ocr = True
    # pipeline_options.do_table_structure = True
    # pipeline_options.table_structure_options.do_cell_matching = True

    # doc_converter = DocumentConverter(
    #     format_options={
    #         InputFormat.PDF: PdfFormatOption(
    #             pipeline_options=pipeline_options, backend=PyPdfiumDocumentBackend
    #         )
    #     }
    # )

    # Docling Parse without EasyOCR
    # -------------------------
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    # Docling Parse with EasyOCR
    # ----------------------
    # pipeline_options = PdfPipelineOptions()
    # pipeline_options.do_ocr = True
    # pipeline_options.do_table_structure = True
    # pipeline_options.table_structure_options.do_cell_matching = True
    # pipeline_options.ocr_options.lang = ["es"]
    # pipeline_options.accelerator_options = AcceleratorOptions(
    #     num_threads=4, device=AcceleratorDevice.AUTO
    # )

    # doc_converter = DocumentConverter(
    #     format_options={
    #         InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    #     }
    # )

    return doc_converter.convert(filepath)
    
    # converter = DocumentConverter()
    # result = converter.convert(filepath)


def convert_from_directory(directory: str = "data/zeolite/pdf", output_format: str = "html") -> None:
    """
    Convert all documents in a directory to the specified output format.
    
    :param directory: The directory containing documents to convert.
    :param output_format: The desired output format (md, html, txt).
    """
    print(f"Converting documents in {directory} to {output_format} format...")
    directory_path = Path(directory)
    parent_dir = directory_path.parent
    original_format = directory_path.name
    output_dir = parent_dir / f"{original_format}_conversion" / output_format
    output_dir.mkdir(parents=True, exist_ok=True)

    for file_path in tqdm(directory_path.iterdir()):
        if not file_path.is_file(): continue
        converted_content = convert_document(str(file_path))
        output_content = formatting_convert_result(converted_content, output_format)
        output_filename = converted_content.input.file.stem
        output_filename = output_dir / f"{file_path.stem}.{output_format}"
        with output_filename.open("w", encoding="utf-8") as f:
            f.write(output_content)
        # break


if __name__ == "__main__":
    convert_from_directory(
        directory="data/zeolite/pdf", output_format="md"
    )
    # convert_from_directory(
    #     directory="data/zeolite/pdf", output_format="html"
    # )
    convert_from_directory(
        directory="data/zeolite/pdf", output_format="txt"
    )
    # python scripts/convert_data_format.py