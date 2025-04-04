import ast
import json
import logging
import os
import re
import time
from typing import Callable

from bs4 import BeautifulSoup
import fire
import pandas as pd
import requests
from requests_ratelimiter import LimiterSession
from tqdm import tqdm

METADATA_KEYS = ["doi", "publisher", "journal", "included_in_dataset", "xml", "pdf"]    # not used in code
API_TIMEOUT = 10

INCLUDED_PUBLISHERS = [
    "Elsevier BV",
    "Springer Science and Business Media LLC",
    "Wiley"
]

INCLUDED_PUBLISHERS_SHORT_NAMES = [
    "elsevier",  # for Elsevier BV
    "springer",  # for Springer Science and Business Media LLC
    "wiley",     # for Wiley
]

DATASET_TO_PATH = {
    "zeolite": "data/zeolite",
    "aluminum": "data/aluminum",
}

def download_elsevier_papers(
    doi_list: list[str], api_key: str, out_folder: str, timeout: float = API_TIMEOUT
) -> dict[str, str]:
    
    session = LimiterSession(per_second=10)
    doi_format = {}
    os.makedirs(os.path.join(out_folder, "xml"), exist_ok=True)

    for doi in tqdm(doi_list, "Downloading Elsevier Papers."):
        file_path = os.path.join(out_folder, "xml", clean_doi(doi) + ".xml")
        if os.path.exists(file_path):
            logging.info(f"Elsevier: DOI {doi} file already exists, skipping.")
            doi_format[doi] = "xml"
            continue
        try:
            response = session.get(
                f"https://api.elsevier.com/content/article/doi/{doi}",
                headers={"Accept": "application/xml", "X-ELS-APIKey": api_key},
                timeout=timeout,
            )

        except requests.exceptions.Timeout:
            logging.error(f"Elsevier: DOI {doi} timed out.")
            doi_format[doi] = "failed"
            continue

        if response.status_code == 200:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(response.text)
            logging.info(f"Elsevier: DOI {doi} saved to file.")
            doi_format[doi] = "xml"
        else:
            logging.error(f"Elsevier: DOI {doi} failed with status code {response.status_code}.")
            doi_format[doi] = "failed"

    succeeded_count = sum(1 for status in doi_format.values() if status != "failed")
    failed_count = len(doi_list) - succeeded_count
    logging.info(f"Elsevier: {succeeded_count} succeeded, {failed_count} failed.")
    return doi_format

def download_springer_papers(
    doi_list: list[str], api_key: str, out_folder: str, timeout: float = API_TIMEOUT
) -> dict[str, str]:
    
    session = LimiterSession(per_minute=100, per_day=500)
    doi_format = {}
    os.makedirs(os.path.join(out_folder, "xml"), exist_ok=True)
    for doi in tqdm(doi_list, "Downloading Springer Papers."):
        file_path = os.path.join(out_folder, "xml", clean_doi(doi) + ".xml")
        if os.path.exists(file_path):
            logging.info(f"Springer: DOI {doi} file already exists, skipping.")
            doi_format[doi] = "xml"
            continue
        try:
            response = session.get(
                f"https://api.springer.com/openaccess/jats?q=doi:{doi}&api_key={api_key}",
                timeout=timeout,
            )
        except requests.exceptions.Timeout:
            logging.error(f"Springer: DOI {doi} timed out.")
            doi_format[doi] = "failed"
            continue
        if response.status_code == 200:
            logging.info(f"Springer: DOI {doi} saved to file.")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(response.text)
            doi_format[doi] = "xml"
        else:
            logging.error(f"Springer: DOI {doi} failed with status code {response.status_code}.")
            doi_format[doi] = "failed"
    
    succeeded_count = sum(1 for status in doi_format.values() if status != "failed")
    failed_count = len(doi_list) - succeeded_count
    logging.info(f"Springer: {succeeded_count} succeeded, {failed_count} failed.")
    return doi_format

def download_wiley_papers(
    doi_list: list[str], api_key: str, out_folder: str, timeout=API_TIMEOUT
) -> dict[str, str]:
    session = LimiterSession(per_second=3, per_minute=6)
    doi_format = {}
    os.makedirs(os.path.join(out_folder, "pdf"), exist_ok=True)
    for doi in tqdm(doi_list, "Downloading Wiley Papers."):
        file_path = os.path.join(out_folder, "pdf", clean_doi(doi) + ".pdf")
        if os.path.exists(file_path):
            logging.info(f"Wiley: DOI {doi} file already exists, skipping.")
            doi_format[doi] = "pdf"
            continue
        try:
            response = session.get(
                f"https://api.wiley.com/onlinelibrary/tdm/v1/articles/{doi}",
                headers={"Wiley-TDM-Client-Token": api_key},
                timeout=timeout,
            )
        except requests.exceptions.Timeout:
            logging.error(f"Wiley: DOI {doi} timed out.")
            doi_format[doi] = "failed"
            continue
        if response.status_code == 200:
            logging.info(f"Wiley: DOI {doi} saved to file.")
            with open(file_path, "wb") as f:
                f.write(response.content)
            doi_format[doi] = "pdf"
        else:
            logging.error(f"Wiley: DOI {doi} failed with status code {response.status_code}.")
            doi_format[doi] = "failed"

    succeeded_count = sum(1 for status in doi_format.values() if status != "failed")
    failed_count = len(doi_list) - succeeded_count
    logging.info(f"Wiley: {succeeded_count} succeeded, {failed_count} failed.")
    return doi_format

def clean_doi(doi: str):
    """Clean a DOI so it can be used as a filename"""
    return re.sub(r'[<>:"/\\|?*]', "_", doi)

def get_zeolite_dois(dataset_directory) -> list[str]:
    """
    Extract DOIs of papers in the Zeolite dataset from the provided xlsx.

    Returns:
        List[str]: A list of DOIs of papers in the Zeolite dataset.
    """
    ZEOLITE_DATASET_PATH = os.path.join(dataset_directory, "ZEOSYN.xlsx")
    assert os.path.exists(ZEOLITE_DATASET_PATH), f"Zeolite dataset excel file not found at {ZEOLITE_DATASET_PATH}"
    zeolite_dataset_df = pd.read_excel(ZEOLITE_DATASET_PATH)
    zeolite_dataset_df = zeolite_dataset_df.dropna(axis=0, thresh=20)
    zeolite_dataset_DOIs = zeolite_dataset_df["doi"].dropna().unique().tolist()
    return zeolite_dataset_DOIs

def get_aluminum_dois(dataset_directory) -> list[str]:
    """
    Extract DOIs of papers in the Al-alloy dataset from the provided CSV file.
    
    Returns:
        List[str]: A list of DOIs of papers in the Al-alloy dataset.
    """
    AL_ALLOY_COMPOSITION_PATH = os.path.join(dataset_directory, "composition.csv")
    AL_ALLOY_PROPERTIES_PATH = os.path.join(dataset_directory, "property.csv")
    assert os.path.exists(AL_ALLOY_COMPOSITION_PATH), f"Al-alloy composition CSV file not found at {AL_ALLOY_COMPOSITION_PATH}"
    assert os.path.exists(AL_ALLOY_PROPERTIES_PATH), f"Al-alloy properties CSV file not found at {AL_ALLOY_PROPERTIES_PATH}"
    Al_alloy_composition_df = pd.read_csv(AL_ALLOY_COMPOSITION_PATH, dtype={"ft_doi_list": str})
    Al_alloy_composition_dois = (
        Al_alloy_composition_df["ft_doi_list"]
        .dropna()                  # drop NaN values
        .apply(ast.literal_eval)   # convert the string of list to an actual list
        .explode()                 # flatten the list of lists
        .unique()                  # get unique DOIs
        .tolist()                  # convert to a list
    )
    Al_alloy_properties_df = pd.read_csv(AL_ALLOY_PROPERTIES_PATH)
    Al_alloy_properties_dois = (
        Al_alloy_properties_df["doi"]
        .dropna()
        .unique()
        .tolist()
    )
    Al_alloy_dataset_DOIs = sorted(list(set(Al_alloy_composition_dois + Al_alloy_properties_dois)))
    return Al_alloy_dataset_DOIs

def get_publisher_metadata(doi_list: list[str]) -> pd.DataFrame:
    meta_dicts = []
    for doi in tqdm(doi_list, "Getting Publisher Metadata"):

        doi_meta = {"doi": doi}
        response = requests.get(
            f"http://dx.doi.org/{doi}",
            headers={"Accept": "application/vnd.crossref.unixsd+xml"},
            timeout=10,
        )
        if response is None:
            meta_dicts.append(doi_meta)
            continue
        soup = BeautifulSoup(response.text, features="xml")

        # Get journal information
        if journal_meta := soup.find("journal_metadata"):
            if full_title := journal_meta.find("full_title"):
                doi_meta["journal"] = full_title.text.strip()

        # Try series metadata if journal metadata not found
        if not doi_meta.get("journal"):
            if series_meta := soup.find("series_metadata"):
                if title := series_meta.find("title"):
                    doi_meta["journal"] = title.text.strip()

        # Get publisher information
        if publisher_tag := soup.find("crm-item", attrs={"name": "publisher-name"}):
            doi_meta["publisher"] = publisher_tag.text.strip()
        else:
            doi_meta["publisher"] = None

        doi_meta["included_in_dataset"] = doi_meta["publisher"] in INCLUDED_PUBLISHERS
        doi_meta["pdf"] = False
        doi_meta["xml"] = False

        meta_dicts.append(doi_meta)

    return pd.DataFrame(meta_dicts)

def construct_dataset(dataset: str, from_scratch: bool, secrets: dict[str, str], excluded_publishers:list[str]) -> None:

    # Check if the user requested dataset is in the recognized options
    if dataset not in DATASET_TO_PATH:
        raise AssertionError(
            f"Specified dataset '{dataset}' not in recognized options '{' '.join(DATASET_TO_PATH.keys())}'"
        )

    dataset_path = DATASET_TO_PATH[dataset]
    publisher_metadata_path = os.path.join(dataset_path, "publisher_metadata.csv")

    # Collect publisher metadata
    if from_scratch:
        logging.info("  Loading dataset...")
        os.makedirs(dataset_path, exist_ok=True)
        if dataset == "zeolite":
            doi_list = get_zeolite_dois(dataset_path)
        elif dataset == "aluminum":
            doi_list = get_aluminum_dois(dataset_path)
        else:
            raise AssertionError("Unrecognized_dataset")

        logging.info("Getting publisher metadata...")
        publisher_metadata = get_publisher_metadata(doi_list)
        publisher_metadata.to_csv(publisher_metadata_path)
    else:
        logging.info("Loading publisher metadata...")
        publisher_metadata = pd.read_csv(publisher_metadata_path)

    grouped_by_publisher = publisher_metadata[publisher_metadata["included_in_dataset"]].groupby(
        "publisher"
    )

    excluded_publishers_short_names = []
    for publisher in excluded_publishers:
        short_name = publisher.lower().split(" ")[0]
        if short_name in INCLUDED_PUBLISHERS_SHORT_NAMES:
            excluded_publishers_short_names.append(short_name)
    
    for publisher, group_index in grouped_by_publisher.groups.items():

        if publisher not in INCLUDED_PUBLISHERS: 
            logging.info(f"Publisher '{publisher}' not in recognized options, skipping.")
            continue

        short_name:str = publisher.lower().split(" ")[0]

        if short_name in excluded_publishers_short_names:
            logging.info(f"Excluding publisher '{publisher}' from download process.")
            continue

        secrets_key = f"{short_name.upper()}_API_KEY"

        assert secrets_key in secrets, f"{secrets_key} not found in secrets.json"

        group = publisher_metadata.loc[group_index]

        match publisher:
            case "Elsevier BV":
                doi_format = download_elsevier_papers(
                    group["doi"].tolist(),
                    api_key=secrets[secrets_key],
                    out_folder=dataset_path,
                )
            case "Springer Science and Business Media LLC":
                doi_format = download_springer_papers(
                    group["doi"].tolist(),
                    api_key=secrets[secrets_key],
                    out_folder=dataset_path,
                )
            case "Wiley":
                doi_format = download_wiley_papers(
                    group["doi"].tolist(),
                    api_key=secrets[secrets_key],
                    out_folder=dataset_path,
                )

        for doi, download_format in doi_format.items():
            match download_format:
                case "xml":
                    publisher_metadata.loc[publisher_metadata["doi"] == doi, "xml"] = True
                case "pdf":
                    publisher_metadata.loc[publisher_metadata["doi"] == doi, "pdf"] = True
                case "both":
                    publisher_metadata.loc[publisher_metadata["doi"] == doi, "xml"] = True
                    publisher_metadata.loc[publisher_metadata["doi"] == doi, "pdf"] = True
                case "failed":
                    pass
                case _:
                    logging.error(f"Unknown format '{download_format}' for DOI: {doi}")

        publisher_metadata.to_csv(publisher_metadata_path)
        logging.info(f"Updated publisher metadata for {publisher}")

    pdf_amount = publisher_metadata["pdf"].sum()
    xml_amount = publisher_metadata["xml"].sum()
    total_amount = len(publisher_metadata)
    pdf_percentage = pdf_amount / total_amount * 100
    xml_percentage = xml_amount / total_amount * 100
    logging.info(f"PDFs: {pdf_amount}/{total_amount} ({pdf_percentage:.2f}%)")
    logging.info(f"XMLs: {xml_amount}/{total_amount} ({xml_percentage:.2f}%)")


def construct_dataset_wrapper(
    dataset: str, from_scratch: bool = False, secrets_file: str = "secrets.json", excluded_publishers: list[str] = []
):
    with open(secrets_file) as f:
        secrets = json.load(f)
    construct_dataset(dataset, from_scratch, secrets, excluded_publishers)

if __name__ == "__main__":
    logging.basicConfig(
        filename="DDD_benchmark_dataset_builder.log",
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s:[%(filename)-25s:%(lineno)-4d] %(message)s",
    )
    fire.Fire(construct_dataset_wrapper)
