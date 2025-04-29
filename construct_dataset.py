import ast
import json
import logging
from multiprocessing import Pool
import os
import re

from bs4 import BeautifulSoup
import fire
import pandas as pd
import requests
from requests import Session
from requests.adapters import HTTPAdapter
from requests_ratelimiter import LimiterSession
from tqdm import tqdm
from urllib3.util import Retry

from data.dataset_metadata import metadata_by_dataset


API_TIMEOUT = 10

INCLUDED_PUBLISHERS = ["Elsevier BV", "Springer Science and Business Media LLC", "Wiley"]

INCLUDED_PUBLISHERS_SHORT_NAMES = [
    "elsevier",  # for Elsevier BV
    "springer",  # for Springer Science and Business Media LLC
    "wiley",  # for Wiley
]


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


def get_springer_file_links_from_crossref(
    session,
    doi,
    mailto_email,
    timeout,
):
    crossref_response = session.get(
        f"http://dx.doi.org/{doi}?mailto={mailto_email}",
        headers={
            "Accept": "application/vnd.crossref.unixsd+xml",
            "User-Agent": "CMU Materials Metadata Getter",
        },
        timeout=timeout,
    )

    crossref_soup = BeautifulSoup(crossref_response.text, "xml")
    text_mining_header = crossref_soup.find("collection", {"property": "text-mining"})
    if text_mining_header is None:
        return {}

    type_to_link = {
        res["mime_type"].split("/")[1]: res.text for res in text_mining_header.find_all("resource")
    }
    return type_to_link


def add_springer_tables(file_soup: BeautifulSoup, session: Session) -> bytes:

    table_links = [
        "https://link.springer.com" + link["href"]
        for link in file_soup.find_all("a", {"data-track-action": "view table"})
    ]

    top_level_elements = [file_soup]
    for table_link in table_links:
        logging.info(f"Getting table from {table_link}")
        table_page_response = session.get(table_link)
        table_soup = BeautifulSoup(table_page_response.text, "lxml")
        top_level_elements.append(table_soup)

    top_level_soup = BeautifulSoup("", "xml")
    for element in top_level_elements:
        top_level_soup.append(element)

    return str(top_level_soup).encode("utf-8")


def download_springer_papers(
    doi_list: list[str],
    mailto_email: str,
    out_folder: str,
    timeout: float = API_TIMEOUT,
) -> dict[str, str]:

    springer_session = LimiterSession(per_minute=100, per_day=500)
    crossref_session = LimiterSession(per_second=50)
    doi_format = {}
    os.makedirs(os.path.join(out_folder, "xml"), exist_ok=True)
    os.makedirs(os.path.join(out_folder, "pdf"), exist_ok=True)

    for doi in tqdm(doi_list, "Downloading Springer Papers."):
        xml_file_path = os.path.join(out_folder, "xml", clean_doi(doi) + ".xml")
        pdf_file_path = os.path.join(out_folder, "pdf", clean_doi(doi) + ".pdf")

        try:
            type_to_file_url = get_springer_file_links_from_crossref(
                crossref_session, doi, mailto_email, timeout
            )

            download_format = {"pdf": False, "html": False}

            if not type_to_file_url:
                logging.error(f"Springer: no file URLs found for doi {doi}. Skipping.")
                continue

            for format, file_url in type_to_file_url.items():

                if format == "pdf":
                    file_path = pdf_file_path
                elif format == "html":
                    file_path = xml_file_path
                else:
                    logging.info(
                        f"Springer: Found URL for file in unrecognized file type {format}, "
                        f"for DOI {doi}, skipping."
                    )
                    continue

                if os.path.exists(file_path):
                    logging.info(f"Springer: DOI {doi} {format} file already exists, skipping.")
                    download_format[format] = True
                    continue

                response = springer_session.get(file_url)
                if response.status_code != 200:
                    logging.error(
                        f"Springer: DOI {doi} in format {format} failed with status code"
                        f" {response.status_code}."
                    )
                download_format[format] = True

                file_content = response.content
                if format == "html":
                    content_soup = BeautifulSoup(file_content.decode("utf-8"))
                    if content_soup.find_all("iframe", {"title": "Article PDF"}):
                        logging.info(
                            f"Springer: paper {doi} HTML is just an embedded PDF. Skipping."
                        )
                        download_format[format] = False
                        continue

                    logging.info(f"Downloading tables for paper {doi}")
                    file_content = add_springer_tables(content_soup, springer_session)

                with open(file_path, "wb") as f:
                    f.write(file_content)

            if download_format["html"] and download_format["pdf"]:
                doi_format[doi] = "both"
            elif download_format["html"]:
                doi_format[doi] = "xml"
            elif download_format["pdf"]:
                doi_format[doi] = "pdf"
            else:
                doi_format[doi] = "failed"

        except requests.exceptions.Timeout:
            logging.error(f"Springer: DOI {doi} timed out.")
            doi_format[doi] = "failed"
            continue

    pdf_succeeded_count = sum(1 for status in doi_format.values() if status in ["pdf", "both"])
    xml_succeeded_count = sum(1 for status in doi_format.values() if status in ["xml", "both"])
    logging.info(
        f"Springer: {pdf_succeeded_count}/{len(doi_list)} PDFs succeeded, "
        f"{xml_succeeded_count}/{len(doi_list)} XMLs succeeded."
    )
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
    Get the DOIs of concern for the zeolite dataset. We are using a subset of the DOIs; The method
    for making this choice is detailed in eda_zeolite.ipynb.

    Returns:
        List[str]: A list of DOIs of papers in the Zeolite dataset.
    """
    from data.zeolite.constraints import sampled_dois

    return sampled_dois


def get_aluminum_dois(dataset_directory) -> list[str]:
    """
    Extract DOIs of papers in the Al-alloy dataset from the provided CSV file.

    Returns:
        List[str]: A list of DOIs of papers in the Al-alloy dataset.
    """
    AL_ALLOY_COMPOSITION_PATH = os.path.join(dataset_directory, "composition.csv")
    AL_ALLOY_PROPERTIES_PATH = os.path.join(dataset_directory, "property.csv")
    assert os.path.exists(
        AL_ALLOY_COMPOSITION_PATH
    ), f"Al-alloy composition CSV file not found at {AL_ALLOY_COMPOSITION_PATH}"
    assert os.path.exists(
        AL_ALLOY_PROPERTIES_PATH
    ), f"Al-alloy properties CSV file not found at {AL_ALLOY_PROPERTIES_PATH}"
    Al_alloy_composition_df = pd.read_csv(AL_ALLOY_COMPOSITION_PATH, dtype={"ft_doi_list": str})
    Al_alloy_composition_dois = (
        Al_alloy_composition_df["ft_doi_list"]
        .dropna()  # drop NaN values
        .apply(ast.literal_eval)  # convert the string of list to an actual list
        .explode()  # flatten the list of lists
        .unique()  # get unique DOIs
        .tolist()  # convert to a list
    )
    Al_alloy_properties_df = pd.read_csv(AL_ALLOY_PROPERTIES_PATH)
    Al_alloy_properties_dois = Al_alloy_properties_df["doi"].dropna().unique().tolist()
    Al_alloy_dataset_DOIs = sorted(list(set(Al_alloy_composition_dois + Al_alloy_properties_dois)))
    return Al_alloy_dataset_DOIs


def get_doi_metadata(doi: str, session=None) -> dict[str, str]:
    if session is None:
        session = requests.Session()

    retry = Retry(total=5, backoff_factor=2)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)

    doi_meta = {"doi": doi}
    response = session.get(
        f"http://dx.doi.org/{doi}",
        headers={"Accept": "application/vnd.crossref.unixsd+xml"},
        timeout=10,
    )

    if response is None or response.status_code != 200:
        doi_meta["meta_fetch_error"] = True
        return doi_meta

    soup = BeautifulSoup(response.text, features="xml")

    doi_meta["article_type"] = soup.find("doi")["type"]
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

    return doi_meta


def get_publisher_metadata_parallel(doi_list: list[str]) -> pd.DataFrame:
    # no more than 50 per second, so 10/sec with 5 pools. I know this isn't exact, so discounting
    # to 8/sec
    session = LimiterSession(per_second=8)

    with Pool(processes=5) as pool:
        meta_dicts = list(
            tqdm(
                pool.starmap(get_doi_metadata, [(doi, session) for doi in doi_list]),
                total=len(doi_list),
            )
        )
    return pd.DataFrame(meta_dicts)


def get_publisher_metadata(doi_list: list[str]) -> pd.DataFrame:
    meta_dicts = []
    session = LimiterSession(per_second=50)
    for doi in tqdm(doi_list, "Getting Publisher Metadata"):
        doi_meta = get_doi_metadata(doi, session)
        meta_dicts.append(doi_meta)
    return pd.DataFrame(meta_dicts)


def construct_dataset(
    dataset: str, from_scratch: bool, secrets: dict[str, str], excluded_publishers: list[str] | None
) -> None:

    # Check if the user requested dataset is in the recognized options
    if dataset not in metadata_by_dataset:
        raise AssertionError(
            f"Specified dataset '{dataset}' not in recognized options '{' '.join(metadata_by_dataset.keys())}'"
        )

    dataset_path = metadata_by_dataset[dataset]["data_directory"]
    publisher_metadata_path = metadata_by_dataset[dataset]["metadata_csv"]

    # Collect publisher metadata
    if from_scratch:
        logging.info("Loading dataset...")
        os.makedirs(dataset_path, exist_ok=True)
        if dataset == "zeolite":
            doi_list = get_zeolite_dois(dataset_path)
        elif dataset == "aluminum":
            doi_list = get_aluminum_dois(dataset_path)
        else:
            raise AssertionError("Unrecognized_dataset")

        logging.info("Getting publisher metadata...")
        publisher_metadata = get_publisher_metadata_parallel(doi_list)
        publisher_metadata.to_csv(publisher_metadata_path)
    else:
        logging.info("Loading publisher metadata...")
        publisher_metadata = pd.read_csv(publisher_metadata_path)

    grouped_by_publisher = publisher_metadata[publisher_metadata["included_in_dataset"]].groupby(
        "publisher"
    )

    excluded_publishers_short_names = []
    if excluded_publishers is None:
        excluded_publishers = []
    for publisher in excluded_publishers:
        short_name = publisher.lower().split(" ")[0]
        if short_name in INCLUDED_PUBLISHERS_SHORT_NAMES:
            excluded_publishers_short_names.append(short_name)

    for publisher, group_index in grouped_by_publisher.groups.items():

        if publisher not in INCLUDED_PUBLISHERS:
            logging.info(f"Publisher '{publisher}' not in recognized options, skipping.")
            continue

        short_name: str = publisher.lower().split(" ")[0]

        if short_name in excluded_publishers_short_names:
            logging.info(f"Excluding publisher '{publisher}' from download process.")
            continue

        group = publisher_metadata.loc[group_index]

        match publisher:
            case "Elsevier BV":
                doi_format = download_elsevier_papers(
                    group["doi"].tolist(),
                    api_key=secrets["ELSEVIER_API_KEY"],
                    out_folder=dataset_path,
                )
            case "Springer Science and Business Media LLC":
                doi_format = download_springer_papers(
                    group["doi"].tolist(),
                    mailto_email=secrets["SPRINGER_MAILTO"],
                    out_folder=dataset_path,
                )
            case "Wiley":
                doi_format = download_wiley_papers(
                    group["doi"].tolist(),
                    api_key=secrets["WILEY_API_KEY"],
                    out_folder=dataset_path,
                )
            case _:
                doi_format = {}

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
    total_pdfs = (
        publisher_metadata["publisher"]
        .isin(["Wiley", "Springer Science and Business " "Media LLC"])
        .sum()
    )

    total_xml = (
        publisher_metadata["publisher"]
        .isin(["Elsevier BV", "Springer Science and Business " "Media LLC"])
        .sum()
    )
    logging.info(f"PDFs: {pdf_amount}/{total_pdfs}")
    logging.info(f"XMLs: {xml_amount}/{total_xml}")


def construct_dataset_wrapper(
    dataset: str,
    from_scratch: bool = False,
    secrets_file: str = "secrets.json",
    excluded_publishers: list[str] | None = None,
):
    with open(secrets_file) as f:
        secrets = json.load(f)
    construct_dataset(dataset, from_scratch, secrets, excluded_publishers)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s:[%(filename)-25s:%(lineno)-4d] %(message)s",
    )
    fire.Fire(construct_dataset_wrapper)
