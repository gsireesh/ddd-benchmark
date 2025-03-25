import ast
import json
import logging
import os
import re
import time
from typing import List, Dict, Optional, cast, Callable

from bs4 import BeautifulSoup
import fire
import pandas as pd
import requests
from requests_ratelimiter import LimiterSession
from tqdm import tqdm

METADATA_KEYS = ["doi", "publisher", "journal", "included_in_dataset"]

INCLUDED_PUBLISHERS = ["Elsevier BV", "Wiley", "Springer Science and Business Media LLC"]

API_TIMEOUT = 10

DATASET_TO_PATH = {
    "zeolite": "data/zeolite",
    "aluminum": "data/aluminum",
}


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
    zeolite_dataset_df = pd.read_excel(ZEOLITE_DATASET_PATH)
    zeolite_dataset_df = zeolite_dataset_df.dropna(axis=0, thresh=20)
    zeolite_dataset_DOIs = zeolite_dataset_df["doi"].dropna().unique().tolist()
    return zeolite_dataset_DOIs


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
            if full_title := journal_meta.find("full_title"):  # type: ignore
                doi_meta["journal"] = full_title.text.strip()

        # Try series metadata if journal metadata not found
        if not doi_meta.get("journal"):
            if series_meta := soup.find("series_metadata"):
                if title := series_meta.find("title"):  # type: ignore
                    doi_meta["journal"] = title.text.strip()

        # Get publisher information
        if publisher_tag := soup.find("crm-item", attrs={"name": "publisher-name"}):
            doi_meta["publisher"] = publisher_tag.text.strip()
        else:
            doi_meta["publisher"] = None

        doi_meta["included_in_dataset"] = doi_meta["publisher"] in INCLUDED_PUBLISHERS

        meta_dicts.append(doi_meta)

    return pd.DataFrame(meta_dicts)


def download_elsevier_papers(
    doi_list: list[str],
    api_key: str,
    out_folder: str,
    timeout: float = API_TIMEOUT,
) -> tuple[list[str], list[str]]:
    session = LimiterSession(per_second=10)
    succeeded_dois = []
    failed_dois = []
    for doi in tqdm(doi_list, "Downloading Elsevier Papers."):
        response = session.get(
            f"https://api.elsevier.com/content/article/doi/{doi}",
            headers={"Accept": "application/xml", "X-ELS-APIKey": api_key},
            timeout=timeout,
        )
        if response.status_code == 200:
            with open(os.path.join(out_folder, "xml", clean_doi(doi) + ".xml")) as f:
                f.write(response.text)
            succeeded_dois.append(doi)
        else:
            logging.error(f"Paper {doi} (Elsevier) failed with status code {response.status_code}.")
            failed_dois.append(doi)
    return succeeded_dois, failed_dois


def download_springer_papers(
    doi_list: list[str], api_key: str, out_folder: str, timeout: float = API_TIMEOUT
) -> tuple[list[str], list[str]]:
    session = LimiterSession(per_minute=100, per_day=500)
    succeeded_dois = []
    failed_dois = []
    for doi in tqdm(doi_list, "Downloading Springer Papers."):
        response = session.get(
            f"https://api.springer.com/openaccess/jats?q=doi:{doi}&api_key={api_key}",
            timeout=timeout,
        )
        if response.status_code == 200:
            with open(os.path.join(out_folder, "xml", clean_doi(doi) + ".xml")) as f:
                f.write(response.text)
            succeeded_dois.append(doi)
        else:
            logging.error(f"Paper {doi} (Springer) failed with status code {response.status_code}.")
            failed_dois.append(doi)
    return succeeded_dois, failed_dois


def download_wiley_papers(
    doi_list: list[str], api_key: str, out_folder: str, timeout=API_TIMEOUT
) -> tuple[list[str], list[str]]:
    session = LimiterSession(per_second=3, per_minute=6)
    succeeded_dois = []
    failed_dois = []
    for doi in tqdm(doi_list, "Downloading Wiley Papers."):
        response = session.get(
            f"https://api.wiley.com/onlinelibrary/tdm/v1/articles/{doi}",
            headers={"Wiley-TDM-Client-Token": api_key},
            timeout=timeout,
        )
        if response.status_code == 200:
            with open(os.path.join(out_folder, "pdf", clean_doi(doi) + ".pdf")) as f:
                f.write(response.text)
            succeeded_dois.append(doi)
        else:
            logging.error(f"Paper {doi} (Wiley) failed with status code {response.status_code}.")
            failed_dois.append(doi)
    return succeeded_dois, failed_dois


def construct_dataset(dataset: str, from_scratch: bool, secrets: dict[str, str]):

    if dataset not in DATASET_TO_PATH:
        raise AssertionError(
            f"Specified dataset '{dataset}' not in recognized options '{' '.join(DATASET_TO_PATH.keys())}'"
        )

    dataset_path = DATASET_TO_PATH[dataset]
    publisher_metadata_path = os.path.join(dataset_path, "publisher_metadata.csv")

    if from_scratch:
        logging.info("  Loading dataset...")
        if dataset == "zeolite":
            doi_list = get_zeolite_dois(dataset_path)
        else:
            raise AssertionError("Unrecognized_dataset")

        logging.info("Getting publisher metadata...")
        publisher_metadata = get_publisher_metadata(doi_list)
        publisher_metadata.to_csv(publisher_metadata_path)
    else:
        logging.info("Loading publisher metadata...")
        publisher_metadata = pd.read_csv(publisher_metadata_path)

    grouped_by_publisher = publisher_metadata[publisher_metadata["included_in_dataset"]].groupby(
        "Publisher"
    )

    for publisher, group_index in grouped_by_publisher.groups:
        group = publisher_metadata.loc[group_index]
        succeeded_dois = []
        failed_dois = []
        if publisher == "Elsevier BV":
            s, f = download_elsevier_papers(
                group["doi"].tolist(), api_key=secrets["ELSEVIER_API_KEY"], out_folder=dataset_path
            )
            succeeded_dois.extend(s)
            failed_dois.extend(f)
        elif publisher == "Springer Science and Business Media LLC":
            s, f = download_springer_papers(
                group["doi"].tolist(), api_key=secrets["SPRINGER_API_KEY"], out_folder=dataset_path
            )
            succeeded_dois.extend(s)
            failed_dois.extend(f)
        elif publisher == "Wiley":
            s, f = download_wiley_papers(
                group["doi"].tolist(), api_key=secrets["WILEY_API_KEY"], out_folder=dataset_path
            )
            succeeded_dois.extend(s)
            failed_dois.extend(f)
    for doi in succeeded_dois:
        publisher_metadata


def construct_dataset_wrapper(
    dataset: str, from_scratch: bool = False, secrets_file: str = "secrets.json"
):
    with open(secrets_file) as f:
        secrets = json.load(f)
    construct_dataset(dataset, from_scratch, secrets)


if __name__ == "__main__":
    logging.basicConfig(
        filename="DDD_benchmark_dataset_builder.log",
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s:[%(filename)-25s:%(lineno)-4d] %(message)s",
    )
    fire.Fire(construct_dataset_wrapper)
