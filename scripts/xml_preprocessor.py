from bs4 import BeautifulSoup # requires lxml
from bs4 import Comment
from pathlib import Path
from enum import StrEnum

class Publisher(StrEnum):
    ELSEVIER = "Elsevier"
    SPRINGER = "Springer"
    DEFAULT = "Unknown"

class Config:
    FULL_DATASET = True  # If True, use the full dataset; if False, use training set only
    DATASET_CATEGORY = "aluminum"  # zeolite, aluminum
    VERSION = "short"  # Version of the dataset

    # --- Generate paths ---
    BASE_PATH = Path(__file__).parent.parent
    DATA_DIR = BASE_PATH / "data" / f"{DATASET_CATEGORY.lower()}" / "xml"
    OUT_DIR  = BASE_PATH / "data" / f"{DATASET_CATEGORY.lower()}" / f"xml{VERSION}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Training set DOIs ---
    TRAIN_SET_DOIS = {
        "zeolite": [
            "10.1016/j.micromeso.2006.10.023",
            "10.1016/j.solidstatesciences.2007.08.002",
            "10.1007/s10934-015-0051-5",
            "10.1002/anie.200461911",
            "10.1007/s11244-013-0170-7",
        ],
        "aluminum": [
            "10.1016/j.scriptamat.2004.07.020",
            "10.1016/j.engfailanal.2010.08.007",
            "10.1016/j.jallcom.2013.08.214",
            "10.1007/s11661-010-0395-z",
            "10.1007/s11661-008-9739-3",
            "10.1007/s11837-016-1896-z",
        ],
    }.get(DATASET_CATEGORY, [])

def reading_xml(file_path: Path) -> BeautifulSoup:
    """
    Reads an XML file and returns a BeautifulSoup object.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist.")
    with open(file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')
    return soup

def identify_publisher(soup: BeautifulSoup) -> Publisher:
    if soup.find('xocs:srctitle'):
        return Publisher.ELSEVIER
    if soup.find('meta'):
        return Publisher.SPRINGER
    return Publisher.DEFAULT

def get_caption_elsevier(tag: BeautifulSoup)->str:
    """
    Extracts the caption text from a given tag.
    """
    caption = tag.find('ce:simple-para')
    if caption:
        return caption.get_text(strip=True)
    return "No caption found"

def extract_springer_features(soup: BeautifulSoup) -> BeautifulSoup:
    for style in soup.find_all('style'):
        style.decompose()
    for script in soup.find_all('script'):
        script.decompose()
    for footer in soup.find_all('footer'):
        footer.decompose()
    for graphic in soup.find_all('div', { 'aria-hidden': "true" ,  'class': "u-visually-hidden", 'data-test': "darwin-icons"}):
        graphic.decompose()
    for link in soup.find_all('link'):
        link.decompose()
    for alink in soup.find_all('a'):
        alink.decompose()
    for bodysuffix in soup.find_all('div', {'id': "MagazineFulltextArticleBodySuffix"}):
        bodysuffix.decompose()
    for head in soup.find_all('head'):
        head.decompose()
    for noscript in soup.find_all('noscript'):
        noscript.decompose()

    # remove comments
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.decompose()
    return soup

def extract_elsevier_features(soup: BeautifulSoup) -> BeautifulSoup:
    for obj in soup.find_all('object'):
        obj.decompose()
    
    for reference in soup.find_all("xocs:references"):
        reference.decompose()

    for attachment in soup.find_all("xocs:attachments"):
        attachment.decompose()

    for entry in soup.find_all("entry"):
        if 'xmlns' in entry.attrs:  #type: ignore
            del entry.attrs['xmlns']    #type: ignore

    for tail in soup.find_all("tail"):
        tail.decompose()

    for itemtoc in soup.find_all("xocs:item-toc"):
        itemtoc.decompose()
    return soup

def extract_features(soup: BeautifulSoup) -> BeautifulSoup:
    publisher = identify_publisher(soup)
    print(f"Publisher: {publisher}")
    match publisher:
        case Publisher.ELSEVIER:
            return extract_elsevier_features(soup)
        case Publisher.SPRINGER:
            return extract_springer_features(soup)
        case _:
            raise ValueError(f"Unknown publisher: {publisher}")

def main():
    if Config.FULL_DATASET:
        print("USING FULL DATASET!!!")
        file_paths = list(Config.DATA_DIR.glob("*.xml"))
    else:
        file_paths = [
            Config.DATA_DIR / (doi.replace("/", "_") + ".xml") 
            for doi in Config.TRAIN_SET_DOIS
        ]
    
    # # For quick testing with a single file
    # file_paths = [Config.DATA_DIR / "10.1007_s11661-014-2207-3.xml"]

    count = 0 
    for file_path in file_paths:
        if not file_path.exists():
            print(f"File {file_path.name} does not exist, skipping...")
            continue
        print(f"Processing {file_path.name}...")
        soup = reading_xml(file_path)
        full_text = extract_features(soup)#.prettify()
        with open(Config.OUT_DIR / file_path.name, 'w', encoding='utf-8') as out_file:
            out_file.write(str(full_text))
        print(f"Processed {file_path.name} and saved")
        count += 1
        # if count == 3: break

def total_characters() -> None:
    file_char = list()
    file_paths = list(Config.OUT_DIR.glob("*.xml"))
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            file_char.append(len(content))
    print(f"Max char: {max(file_char)}")
    print(f"Min char: {min(file_char)}")
    print(f"Avg char: {sum(file_char)/len(file_char)}")

if __name__ == "__main__":
    # main()
    total_characters()