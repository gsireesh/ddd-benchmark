from abc import ABC, abstractmethod
import os
from typing import Iterable


class DDDPredictorABC(ABC):

    @property
    @abstractmethod
    def accepts_pdf(self) -> bool:
        pass

    @abstractmethod
    def predict_from_xml(self, prompt: str):
        pass

    @abstractmethod
    def predict_from_pdf(self, prompt: str, pdf_filename: str):
        pass

    @abstractmethod
    def predict_from_page_images(self, prompt: str, image_filenames: Iterable[str]):
        pass
