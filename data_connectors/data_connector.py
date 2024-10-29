from abc import ABC, abstractmethod


class Connector(ABC):
    @abstractmethod
    def get_data(self):
        pass
