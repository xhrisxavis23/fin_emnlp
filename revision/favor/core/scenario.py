from abc import ABC, abstractmethod
from typing import Optional

from core.experiment import Task


class Scenario(ABC):
    @property
    @abstractmethod
    def background(self) -> str:
        """Background information"""

    # TODO: We have to change all the sub classes to override get_source_data_desc instead of `source_data`
    def get_source_data_desc(self, task: Optional[Task] = None) -> str:  # noqa: ARG002
        """
        Source data description

        The choice of data may vary based on the specific task at hand.
        """
        return ""

    @property
    def source_data(self) -> str:
        """
        A convenient shortcut for describing source data
        """
        return self.get_source_data_desc()

    @property
    @abstractmethod
    def interface(self) -> str:
        """Interface description about how to run the code"""

    @property
    @abstractmethod
    def output_format(self) -> str:
        """Output format description"""

    @property
    @abstractmethod
    def simulator(self) -> str:
        """Simulator description"""

    @property
    @abstractmethod
    def rich_style_description(self) -> str:
        """Rich style description to present"""

    @abstractmethod
    def get_scenario_all_desc(
        self,
        task: Optional[Task] = None,
        filtered_tag: Optional[str] = None,
        simple_background: Optional[bool] = None,
    ) -> str:
        """
        Combine all descriptions together

        The scenario description varies based on the task being performed.
        """

    @property
    def experiment_setting(self) -> Optional[str]:
        """Get experiment setting and return as rich text string"""
        return None
