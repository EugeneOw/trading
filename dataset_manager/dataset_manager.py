import os
import logging
import pandas as pd
from datetime import datetime
from constants import constants as c
from financial_instruments import rsi
from database_manager import database_manager


class DatasetManager:
    def __init__(self):
        self.valid_files = ""
        self.csv_folder_path: str = ""
        self.dataset_final_path: str = ""
        self.dataset_instr: list = []
        self.combined_dataset: list = []

        self.retrieve_csv_path()
        self.sort_files()
        self.create_dataset()
        self.save_dataset()
        self.add_instr()

    def retrieve_csv_path(self):
        """
        Retrieves the initial CSV path (individual) and the final CSV path (Combined).

        :return: None
        """
        db_file = os.path.abspath(c.PATH_DB)
        database_manager.DBManager(db_file)
        file_path_manager = database_manager.FilePathManager(db_file)
        self.csv_folder_path = file_path_manager.fetch_file_path(3)
        self.dataset_final_path = file_path_manager.fetch_file_path(4)

    @staticmethod
    def extract_date(filename):
        """
        Extracts the date from file name to sort the files by date.

        :param filename: File name
        :type filename: str
        :return: Returns date in datetime format
        """
        try:
            date_str = filename.split('_')[-1].split('-')[0]
            return datetime.strptime(date_str, '%d.%m.%Y')
        except (IndexError, ValueError) as e:
            logging.error(f"Error extracting date from {filename}: {e}")
            return None

    def sort_files(self):
        """
        Sort files based on their date as without, they might misread certain files before others.

        :return: None
        """
        file_paths = []
        for filename in os.listdir(self.csv_folder_path):
            if filename.endswith('.csv'):
                file_path = os.path.join(self.csv_folder_path, filename)
                file_paths.append(file_path)

        valid_files = [
            file_path for file_path in file_paths
            if self.extract_date(os.path.basename(file_path)) is not None
        ]
        self.valid_files = sorted(valid_files, key=lambda x: self.extract_date(os.path.basename(x)))

    def create_dataset(self):
        """
        Creates the dataset that will hold all the combined individual data sets.

        :return: None
        """
        dataframes = []
        for filename in self.valid_files:
            df = pd.read_csv(filename)
            dataframes.append(df)
        if dataframes:
            self.combined_dataset = pd.concat(dataframes, ignore_index=True)
        else:
            self.combined_dataset = pd.DataFrame()

    def save_dataset(self):
        """
        Save the new (combined) data set into a new folder.
        :return: None
        """
        if not isinstance(self.combined_dataset, pd.DataFrame) or self.combined_dataset.empty:
            logging.error("Combined dataset doesn't exist or is empty.")
            return
        self.combined_dataset.to_csv(self.dataset_final_path, index=False)

    def add_instr(self):
        """
        Execute the RSI code which will execute the remaining training instruments code.
        :return: None
        """
        self.dataset_instr = rsi.RSI().calculate_rsi()
        self.dataset_instr.to_csv(self.dataset_final_path, index=False)
