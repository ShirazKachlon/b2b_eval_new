import shutil
from pathlib import Path
import os

class OutputOrganizer:
    def __init__(self, output_path):
        self.output_path = Path(output_path)
        os.makedirs(self.output_path, exist_ok=True)

    def path_handler(self, path_name):
        path = os.path.join(self.output_path, path_name)
        os.makedirs(path, exist_ok=True)
        return path

    def input_folder(self):
        return self.path_handler('inputs')

    def crops_folder(self):
        return os.path.join(self.input_folder(), 'crops')

    def output_folder(self):
        return self.path_handler('outputs')

    def summary_folder(self):
        return self.path_handler('summary')

    def copy_input_folder(self, inputs):
        input_path = self.input_folder()
        for i in inputs:
            try:
                shutil.copy(i, input_path)
            except IOError as e:
                print(f"Error copying file {i}: {e}")
