import subprocess

class TrainLogger:
    """
    Log detail of train parameters
    
    Attributes:
       file_path (str): File to write log
       
    """
    
    def __init__(self, file_path):
        """
        Intialize logger
        
        Args:
           file_path (str): Log file path
           
        """
        self.file_path = file_path

    def write_train_info(self, desc):
        """
        Add some information on top of train log text file
        
        Args:
           desc (str): Descroption of train
        """
        with open(self.file_path, 'a') as file:
            file.write(f'train description : {desc} \n')

    def write_header(self, title):
        """
        Add header to log text file
        
        Args:
           title (str): Title to writen
        """
        with open(self.file_path, 'a') as file:
            file.write(f'\n---------------{title}---------------\n\n')

    def write_dictionary(self, values_dict):
        """
        Write key-values of a dictonary on log text file
        
        values_dict (dict): Dictionary of values
        """
        with open(self.file_path, 'a') as file:
            report_line = ''
            for values_dict_key in values_dict.keys():
                report_line += f'{values_dict_key} : {values_dict.get(values_dict_key)}   '

            report_line += '\n'

            file.write(report_line)

