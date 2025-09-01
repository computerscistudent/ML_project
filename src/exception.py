import sys
from typing import Any
import logging

def error_message_detail(error:Exception) -> str:
    _, _, exc_tb = sys.exc_info()

    if exc_tb:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        error_message = f"Error occurred in python script name [{file_name}] line number [{line_number}] error message [{str(error)}]"
        return error_message
    else:
        return f"An error occurred: [{str(error)}]"

class customException(Exception):
    def __init__(self, error: Exception):
        formatted_message = error_message_detail(error)
        super().__init__(formatted_message)
        self.error_message = formatted_message


    def __str__(self) -> str:
        return self.error_message 


