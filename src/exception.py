import sys

def error_msg_detail(error, error_detail: sys):
    """
    This function returns a detailed error message.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error occurred in script: [{file_name}] at line number: [{exc_tb.tb_lineno}] with error message: [{str(error)}]"
    return error_message

class CustomException(Exception):
    """
    Custom exception class that inherits from the built-in Exception class.
    It overrides the constructor to provide a detailed error message.
    """
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_msg_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message