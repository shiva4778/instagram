import os
import sys
from health.logger import logging


class WeightException(Exception):
    def __init__(self,error_message:Exception,error_detail:sys):
        super().__init__(error_message)
        self.error_message=WeightException.get_detailed_error_message(error_message,error_detail)



    @staticmethod
    def get_detailed_error_message(error_message:Exception,error_detail:sys)->str:
        """
        error_message:Exception object is created 
        error_detail:object of sys module
        """
        _,_,exec_tb=error_detail.exc_info()
        # Above line will return tuple and we are unpacking it
        exception_block_line_no=exec_tb.tb_frame.f_lineno
        try_block_line_number=exec_tb.tb_lineno
        file_name=exec_tb.tb_frame.f_code.co_filename

        error_message=f"""
        
        Error occured in script :
        [{file_name}] at
         try block line no: [{try_block_line_number}] and exception block line no:[{exception_block_line_no}] 
        error message:[{error_message}]
        
        """
        return error_message
    
    def __str__(self):
        return self.error_message
    
    def __repr__(self)->str:
        return WeightException.__name__.str()

    
