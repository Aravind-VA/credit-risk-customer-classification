import logging
import sys
def setup_logging(script_name): # main
    try:
        logger = logging.getLogger(script_name) # here logger is object name where we are storing
        if not logger.handlers:
            logger.setLevel(logging.DEBUG)  # there will 5 types of log levels refer to google for clarification
            # Create a file handler for the script
            handler = logging.FileHandler(f'D:\\ML_Pipeline\\logs\\{script_name}.log', mode='w') # here it will create main.log file as we are calling function in main.py file
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s') # in log file this will create as per mentioned process like time level and message
            handler.setFormatter(formatter)
            logger.addHandler(handler)                                # also mode = w means it will overwrite and erase old data at each run if we want add we need to use mode = a
            logger.propagate = False

        return logger

    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in line no : {er_line.tb_lineno} due to : {er_msg}")