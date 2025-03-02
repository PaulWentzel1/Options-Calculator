from datetime import datetime
import inspect
"""
utils.py contains general utiliy functions
"""



def log_function(action : str, write_log_to_file : bool = False):
    caller_function = inspect.stack()[1].function
    time = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[LOG] Performed action \'{action}\' in function \'{caller_function}\' at [{time}]")