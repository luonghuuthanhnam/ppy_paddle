import datetime
import os

def logging(log_content, log_dir = "log_files"):
    log_content = str(log_content)
    cur_dateime = (datetime.datetime.now() + datetime.timedelta(hours=7))
    current_date = cur_dateime.strftime('%Y%m%d')
    if os.path.isdir(log_dir) == False:
        os.mkdir(log_dir)
    log_file_name = f"{log_dir}/{current_date}_log.log"
    with open(log_file_name, "a+", encoding="utf-8") as f:
        cur_time = cur_dateime.strftime("%Y-%m-%d %H:%M:%S")
        f.writelines(f"[{cur_time}]\t" + log_content + "\n")
        

def handle_exception(exception_e):
    error_mapping_codition = {
        "(404)": "404"
    }
    return_exception = {
        "errorCode": "500",
        "errorMessage": "Internal Server Error",
    }
    for each in error_mapping_codition.keys():
        if each in str(exception_e):
            return_exception["errorCode"] = error_mapping_codition[each]
            return_exception["errorMessage"] = str(exception_e)
    return return_exception