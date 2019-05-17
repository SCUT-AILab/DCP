import os

def write_log(dir_name, file_name, log_str):
    """
    Write log to file
    :param dir_name:  the path of directory
    :param file_name: the name of the saved file
    :param log_str: the string that need to be saved
    """

    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    with open(os.path.join(dir_name, file_name), "a+") as f:
        f.write(log_str)

def write_settings(settings):
    """
    Save expriment settings to a file
    :param settings: the instance of option
    """

    with open(os.path.join(settings.save_path, "settings.log"), "w") as f:
        for k, v in settings.__dict__.items():
            f.write(str(k) + ": " + str(v) + "\n")