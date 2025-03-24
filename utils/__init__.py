import platform


def emojis(str=""):
    return str.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else str
