import os
import shutil
import time
from ftplib import FTP, error_perm
from typing import Dict, Set, List

from source.experiments.seedbox_service.Logger import Log

debug = False
temp_dir_path = "/tmp/"


class ResponseStorage:
    def __init__(self):
        self.responses = []

    def catch_response(self, response):
        self.responses.append(response)

    def clear(self):
        self.responses.clear()


def login_server(server: FTP, config: Dict[str, str]):
    Log.info("Logging in to {:s}...".format(server.host))
    server.login(config["user_name"], config["password"])
    server.cwd(config["directory"])
    return server


def deconstruct_path(path: str) -> List[str]:
    folders = []
    while True:
        path, folder = os.path.split(path)

        if folder != "":
            folders.append(folder + ("/" if folder[-1] != "/" else ""))

        else:
            if path != "":
                folders.append(path + ("/" if path[-1] != "/" else ""))

            break

    folders.reverse()
    return folders


def dir_exists(server: FTP, path: str) -> bool:
    try:
        server.cwd(path)
        for _ in deconstruct_path(path):
            server.cwd("..")
        return True

    except error_perm as e:
        return False


def get_server_contents(server: FTP, directory_path: str) -> Set[str]:
    if 0 < len(directory_path) and directory_path[-1] != "/":
        raise ValueError("directory_path does not end in '/'.")

    r = ResponseStorage()
    if 0 < len(directory_path):
        server.dir(directory_path, r.catch_response)
    else:
        server.dir(r.catch_response)

    contents = set()
    for each_response in r.responses:
        _i, _j = 0, 0
        while _i < len(each_response):
            if each_response[_i] == " ":
                while each_response[_i] == " ":
                    _i += 1
                _j += 1
            if _j == 8:
                contents.add(directory_path + each_response[_i:] + ("/" if each_response[0] == "d" else ""))
                break
            elif 8 < _j:
                raise ValueError("Erroneous return response.")
            _i += 1
        if _j < 8:
            raise ValueError("Erroneous return response.")

    return contents


def get_local_contents(directory_path: str) -> Set[str]:
    if 0 < len(directory_path) and directory_path[-1] != "/":
        raise ValueError("directory_path does not end in '/'.")

    if len(directory_path) < 1:
        list_dir = os.listdir(".")
    else:
        list_dir = os.listdir(directory_path)

    contents = set()
    for each_entry in list_dir:
        conc = directory_path + each_entry
        contents.add(conc + ("/" if os.path.isdir(conc) else ""))

    return contents


def download(server: FTP, from_path: str, to_dir_path: str):
    Log.info("Downloading {:s}@{:s} to {:s}@local...".format(from_path, server.host, to_dir_path))
    if 0 < len(to_dir_path) and to_dir_path[-1] != "/":
        raise ValueError("target does not end in '/'.")

    source_dir, source_file = os.path.split(from_path)
    if len(source_file) < 1:
        target_dir_path = to_dir_path + from_path

        if not to_dir_path == "./":
            if os.path.isdir(target_dir_path):
                raise IOError("directory {:s}@local already exists".format(target_dir_path))

            if not debug and not target_dir_path == "./":
                os.mkdir(target_dir_path)

        contents = get_server_contents(server, from_path)
        for each_element in sorted(contents):
            download(server, each_element, target_dir_path)

    elif not debug:
        with open(to_dir_path + source_file, mode="wb") as file:
            server.retrbinary("RETR {:s}".format(from_path), file.write)


def upload(server: FTP, from_path: str, to_dir_path: str):
    Log.info("Uploading {:s}@local to {:s}@{:s}...".format(from_path, to_dir_path, server.host))
    if 0 < len(to_dir_path) and to_dir_path[-1] != "/":
        raise ValueError("target does not end in '/'.")

    source_dir, source_file = os.path.split(from_path)
    if len(source_file) < 1:
        target_dir_path = to_dir_path + from_path

        if not to_dir_path == "./":
            if dir_exists(server, target_dir_path):
                raise IOError("directory {:s}@{:s} already exists".format(target_dir_path, server.host))

            if not debug and not to_dir_path == "./":
                server.mkd(target_dir_path)

        contents = get_local_contents(from_path)
        for each_element in sorted(contents):
            upload(server, each_element, target_dir_path)

    elif not debug:
        with open(from_path, mode="rb") as file:
            server.storbinary("STOR {:s}".format(to_dir_path + source_file), file)


def delete(server: FTP, file_path: str):
    Log.info("Deleting {:s}@{:s}...".format(file_path, server.host))
    if debug or len(file_path) < 1:
        return

    file_dir, file_name = os.path.split(file_path)
    if len(file_name) < 1 and 0 < len(file_dir):
        server.rmd(file_path)
    else:
        server.delete(file_path)


def reset_temp():
    if os.path.isdir(temp_dir_path):
        Log.info("Temp directory reset...")
        shutil.rmtree(temp_dir_path)
        time.sleep(1)

    os.mkdir(temp_dir_path)


def move(from_server: FTP, from_path: str, to_server: FTP, to_dir_path: str):
    reset_temp()
    _move(from_server, from_path, to_server, to_dir_path)
    reset_temp()


def _move(from_server: FTP, from_path: str, to_server: FTP, to_dir_path: str):
    Log.info("Moving {:s}@{:s} to {:s}@{:s}...".format(from_path, from_server.host, to_dir_path, to_server.host))
    if 0 < len(to_dir_path) and not to_dir_path[-1] == "/":
        raise ValueError("target does not end in '/'.")

    from_dir, from_file = os.path.split(from_path)

    if 0 < len(from_file):
        download(from_server, from_path, temp_dir_path)
        temp_file_path = temp_dir_path + from_file
        upload(to_server, temp_file_path, to_dir_path)

        if debug:
            return

        Log.info("Removing temp file {:s}".format(temp_file_path))
        os.remove(temp_file_path)

    else:
        if to_dir_path == from_path == "./":
            final_path = "./"
        else:
            final_path = to_dir_path + from_path

        if 0 < len(final_path) and not final_path == "./":
            if dir_exists(to_server, final_path):
                raise IOError("directory {:s}@{:s} already exists".format(final_path, to_server.host))

            if not debug:
                to_server.mkd(final_path[:-1] if final_path[-1] == "/" else final_path)

        contents = get_server_contents(from_server, from_path)
        for each_element in sorted(contents):
            _move(from_server, each_element, to_server, final_path)

    delete(from_server, from_path)
