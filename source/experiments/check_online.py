import datetime
import os
import time
from socket import socket, AF_INET, SOCK_STREAM


def check_online(host, port):
    with socket(AF_INET, SOCK_STREAM) as s:     # Creates socket
        try:
            s.connect((host, port))             # tries to connect to the host

        except Exception as e:                  # if failed to connect
            return False

    return True


def main():
    host = "google.de"
    port = 443
    file_path = "online_record.csv"
    if not os.path.isfile(file_path):
        header = ["offline_start", "offline_end"]
        with open(file_path, mode="w") as file:
            file.write("\t".join(header) + "\n")

    offline_start = None
    while True:
        is_online = check_online(host, port)

        if offline_start is None and not is_online:
            offline_start = datetime.datetime.now()

        elif offline_start is not None and is_online:
            now = datetime.datetime.now()
            span = now - offline_start

            if 10 < span.seconds:
                row = [offline_start.strftime("%Y-%m-%d %H:%M:%S"), now.strftime("%Y-%m-%d %H:%M:%S")]

                with open(file_path, mode="a") as file:
                    file.write("\t".join(row) + "\n")

            offline_start = None

        time.sleep(5)


if __name__ == "__main__":
    main()
