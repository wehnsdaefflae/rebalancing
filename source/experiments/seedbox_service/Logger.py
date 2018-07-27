from datetime import datetime


class Log:
    log_path = ""

    @staticmethod
    def info(message: str):
        print(message)

        if 0 < len(Log.log_path):
            with open(Log.log_path, mode="a") as file:
                now = datetime.now()
                row = [now.strftime("%Y-%m-%d %H:%M:%S"), message]
                file.write("\t".join(row) + "\n")