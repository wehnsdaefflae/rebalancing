import glob
import os


def main():
    folder_old = "C:/Users/Mark/PycharmProjects/rebalancing/data/binance/old/"
    folder_new = "C:/Users/Mark/PycharmProjects/rebalancing/data/binance/new/"

    files = {os.path.basename(x) for x in glob.glob(folder_old + "*.csv")} & {os.path.basename(x) for x in glob.glob(folder_new + "*.csv")}

    for each_file_name in sorted(files):
        print(f"reading {each_file_name:s}...")
        timestamp_last = -1
        with open(folder_old + each_file_name, mode="r") as file_old:
            for line in file_old:
                timestamp_this = int(line.split("\t", maxsplit=1)[0])
                timestamp_last = max(timestamp_last, timestamp_this)

        print(f"appending {each_file_name:s}...")
        with open(folder_new + each_file_name, mode="r") as file_new, open(folder_old + each_file_name, mode="a") as file_old:
            for line in file_new:
                timestamp_this = int(line.split("\t", maxsplit=1)[0])
                if timestamp_last < timestamp_this:
                    file_old.write(line)


if __name__ == "__main__":
    main()
