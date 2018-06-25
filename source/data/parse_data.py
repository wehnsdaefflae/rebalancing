import datetime


def historic_data(in_file_path, out_file_path):
    data = dict()

    min_date, max_date = None, None
    symbols = set()
    with open(in_file_path, mode="r") as file:
        first_line = file.readline()
        header = first_line.split(",")
        for each_line in file:
            cells = each_line.split(",")
            date_str = cells[1]
            symbol_str = cells[2]
            close_str = cells[6]
            sub_dict = data.get(date_str)
            if sub_dict is None:
                sub_dict = {symbol_str: float(close_str)}
                data[date_str] = sub_dict
            else:
                sub_dict[symbol_str] = float(close_str)
            symbols.add(symbol_str)
            date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            if min_date is None or date < min_date:
                min_date = date
            elif max_date is None or max_date < date:
                max_date = date

    sorted_symbols = sorted(symbols)
    with open(out_file_path, mode="w") as file:
        file.write("date\t" + "\t".join(sorted_symbols) + "\n")
        for n in range(int((max_date-min_date).days)):
            each_date = min_date + datetime.timedelta(n)
            each_date_str = each_date.strftime("%Y-%m-%d")
            sub_dict = data.get(each_date_str)
            file.write(each_date_str + "\t" + "\t".join(["{:f}".format(sub_dict.get(x, 0.)) for x in sorted_symbols]) + "\n")


if __name__ == "__main__":
    historic_data("../data/all_currencies.csv", "../data/all_currencies_new.csv")
