import requests

url = "https://api.bybit.com/spot/v1/symbols"
response = requests.get(url)

print("Status Code:", response.status_code)


def get_trading_pairs():
    url = "https://api.bybit.com/spot/v1/symbols"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Failed to fetch trading pairs.")
    data = response.json()
    return [item["name"] for item in data.get("result", [])]


if response.status_code != 200:
    print("Error fetching data. Status Code:", response.status_code)
else:
    try:
        data = response.json()
        symbols = data.get("result", [])

        # Open a file in write mode and write each symbol name on a new line.
        with open("symbols_to_test.txt", "w") as f:
            for symbol in symbols:
                name = symbol["name"]
                f.write(name + "\n")

        print("Trading pairs have been written to trading_pairs.txt")
    except Exception as e:
        print("Error decoding JSON:", e)
