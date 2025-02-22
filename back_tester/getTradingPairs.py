import requests

url = "https://api.bybit.com/spot/v1/symbols"
response = requests.get(url)

print("Status Code:", response.status_code)

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
