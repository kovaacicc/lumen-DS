with open("../LUMEN_DS.csv", "r", encoding="utf-16") as file:
    data = file.read()

data = data.replace('"', '')

with open("../data.csv", "w", encoding="utf-8") as file:
    file.write(data)