from csv import DictReader

with open (pollution-smal.csv) as arquivo:
    leitor_csv = DictReader(arquivo, delimiter= ' ')
    for linha in leitor csv:
        print(f"{linha[pm10]} {linha[no2]} {linha[so2]} {linha[soot]}")