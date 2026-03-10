import pandas as pd

df = pd.read_csv("../data.csv", encoding="utf-8", sep="|", low_memory=False,
                 parse_dates=["Invoice Date", 
                              "Order Date", 
                              "Customer First Invoice Date", 
                              "Price last modified date in the ERP", 
                              "Born on date"])

# maknemo stupce koje netrebamo
df = df.drop(columns=[
    "Invoice Line #",
    "Order Date", 
    "Ordered qty",
    "Material cost of part",
    "Labor cost of part", 
    "Overhead cost of part",
    "Sales Channel - Grouping",
    "Born on date",
    "Price last modified date in the ERP",
    "# of unique products on a quote",
    "Order Line #",
    "Order #"
])



# maknemo redove kojima nedostaju ti stupci, jer nam je to najbitniji data
df = df.dropna(subset=["Item Code", "Invoiced price", "CustomerID", "Invoice #", "Invoice Date"])
# fali samo 60 Item Code - zanemarivo malo, i 0.6% svih podataka za Invoiced price, sto mi se isto cini malo,
# eventualno se moze popuniti srednjim vrijednostima cijena s istim Item Code koje imaju blizu datum kupovine, 0.6% mi se zanemarivo cini

# ubacimo Unknown tamo gdje su stupci null, ali samo za str, one s int, float i datetime cemo ostaviti null i poslije vidjeti sto s njima
fill_unknown = [
    "Manufacturing Region",
    "Manufacturing Location Code",
    "Intercompany",
    "Customer industry",
    "Customer Region",
    "Top Customer Group",
    "Product family",
    "Product group",
    "Make vs Buy",
    "Sales Channel - Internal",
    "Sales Channel - External"
]

df[fill_unknown] = df[fill_unknown].fillna("unknown")

