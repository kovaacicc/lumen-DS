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


"""
# maknemo redove kojima nedostaju ti stupci, jer nam je to najbitniji data
df = df.dropna(subset=["Item Code", "Invoiced price", "CustomerID"])

# ubacimo Unknown tamo gdje su nepotpuni podaci
fill_unknown = [
    "Customer Region",
    "Product group",
    "Make vs Buy",
    "Sales Channel - Internal",
    "Sales Channel - External",
    "Manufacturing Region",
    "Manufacturing Location Code",
    "Born on date"
]
df[fill_unknown] = df[fill_unknown].fillna("Unknown")
"""
