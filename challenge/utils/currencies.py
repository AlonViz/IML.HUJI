import pandas as pd

"""Load currencies file"""
xls = pd.read_csv('bin/currencies_2018.csv', encoding='latin-1')
currency_dict = dict(zip(xls.id, xls.value))


def convert_currency(currency, amount):
    return amount / currency_dict[currency]
