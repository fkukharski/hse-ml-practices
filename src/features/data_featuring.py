import pandas as pd
import ipaddress as ip
from urllib.parse import urlparse
from tldextract import extract
import re
from data.global_ import DIR_PATH

def manual_feature_engineering(df_input: pd.DataFrame) -> pd.DataFrame:
    def check_ip(domain):
        try:
            if ip.ip_address(domain):
                return 1
        except:
            return 0
    
    df = df_input.copy()
    df["url_len"] = df["url"].apply(len)
    df["tfldextract"] = df["url"].apply(extract)
    df["num_digits_dom"] = df["tfldextract"].apply(
        lambda x: len(re.findall("(\d+)", x.domain)))
    df["num_@"] = df["url"].apply(lambda x: x.count("@"))
    df["num_slash"] = df["url"].apply(lambda x: x.count("/"))
    df["query"] = (
        df["url"].apply(urlparse).apply(lambda x: 0 if x.query == "" else 1))
    df["caps"] = df["tfldextract"].apply(
        lambda x: 1 if len(re.compile("[A-Z]+").findall(x.domain)) > 0 else 0)
    df["domain_ip"] = df["tfldextract"].apply(lambda x: check_ip(x.domain))
    df.drop(['url', 'tfldextract'], axis=1, inplace=True)
    df.to_csv(DIR_PATH + 'data/processed/processed_data.csv', index=False)
    return df
