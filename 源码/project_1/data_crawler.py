import pandas as pd
import numpy as np
import re
import os
import requests


def get_url(page):
    url = "http://8.push2.eastmoney.com/api/qt/clist/get?cb=jQuery1124007976118979482671_1667021505121&pn=" + str(page) + "&pz=200&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&wbp2u=7925395570219924|0|1|0|web&fid=f3&fs=m:90+t:2+f:!50&fields=f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f26,f22,f33,f11,f62,f128,f136,f115,f152,f124,f107,f104,f105,f140,f141,f207,f208,f209,f222&_=1667021505133"
    return url

def get_url_comp(page, indus):
    url = "http://8.push2.eastmoney.com/api/qt/clist/get?cb=jQuery1124007976118979482671_1667021505119&pn=" + str(page) + "&pz=500&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&wbp2u=7925395570219924|0|1|0|web&fid=f3&fs=b:" + indus + "+f:!50&fields=f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f22,f11,f62,f128,f136,f115,f152,f45&_=1667021505710"
    return url

def get_url_klines(indus):
    url = "http://51.push2his.eastmoney.com/api/qt/stock/kline/get?cb=jQuery3510001971463579467425_1667028633458&secid=90." + indus + "&ut=fa5fd1943c7b386f172d6893dbfba10b&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5%2Cf6&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58%2Cf59%2Cf60%2Cf61&klt=101&fqt=1&beg=0&end=20500101&smplmt=1419&lmt=1000000&_=1667028633462"
    return url

def get_content_4_1_pg(page):

    # 获取数据源
    url = get_url(page)

    # 向数据源发出请求
    response = requests.get(url)

    # 从返回的字符串中提取我们要的数据
    getTitles = re.compile("\"f12\":\"(.*?)\",\"f13")
    code = re.findall(getTitles, response.text)
    getTitles = re.compile("\"f14\":\"(.*?)\",\"f15")
    name = re.findall(getTitles, response.text)
    getTitles = re.compile("\"f13\":(.*?),\"f14")
    database_flag = re.findall(getTitles, response.text)

    df = pd.DataFrame()
    if len(code) > 0:
        df["code"] = code
        df["name"] = name
        df["database_flag"] = database_flag

    return df


def get_content_4_all_pg():

    df_tt = pd.DataFrame()
    page = 1
    while True:
        df = get_content_4_1_pg(page)

        if len(df) > 0:
            df_tt = df_tt.append(df)
        else:
            break
        page += 1
    return df_tt


def get_comp_4_1_pg(page, indus):

    # 获取数据源
    url = get_url_comp(page, indus)

    # 向数据源发出请求
    response = requests.get(url)

    # 从返回的字符串中提取我们要的数据
    getTitles = re.compile("\"f12\":\"(.*?)\",\"f13")
    code = re.findall(getTitles, response.text)
    getTitles = re.compile("\"f14\":\"(.*?)\",\"f15")
    name = re.findall(getTitles, response.text)

    df = pd.DataFrame()
    if len(code) > 0:
        df["code"] = code
        df["name"] = name
        df["indus"] = indus

    return df

def get_comp_4_all_pg(df_indus):

    df_tt = pd.DataFrame()
    for indus_i in range(len(df_indus)):
        indus = df_indus.code[indus_i]
        df_4_1indus = pd.DataFrame()
        page = 1
        print("indus:{}".format(indus))
        while True:
            df = get_comp_4_1_pg(page, indus)

            if len(df) > 0:
                df_4_1indus = df_4_1indus.append(df)
            else:
                break
            page += 1
        df_tt = df_tt.append(df_4_1indus, ignore_index=True)
    df_tt = df_tt.drop_duplicates().reset_index(drop=True)
    return df_tt

def get_kline_4_all_indus(df_indus):

    df_indus_klines_tt = pd.DataFrame()
    for indus_code in df_indus.code:
        print("indus_code:{}".format(indus_code))

        url = get_url_klines(indus_code)

        # 向数据源发出请求
        response = requests.get(url)

        # 从返回的字符串中提取我们要的数据
        getTitles = re.compile("klines\":\[\"(.*?)\"\]}}\)")
        tmp = re.findall(getTitles, response.text)
        tmp_lst = tmp[0].split("\",\"")
        tmp_lst1 = [x.split(",") for x in tmp_lst]
        df = pd.DataFrame(tmp_lst1, columns=["date","open","close","high","low","volume","amount","oscil","rise_pct","rise_amount","turnover_ratio"])
        df[["open","close","high","low","volume","amount","oscil","rise_pct","rise_amount","turnover_ratio"]] = df[["open","close","high","low","volume","amount","oscil","rise_pct","rise_amount","turnover_ratio"]].astype(float)
        df["indus"] = indus_code
        df_indus_klines_tt = df_indus_klines_tt.append(df, ignore_index=True)
    df_indus_klines_tt = df_indus_klines_tt.drop_duplicates().reset_index(drop=True)

    return df_indus_klines_tt

if __name__ == "__main__":

    path_out = "./data/"
    if not os.path.exists(path_out):
        os.mkdir(path_out)

    path_out = path_out + "indus/"
    if not os.path.exists(path_out):
        os.mkdir(path_out)

    df_tt = get_content_4_all_pg()
    df_tt.to_csv(path_out + "code_name_indus.csv", index=False, encoding="utf_8_sig")
    df_tt.to_feather(path_out + "code_name_indus.feather")

    df_klines_indus = get_kline_4_all_indus(df_tt)
    df_klines_indus.to_csv(path_out + "df_klines_indus.csv", index=False, encoding="utf_8_sig")
    df_klines_indus.to_feather(path_out + "df_klines_indus.feather")

    df_comp = get_comp_4_all_pg(df_tt)
    df_comp.to_csv(path_out + "df_comp.csv", index=False, encoding="utf_8_sig")
    df_comp.to_feather(path_out + "df_comp.feather")



