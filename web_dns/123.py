import pandas as pd
def get_sta():
    """
    获取恶意样本统计文件
    :return:
    """
    df = pd.read_csv(sta_file)
    df = df.sort_values(by=['time'])
    dic = {}
    dic['time'] = df["time"].tolist()
    dic['count'] = df["count"].tolist()
    # dic['count'] = [str(i) for i in dic['count']]
    t = range(0,len(dic["time"]),15)
    a = [dic["time"][i] for i in t]
    print(dic['time'])
    print(a)
    print(dic['count'])

sta_file = r"./data/result_sta.csv"
get_sta()