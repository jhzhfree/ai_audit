import pandas as pd
import pickle
import matplotlib.pyplot as plt
import cchardet as chardet
from matplotlib import font_manager as fm
import json
import ipaddress
import random
import logging
import socket
import struct

logging.basicConfig(
    level=logging.DEBUG,  # 设置为 DEBUG 级别以捕获调试信息
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]  # 输出到控制台
)

def ip_to_int(ip):
    return struct.unpack("!I", socket.inet_aton(ip))[0]

def process_rules(df, rules_json):
    # 假设 rules_json 是一个包含规则的字典，按规则处理数据
    rule_results = []
    stats = {"规则数": 0, "异常记录数": 0}
    # 示例规则逻辑处理
    # for rule in rules_json:
    #    ... 
    return rule_results, stats

def train_model(df):
    # 示例训练逻辑
    model = {"mock_model": "demo"}  # 替换为实际模型训练代码
    plot = plt.figure()
    # 绘制训练效果
    plt.plot([1, 2, 3], [0.9, 0.95, 0.98])  # 示例数据
    return pickle.dumps(model), plot

def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def predict(model, df):
    # 示例预测逻辑
    predictions = df.copy()
    predictions["预测值"] = 0.5  # 替换为实际预测逻辑
    return predictions

def evaluate_model(model, test_data):
    """
    示例 evaluate_model 函数: 返回模型评估结果
    """
    accuracy = 0.95  # 示例值，替换为实际实现
    return {"accuracy": accuracy, "details": "示例评估"}

def detect_encoding(file_path):
    """
    检测文件编码
    """
    with open(file_path, 'rb') as f:
        raw_data = f.read(1024)
        result = chardet.detect(raw_data)
    return result['encoding']

def ip_in_subnet(ip, subnets):
    """
    检查 IP 是否属于指定网段列表中的任何一个
    """
    try:
        ip_addr = ipaddress.ip_address(ip)
        for subnet in subnets:
            if ip_addr in ipaddress.ip_network(subnet):
                return True
        return False
    except ValueError:
        return False
    
# 设置中文字体
font_path = "conf/simsun.ttc"
my_font = fm.FontProperties(fname=font_path)


def generate_test_data(file_path="user_login_test_data.csv", num_records=100):
    """
    生成测试数据集，包括正常和异常数据，列与原始数据一致
    """
    user_ids = [f"test_user{i}" for i in range(1, 21)]
    ip_addresses = ["192.168.1." + str(i) for i in range(50, 60)] + \
                   ["10.0.0." + str(i) for i in range(50, 60)] + \
                   ["172.16.0." + str(i) for i in range(50, 60)] + \
                   ["203.0.113." + str(i) for i in range(50, 60)]
    login_resources = ["server1", "server2", "server3"]
    login_results = ["success", "failure"]

    data = []

    # 生成正常数据 (98条)
    for _ in range(num_records - 2):
        user = random.choice(user_ids)
        login_times = pd.date_range(
            start="2023-02-01 08:00:00",
            periods=random.randint(5, 10),
            freq="H"
        )
        for login_time in login_times:
            record = {
                "用户ID": user,
                "登录时间": login_time,
                "登录地址": random.choice(ip_addresses),
                "登录资源": random.choice(login_resources),
                "登录结果": np.random.choice(["success", "failure"], p=[0.9, 0.1])  # 修正
            }
            data.append(record)

    # 生成异常数据 (2条)

    # 异常1：频繁登录失败
    user = random.choice(user_ids)
    failure_times = pd.date_range("2023-02-02 09:00:00", periods=5, freq="T")
    for login_time in failure_times:
        record = {
            "用户ID": user,
            "登录时间": login_time,
            "登录地址": random.choice(ip_addresses),
            "登录资源": random.choice(login_resources),
            "登录结果": "failure"
        }
        data.append(record)

    # 异常2：同账号多地址登录
    user = random.choice(user_ids)
    login_time = pd.Timestamp("2023-02-03 10:00:00")
    ip_list = random.sample(ip_addresses, 5)
    for ip in ip_list:
        record = {
            "用户ID": user,
            "登录时间": login_time,
            "登录地址": ip,
            "登录资源": random.choice(login_resources),
            "登录结果": "success"
        }
        data.append(record)

    # 转换为 DataFrame
    df = pd.DataFrame(data)

    # 打乱数据顺序
    df = df.sample(frac=1).reset_index(drop=True)

    # 保存测试数据
    df.to_csv(file_path, index=False, encoding="utf-8")
    logging.info(f"测试数据已生成并保存到 {file_path}")
