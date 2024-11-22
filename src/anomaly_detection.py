import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import ipaddress
import logging
from matplotlib import font_manager as fm

# 设置中文字体
font_path = "conf/simsun.ttc"
my_font = fm.FontProperties(fname=font_path)

# ------------------------------
# 设置日志
# ------------------------------
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# ------------------------------
# 工具函数
# ------------------------------
def ip_in_subnet(ip, subnet):
    """
    检查 IP 是否属于指定网段
    """
    try:
        return ipaddress.ip_address(ip) in ipaddress.ip_network(subnet)
    except ValueError:
        return False

# ------------------------------
# 异常规则模块
# ------------------------------
class AnomalyRules:
    def __init__(self):
        """
        初始化异常规则管理
        """
        self.rules = []

    def add_rule(self, name, func, description=""):
        """
        添加新的异常规则
        """
        if not callable(func):
            raise ValueError(f"规则 {name} 的逻辑必须是一个可调用对象（函数）。")
        self.rules.append({"name": name, "func": func, "description": description})
        logging.info(f"规则已添加: {name} - {description}")

    def apply_rules(self, df):
        """
        应用所有规则并生成 '是否异常' 标记
        """
        results = {}
        for rule in self.rules:
            logging.info(f"正在应用规则: {rule['name']} ({rule['description']})")
            results[rule["name"]] = rule["func"](df)

        # 合并规则结果
        anomaly_flags = pd.DataFrame(results)
        df["是否异常"] = anomaly_flags.any(axis=1).astype(int)
        df["异常类型"] = anomaly_flags.apply(
            lambda row: ", ".join([name for name, val in row.items() if val]), axis=1
        )

        # 统计每个规则匹配的数量
        for rule_name, matches in results.items():
            match_count = matches.sum()
            match_ratio = match_count / len(df)
            logging.info(f"规则 '{rule_name}' 匹配数量: {match_count} ({match_ratio:.2%})")

            # 打印部分示例记录
            matched_records = df.loc[matches, ["用户ID", "登录地址", "登录失败次数", "每分钟失败比例", "登录频率"]].head(5)
            logging.debug(f"规则 '{rule_name}' 部分匹配记录:\n{matched_records}")

        logging.info("规则应用完成")
        return df


# ------------------------------
# 数据加载与生成
# ------------------------------
def load_login_data(file_path="data/login_logs.csv"):
    """
    加载指定文件中的用户登录数据，如果文件不存在则生成测试数据并返回
    """
    if os.path.exists(file_path):
        # 文件存在时，读取数据
        try:
            df = pd.read_csv(file_path, encoding="GBK")
            logging.info(f"成功加载文件: {file_path}")
        except Exception as e:
            logging.error(f"无法加载文件 {file_path}: {e}")
            raise
    else:
        # 文件不存在时，生成模拟测试数据
        logging.warning(f"文件 {file_path} 不存在，生成测试数据")
        data = {
            "用户ID": np.random.choice(["user1", "user2", "user3", "user4"], size=1000),
            "登录时间": pd.date_range("2024-11-01", periods=1000, freq="min"),
            "登录地址": np.random.choice(["192.168.1.1", "10.16.0.1", "172.16.0.1", "10.17.0.1"], size=1000),
            "登录资源": np.random.choice(["server1", "server2"], size=1000),
            "是否登录成功": np.random.choice([0, 1], size=1000, p=[0.3, 0.7]),
        }
        df = pd.DataFrame(data)

        # 添加模拟异常
        df.loc[np.random.choice(df.index, 50), "用户ID"] = "unknown_user"
        df.loc[np.random.choice(df.index, 50), "登录地址"] = "suspicious_ip"

        logging.info("模拟登录数据生成完成")
    
    return df

# ------------------------------
# 数据预处理
# ------------------------------
def preprocess_data(df):
    """
    数据清洗和特征提取
    """
    df["登录时间"] = pd.to_datetime(df["登录时间"], errors="coerce")

    # 按用户ID计算时间范围和失败次数
    df["时间范围分钟"] = df.groupby("用户ID")["登录时间"].transform(
        lambda x: max(1, (x.max() - x.min()).total_seconds() / 60)
    )
    df["登录失败次数"] = df.groupby("用户ID")["是否登录成功"].transform(lambda x: (x == 0).sum())
    df["每分钟失败比例"] = df["登录失败次数"] / df["时间范围分钟"]

    # 计算其他特征
    df["登录成功率"] = df.groupby("用户ID")["是否登录成功"].transform("mean")
    df["登录频率"] = df.groupby("用户ID")["登录时间"].transform(
        lambda x: len(x) / ((x.max() - x.min()).total_seconds() / 60)
    )

    # 编码用户ID和地址
    encoder = LabelEncoder()
    df["用户编码"] = encoder.fit_transform(df["用户ID"])
    df["地址编码"] = encoder.fit_transform(df["登录地址"])

    print("预处理后字段:", df.columns.tolist())  # 调试输出
    logging.info("数据预处理完成")
    return df


# ------------------------------
# 构建和训练模型
# ------------------------------
def train_model(df):
    """
    训练分类模型
    """
    features = ["用户编码", "地址编码", "登录失败次数", "登录成功率", "登录频率"]
    X = df[features]
    y = df["是否异常"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    logging.info("模型训练完成")
    print("分类报告：")
    print(classification_report(y_test, y_pred))
    print(f"模型准确率：{accuracy_score(y_test, y_pred):.2f}")

    return model

# ------------------------------
# 可视化异常数据
# ------------------------------
def visualize_anomalies(df):
    """
    绘制异常统计结果
    """
    anomaly_count = df["是否异常"].value_counts()
    labels = anomaly_count.index.map(lambda x: "异常登录" if x == 1 else "正常登录").tolist()

    plt.figure(figsize=(6, 6))
    plt.pie(
        anomaly_count,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=["green", "red"][:len(anomaly_count)],
        textprops={"fontproperties": my_font}  # 设置中文字体
    )
    plt.title("登录行为异常占比", fontproperties=my_font)
    plt.show()

    anomalies = df[df["是否异常"] == 1]
    print("\n异常登录记录：")
    print(anomalies[["用户ID", "登录地址", "登录失败次数", "登录成功率", "登录频率", "异常类型"]].head(10))

# ------------------------------
# 主程序
# ------------------------------
if __name__ == "__main__":
    # 加载数据
    df = load_login_data()

    # 数据预处理
    df = preprocess_data(df)

    # 定义异常规则
    anomaly_rules = AnomalyRules()
    anomaly_rules.add_rule(
        "频繁失败", 
        lambda x: x["每分钟失败比例"] > 0.1,  # 每分钟失败比例大于0.1
        "每分钟失败比例超过 0.1"
    )
    anomaly_rules.add_rule(
        "高频登录", 
        lambda x: x["登录频率"] > 0.5, 
        "短时间内高频登录"
    )
    anomaly_rules.add_rule(
        "可疑地址",
        lambda x: x["登录地址"].apply(lambda ip: ip_in_subnet(ip, "10.16.0.0/16")),
        "登录地址属于可疑网段"
    )
    anomaly_rules.add_rule(
        "未知用户", 
        lambda x: x["用户ID"] == "unknown_user", 
        "未知用户尝试登录"
    )

    # 应用规则
    df = anomaly_rules.apply_rules(df)

    # 模型训练
    model = train_model(df)

    # 可视化异常数据
    visualize_anomalies(df)
