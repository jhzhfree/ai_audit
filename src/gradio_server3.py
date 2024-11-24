import os
import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import cchardet as chardet
import gradio as gr
from anomaly_detection import preprocess_data, AnomalyRules, ip_in_subnet
from matplotlib import font_manager as fm
from config import Config
from llm import create_logs_llm_chain
import logging

# 设置中文字体
font_path = "conf/simsun.ttc"
my_font = fm.FontProperties(fname=font_path)

# 初始化配置
logging.basicConfig(
    level=logging.INFO,  # 日志级别
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)
try:
    logger.debug("正在初始化配置...")
    config = Config()
    logger.info("配置初始化成功。")
except Exception as e:
    logger.error(f"配置初始化失败: {str(e)}")
    raise

# 初始化 LLMChain
try:
    logger.debug("正在初始化 LLMChain...")
    llm_chain = create_logs_llm_chain(config)
    logger.info("LLMChain 初始化成功。")
except Exception as e:
    logger.error(f"LLMChain 初始化失败: {str(e)}")
    raise


# ------------------------------
# Gradio 逻辑层
# ------------------------------


def load_rules_from_config(config_path="conf/rule.conf"):
    """
    从配置文件加载规则配置
    :param config_path: 配置文件路径
    :return: 规则配置字典
    """
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            rules = json.load(file)
            print(f"规则配置成功加载：{rules}")
            return rules
    except FileNotFoundError:
        raise FileNotFoundError(f"配置文件未找到：{config_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"配置文件解析失败：{e}")

# 规则管理对象
anomaly_rules = AnomalyRules()

def reset_rules():
    """
    重置规则集合
    """
    global anomaly_rules
    anomaly_rules = AnomalyRules()

def add_rule(rule_name: str, rule_type: str, param: str, target_col: str = None):
    """
    动态添加规则
    """
    global anomaly_rules
    target_col = target_col or rule_name
    try:
        if rule_type == "网段匹配":
            anomaly_rules.add_rule(
                rule_name,
                lambda x: x["登录地址"].apply(lambda ip: ip_in_subnet(ip, param)),
                f"地址匹配网段 {param}"
            )
        elif rule_type == "数值大于":
            param_value = float(param)
            anomaly_rules.add_rule(
                rule_name,
                lambda x: x[target_col] > param_value,
                f"{target_col} 值大于 {param_value}"
            )
        elif rule_type == "字符串匹配":
            anomaly_rules.add_rule(
                rule_name,
                lambda x: x[target_col] == param,
                f"{target_col} 字符串匹配 {param}"
            )
        else:
            raise ValueError(f"未知规则类型: {rule_type}")
    except ValueError as e:
        logging.error(f"规则 '{rule_name}' 的参数无效：{param} ({e})")
        return f"规则 '{rule_name}' 的参数无效：{param}"
    return f"规则 '{rule_name}' 添加成功！"

def detect_encoding(file_path):
    """
    检测文件编码
    """
    with open(file_path, 'rb') as f:
        raw_data = f.read(1024)
        result = chardet.detect(raw_data)
    return result['encoding']

def process_file(file, rules_config):
    """
    处理上传的文件，并应用规则检测异常
    """
    imp_folder = "imp"
    os.makedirs(imp_folder, exist_ok=True)

    # 获取文件路径
    file_path = get_default_file(file, default_file_path=os.path.join(imp_folder, "default_login_logs.csv"))
    
    if file is not None and file.name:  # 检查是否上传文件
        logger.info("检测到上传文件，准备处理...")
        file_path = file.name
        new_file_path = os.path.join(imp_folder, os.path.basename(file_path))
    else:
        logger.info("未检测到上传文件，尝试使用默认文件...")
        file_path = get_default_file(None, default_file_path=os.path.join(imp_folder, "default_login_logs.csv"))
        new_file_path = file_path  # 默认文件无需复制

    if file is not None and file.name:
        with open(file_path, "rb") as src, open(new_file_path, "wb") as dest:
            dest.write(src.read())

    # 检测文件编码
    encoding = detect_encoding(new_file_path)
    print(f"检测到的文件编码为: {encoding}")

    # 根据编码读取文件
    try:
        df = pd.read_csv(new_file_path, encoding=encoding)
        print("上传文件预览:\n", df.head())  # 打印上传文件的前几行
    except UnicodeDecodeError as e:
        raise ValueError(f"文件读取失败，可能是编码不匹配。检测到的编码为 {encoding}, 请确认文件格式！") from e

    # 数据预处理
    df = preprocess_data(df)

    # 动态添加规则
    for rule_name, config in rules_config.items():
        try:
            rule_type, param, target_col = config  # 拆解规则配置
            add_rule(rule_name, rule_type, param, target_col)
        except ValueError as e:
            logging.error(f"规则 '{rule_name}' 配置无效: {e}")
            raise ValueError(f"规则 '{rule_name}' 配置无效: {e}")

    # 应用规则
    df = anomaly_rules.apply_rules(df)

    # 输出结果
    anomalies = df[df["是否异常"] == 1]

    # 保存异常记录到 imp 文件夹
    anomalies_file_path = os.path.join(imp_folder, "anomalies.csv")
    anomalies.to_csv(anomalies_file_path, index=False, encoding=encoding)

    # 统计图
    anomaly_count = df["是否异常"].value_counts()
    labels = anomaly_count.index.map(lambda x: "异常登录" if x == 1 else "正常登录").tolist()

    # 绘制饼图
    plt.figure(figsize=(6, 6))
    plt.pie(
        anomaly_count,
        labels=labels,
        autopct="%1.1f%%" if len(anomaly_count) > 1 else None,
        startangle=90,
        colors=["green", "red"][:len(anomaly_count)],
        textprops={"fontproperties": my_font}  # 设置中文字体
    )
    plt.title("登录行为异常占比", fontproperties=my_font)
    
    # 保存统计图到 imp 文件夹
    plot_file_path = os.path.join(imp_folder, "anomalies_plot.png")
    plt.savefig(plot_file_path, format="png")

    return anomalies_file_path, encoding, plot_file_path


def get_default_file(file, default_file_path="imp/default_login_logs.csv"):
    """
    如果未提供上传文件，则使用默认文件。
    :param file: 上传的文件对象
    :param default_file_path: 默认文件路径
    :return: 文件路径
    """
    if file is None:
        logger.info("未检测到上传文件，尝试使用默认文件...")
        if not os.path.exists(default_file_path):
            raise FileNotFoundError(f"默认文件 {default_file_path} 不存在，请上传文件！")
        return default_file_path
    return file.name

# ------------------------------
# Gradio 接口定义
# ------------------------------
def gradio_ui():
    # Example JSON structure expected by process_file
    # example_rules = {
    #     "频繁失败": ["数值大于", "1", "每分钟失败比例"],  # 每分钟失败比例大于1
    #     "高频登录": ["数值大于", "0.5", "登录频率"],    # 登录频率大于0.5
    #     "可疑地址": ["网段匹配", "10.16.0.0/16", "登录地址"],  # 地址在可疑网段
    #     "未知用户": ["字符串匹配", "unknown_user", "用户ID"]  # 未知用户登录
    # }
    example_config_path = "conf/rule.conf"
    example_rules = load_rules_from_config(example_config_path)

    with gr.Blocks() as demo:
        gr.Markdown("### 登录行为异常检测工具")

        with gr.Row():
            file_input = gr.File(label="上传 CSV 文件", file_types=[".csv"])
            rules_input = gr.JSON(label="规则配置", value=example_rules)

        with gr.Row():
            detect_btn = gr.Button("运行检测")
            reset_btn = gr.Button("重置规则")

        result_csv = gr.File(label="异常记录导出 (CSV)")
        result_plot = gr.Image(label="异常统计饼图")
        result_stats = gr.Textbox(label="分析报告", lines=10) 

        def handle_detect(file, rules_config):
            csv_path, encode, plot_path = process_file(file, rules_config)

            # 加载异常记录，生成统计信息
            anomaly_df = pd.read_csv(csv_path, encoding=encode)
            anomaly_stats = anomaly_df["异常类型"].value_counts().to_dict()
            
            # 调试日志：记录输入内容
            logger.debug("llm_chain...")
            start_date = anomaly_df["登录时间"].min()
            end_date = anomaly_df["登录时间"].max()
            #date_range = (start_date, end_date)
            
            # 调用 LangChain 进行推理
            result = llm_chain.run({
                "start_date": start_date.strip(),
                "end_date": end_date.strip(),
                "stats":anomaly_stats
            })

            # 调试日志：记录推理结果
            logger.debug("分析完成，生成结果成功。")
            logger.debug(f"生成的分析结果: {result}")

            return csv_path, plot_path, result

        detect_btn.click(
            fn=handle_detect,
            inputs=[file_input, rules_input],
            outputs=[result_csv, result_plot, result_stats],
        )

        reset_btn.click(fn=reset_rules, inputs=[], outputs=[])

    return demo


# ------------------------------
# 主程序
# ------------------------------
if __name__ == "__main__":
    demo_app = gradio_ui()
    demo_app.launch(share=True, server_name="0.0.0.0", server_port=7861)
