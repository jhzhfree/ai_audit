import os
import json
import pandas as pd
import logging
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
from utils import detect_encoding, ip_in_subnet, my_font, generate_test_data  # 这些方法假设存在utils.py中
from rule import AnomalyRules
import gradio as gr
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import joblib
import datetime
from utils import ip_to_int


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]  # 输出到控制台
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # 单独设置为 DEBUG 级别

class ModelTrainer:
    def __init__(self, config=None):
        """
        初始化 ModelTrainer，设置默认路径和规则文件。
        :param config: 配置字典，用于覆盖默认参数
        """
        default_config = {
            "data_dir": "data",
            "conf_dir": "conf",
            "default_data_file": "default_login_logs.csv",
            "default_test_file": "test_login_logs.csv",
            "default_rule_conf": "rule.yaml",
            "data_file_encoding": "utf-8",
            #"test_file_encoding": "utf-8",
            "model_file": "anomaly_detection_model.pkl"
        }
        
        self.trani_features = ["用户编码", "登录地址", "登录失败次数", "登录成功率", "时间范围分钟", "每分钟失败比例", "连续失败3次"]
        
        #self.trani_features = ["登录失败次数", "登录成功率", "时间范围分钟", "每分钟失败比例", "连续失败3次"]
        
        self.config = {**default_config, **(config or {})}
        self.data_dir = self.config["data_dir"]
        self.conf_dir = self.config["conf_dir"]
        self.default_data_file_path = os.path.join(self.data_dir, self.config["default_data_file"])
        self.default_test_file_path = os.path.join(self.data_dir, self.config["default_test_file"])
        self.default_rule_conf_path = os.path.join(self.conf_dir, self.config["default_rule_conf"])
        self.model_file = self.config["model_file"]

        # 创建目录并检查默认文件
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.conf_dir, exist_ok=True)

        logging.debug(f"初始化 ModelTrainer，配置：{self.config}")
        if not os.path.exists(self.default_data_file_path):
            logging.warning(f"默认数据文件 {self.default_data_file_path} 不存在。")
        if not os.path.exists(self.default_rule_conf_path):
            logging.warning(f"默认规则文件 {self.default_rule_conf_path} 不存在。")

        
    def __init_train_conf__(self, train_data_file=None, rule_conf=None):
        """
        初始化训练配置
        """
        # 检查文件路径是否有效
        logging.debug(f"初始化训练配置，输入文件：{train_data_file}, 规则配置：{rule_conf}")
        if train_data_file:
            if hasattr(train_data_file, "name"):  # 如果是上传的文件对象
                self.train_data_file = train_data_file.name
            else:
                self.train_data_file = train_data_file
            logging.info(f"使用上传的训练数据文件：{self.train_data_file}")
        else:
            if not os.path.exists(self.default_data_file_path):
                raise FileNotFoundError(f"默认数据文件 {self.default_data_file_path} 不存在，请上传文件或检查配置。")
            self.train_data_file = self.default_data_file_path
            logging.info(f"使用默认训练数据文件：{self.train_data_file}")
    
    def load_train_data(self, data_file_type, data_file):
        """
        加载训练数据文件
        """
        logging.debug(f"加载数据文件类型：{data_file_type}, 路径：{data_file}")
        if data_file_type not in ["test", "origin"]:
            raise ValueError(f"参数 data_file_type 的值必须为 'test' 或 'origin'，而不是 '{data_file_type}'")

        encoding = detect_encoding(data_file)
        
        self.config["data_file_encoding"] = encoding
        
        logging.debug(f"检测到的数据文件编码：{encoding}")
        df = pd.read_csv(data_file, encoding=encoding)
        logging.info(f"{data_file_type} 数据文件加载完成，共 {len(df)} 行记录。")
        logging.debug(f"数据预览：\n{df.head()}")
        return df
    
    def preprocess_data(self, df):
        """
        数据清洗与特征提取
        """
        logging.info("开始数据清洗和特征提取...")

        # 检查空值并填充
        if df.isnull().sum().sum() > 0:
            logging.warning(f"原始数据中存在空值，进行填充处理。\n空值统计:\n{df.isnull().sum()}")
            df.fillna(value={"登录时间": pd.NaT, "登录结果": "failure", "用户ID": "unknown_user", "登录地址": "0.0.0.0"}, inplace=True)

        # 转换登录时间
        df["登录时间"] = pd.to_datetime(df["登录时间"], errors="coerce")
        if df["登录时间"].isnull().any():
            logging.warning("部分登录时间无法转换为时间格式，将填充为最早时间。")
            df["登录时间"].fillna(df["登录时间"].min(), inplace=True)

        # 将登录结果转换为数值
        df["登录结果"] = df["登录结果"].apply(lambda x: 1 if x == "success" else 0)
        
        
        df["登录地址"] = df["登录地址"].apply(ip_to_int)

        # 计算登录失败次数
        df["登录失败次数"] = df.groupby("用户ID")["登录结果"].transform(lambda x: (x == 0).sum())

        # 计算登录成功率
        df["登录成功率"] = df.groupby("用户ID")["登录结果"].transform(lambda x: (x == 1).mean())

        # 计算时间范围（分钟）
        df["时间范围分钟"] = df.groupby("用户ID")["登录时间"].transform(
            lambda x: (x.max() - x.min()).total_seconds() / 60 if len(x.dropna()) > 1 else 0
        )

        # 每分钟失败比例
        df["每分钟失败比例"] = df["登录失败次数"] / (df["时间范围分钟"].replace(0, 1))  # 避免除以零

        # 按用户ID和登录时间排序
        df = df.sort_values(by=["用户ID", "登录时间"])

        # 标记失败登录
        df["失败登录"] = (df["登录结果"] == 0).astype(int)

        # 计算时间间隔（秒）
        df["时间间隔"] = df.groupby("用户ID")["登录时间"].diff().dt.total_seconds().fillna(0)

        # 标记连续失败组（时间间隔大于60秒为新组）
        df["失败组"] = (
            (df["失败登录"] == 1) & ((df["时间间隔"] > 60) | (df["时间间隔"].isna()))
        ).cumsum()

        # 统计每组失败次数
        df["连续失败次数"] = df.groupby(["用户ID", "失败组"])["失败登录"].transform("sum")

        # 添加连续失败3次特征
        df["连续失败3次"] = (df["连续失败次数"] >= 3).astype(int)

        # 对用户ID进行编码
        encoder = LabelEncoder()
        df["用户编码"] = encoder.fit_transform(df["用户ID"])
        
        # user_encoding = pd.get_dummies(df['用户ID'], prefix='用户编码')
        # df = pd.concat([df, user_encoding], axis=1)

        # 检查非数值列是否仍有异常值
        non_numeric_columns = df.select_dtypes(include=["object"]).columns
        for col in non_numeric_columns:
            unique_values = df[col].unique()
            logging.info(f"列 {col} 的唯一值：{unique_values[:5]} (仅显示前5个)")

        # 确保所有列无缺失值
        if df.isnull().sum().sum() > 0:
            logging.error("数据清洗后仍存在缺失值，请检查数据处理流程。")
            raise ValueError("数据清洗后仍存在缺失值。")

        logging.info("数据清洗完成。")
        logging.info(f"清洗后的数据预览：\n{df.head()}")
        return df

    
    def build_model(self, df):
        """
        使用 Isolation Forest 构建异常检测模型
        """
        logging.info("开始训练异常检测模型...")

        X = df[self.trani_features]
        
        print(X)
        
        if "异常类型编码" in df.columns:
            y = df["异常类型编码"]
        else:
            logging.info("数据中缺少 '异常类型编码' 列，将跳过监督模型训练。")
            y = None  # 如果没有异常类型列

        if X.isnull().sum().sum() > 0:
            raise ValueError("特征数据中仍存在空值，请检查数据预处理流程。")
        
        #将数据集进行80%（训练集）和20%（验证集）的分割
        from sklearn.model_selection import train_test_split #导入train_test_split工具
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                   test_size=0.2, random_state=0)

        #from sklearn.linear_model import LinearRegression # 导入线性回归算法模型
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        #model = LinearRegression() # 使用线性回归算法创建模型

        model.fit(X_train, y_train) # 用训练集数据，训练机器，拟合函数，确定参数

        y_pred = model.predict(X_test) #预测测试集的Y值
         
        df_ads_pred = X_test.copy() #测试集特征数据
        df_ads_pred['浏览量真值'] = y_test #测试集标签真值
        df_ads_pred['浏览量预测值'] = y_pred #测试集标签预测值
     
        logging.info("模型训练完成，生成异常检测结果。")
        # 保存异常记录到文件
        train_result_file_path = os.path.join(self.data_dir, "train_result.csv")
        df_ads_pred.to_csv(train_result_file_path, index=False, encoding=self.config["data_file_encoding"])
        
        return model, df_ads_pred
    
    def apply_anomaly_rules(self, df):
        anomaly_rules = AnomalyRules()
        df = anomaly_rules.apply_rules(df)
        
        # 映射异常类型为数值
        if "异常类型" in df.columns:
            logging.info("将异常类型从字符串映射为数值...")
            # 创建映射表
            unique_types = df["异常类型"].dropna().unique()
            logging.info(f"异常类型列包含的唯一类型：: {unique_types}")
            unique_types = [t for t in unique_types if t != "正常"]  # 排除"正常"类型
            type_to_id_map = {0: "正常"}  # "正常" 映射为 0
            type_to_id_map.update({idx + 1: t for idx, t in enumerate(unique_types)})  # 为其他类型分配 ID
            logging.info(f"异常类型映射表: {type_to_id_map}")
            
            # 映射异常类型
            #df["异常类型编码"] = df["异常类型"].map(type_to_id_map).fillna(-1).astype(int)  # 将无法匹配的类型映射为 -1
            df["异常类型编码"] = df["异常类型"].apply(lambda x: next((k for k, v in type_to_id_map.items() if v == x), -1))
            tag_unique_types = df["异常类型编码"].dropna().unique()
            logging.info(f"异常类型编码列包含的唯一类型：: {tag_unique_types}")
            
            if '异常类型编码' in df.columns:
                logging.info("异常类型编码列已成功创建。")
            else:
                logging.error("异常类型编码列未成功创建，请检查映射逻辑。")
        else:
            logging.error("数据中未找到 '异常类型' 列，请检查数据处理流程。")
        
        return df,type_to_id_map
     
    # def combine_anomaly_results(self, df, user_stats):
    #     """
    #     合并模型和规则的异常检测结果
    #     """
    #     # 合并必要的列，包括登录失败次数、登录成功率、平均登录间隔等
    #     df = df.merge(user_stats[["用户ID", "登录失败次数", "登录成功率", "平均登录间隔", "模型异常"]], on="用户ID", how="left")
        
    #     df["是否异常"] = df.apply(
    #         lambda row: 1 if row["模型异常"] == -1 or row["规则异常"] == 1 else 0, axis=1
    #     )
    #     logging.info("异常检测结果合并完成")
    #     return df

    def visualize_anomalies(self, df, tran_type):
        """
        绘制按异常类型分类的统计结果
        """
        logging.info("开始绘制异常数据分布图...")
        
        # 获取总记录数
        total_count = len(df)
        logging.debug(f"总记录数: {total_count}")

        # 统计异常类型数量
        anomaly_counts = df["异常类型"].value_counts()
        logging.debug(f"异常类型统计:\n{anomaly_counts}")

        # 统计正常登录数量
        #normal_count = len(df[df["是否异常"] == 0])
        normal_count = len(df[df["异常类型"] == "正常"])
        
        non_normal_count = len(df) - normal_count
        
        logging.debug(f"正常登录记录数: {normal_count}")
        

        computed_total = anomaly_counts.sum()
        if computed_total != total_count:
            logging.warning(f"统计结果总数 ({computed_total}) 与记录总数 ({total_count}) 不一致，请检查数据处理流程！")
        else:
            logging.info(f"统计结果总数与记录总数一致: {computed_total}")
        
        # 合并正常登录和异常类型统计
        all_ratios = non_normal_count / total_count * 100
        logging.debug(f"异常类型占比:\n{all_ratios}")

        # 绘制饼图
        plt.figure(figsize=(10, 8))
        explode = [0.05] * len(anomaly_counts)  # 将每一块分离一点，避免标签重叠
        wedges, texts, autotexts = plt.pie(
            anomaly_counts,
            autopct="%1.1f%%",
            explode=explode,
            startangle=90,
            colors=plt.cm.tab20.colors[:len(anomaly_counts)],  # 使用预定义颜色
            textprops={"fontsize": 10, "fontproperties": my_font},  # 设置文字大小和字体
        )
        
        logging.info("饼图绘制完成。")

        # 设置图例，避免标签重叠
        legend = plt.legend(
            wedges,
            anomaly_counts.index,
            loc="center left",
            bbox_to_anchor=(1, 0.5),  # 图例放置在右侧
            title="登录行为类型",
            fontsize=10,
            title_fontsize=12,
            prop=my_font
        )
    
        if legend.get_title():
            legend.get_title().set_fontproperties(my_font)
            legend.get_title().set_fontsize(14)
            
        logging.info("图例设置完成。")
        
        plt.title("登录行为分布", fontproperties=my_font, fontsize=14)

        # 保存图表
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file_path = os.path.join(self.data_dir, f"anomalies_plot_{timestamp}.png")
        plt.savefig(plot_file_path, format="png", bbox_inches="tight", dpi=300)
        
        logging.info(f"饼图已保存到文件: {plot_file_path}")

        # 显示图表（可选）
        plt.show()

        # 保存异常记录到文件
        #anomalies = df[df["是否异常"] == 1]
        anomalies_file_path = os.path.join(self.data_dir, f"anomalies_{tran_type}.csv")
        df.to_csv(anomalies_file_path, index=False, encoding=self.config["data_file_encoding"])
        logging.info(f"异常记录已保存到文件: {anomalies_file_path}")

        logging.info("\n检测到的异常登录记录：")
        logging.info(df[["用户ID", "登录时间", "登录地址", "登录资源", "登录失败次数", "登录成功率", "异常类型"]].head(10))

        return anomalies_file_path, df, plot_file_path

     

    def evaluate_model(self, model, test_user_stats):
        """
        使用测试数据评估模型
        """
        features = ["登录失败次数", "登录成功率"]
        X_test = test_user_stats[features]

        # 使用模型进行预测
        test_user_stats["模型预测"] = model.predict(X_test)
        test_user_stats["模型异常"] = test_user_stats["模型预测"]  # 将模型预测结果映射为模型异常列

        # 计算准确率、召回率、F1等指标
        y_true = test_user_stats["模型异常"]
        y_pred = test_user_stats["模型预测"]

        # 输出分类报告
        report = classification_report(y_true, y_pred, target_names=["正常", "异常"])
        cm = confusion_matrix(y_true, y_pred)
        
        logging.info(f"模型评估报告：\n{report}")
        logging.info(f"混淆矩阵：\n{cm}")
        return report, cm
    
    def optimize_model(self, X_train, y_train, model):
        """
        使用网格搜索对模型进行优化
        """
        param_grid = {
            "contamination": [0.05, 0.1, 0.15, 0.2],  # 异常比例
            "n_estimators": [50, 100, 200],           # 决策树数量
            "max_samples": [0.8, 1.0]                  # 训练样本比例
        }

        grid_search = GridSearchCV(model, param_grid, cv=5, scoring="accuracy", verbose=2)
        grid_search.fit(X_train, y_train)

        # 输出最优参数
        logging.info(f"最优参数：{grid_search.best_params_}")
        logging.info(f"最佳模型得分：{grid_search.best_score_}")

        return grid_search.best_estimator_
 
    def predict(self, uploaded_file=None, anomalies_map=None):
        """
        使用上传的测试数据文件和已训练模型进行预测，并将结果保存为 CSV 文件。

        :param uploaded_file: 上传的测试数据文件
        :param anomalies_map: 异常类型映射，必须是 JSON 格式的字典
        :return: 保存预测结果的 CSV 文件路径
        """
        # 参数验证
        if not isinstance(anomalies_map, dict):
            raise ValueError("参数 anomalies_map 必须是 JSON 格式的字典。")

        logging.info("开始预测流程...")

        # 初始化预测配置
        if uploaded_file is None:
            uploaded_file = self.default_test_file_path
        self.__init_train_conf__(uploaded_file)

        # 加载测试数据
        df = self.load_train_data(data_file_type="test", data_file=self.train_data_file)
        pre_df = self.preprocess_data(df)
        
        # 提取特征
        X_test = pre_df[self.trani_features]

        # 加载训练好的模型
        
        model_file = self.model_file
        model = joblib.load(model_file)
        logging.info(f"模型已加载，路径: {model_file}")

        # 使用模型进行预测
        pre_df["异常类型编码"] = model.predict(X_test)
        
        logging.info("模型预测完成。")

        pre_df["异常类型编码"] = pre_df["异常类型编码"].round().astype(int)
        
        unique_types2 = pre_df["异常类型编码"].dropna().unique()
        
        logging.info(f"预测异常类型编码2:{unique_types2}")
        
        # 将异常类型编码映射为对应的文本描述，未匹配的编码显示为 "未知"
        pre_df["异常类型"] = pre_df["异常类型编码"].apply(
            lambda x: anomalies_map.get(str(x), "未知")
        )
        
        unique_types2 = pre_df["异常类型"].dropna().unique()
        
        logging.info(f"预测异常类型2:{unique_types2}")
        
        logging.info("异常类型映射完成。")

        # 获取最早时间（最小时间）
        earliest_time = df["登录时间"].min()

        # 获取最晚时间（最大时间）
        latest_time = df["登录时间"].max()
        
        # 统计每种异常类型的数量和其他统计信息
        summary_predict = pre_df.groupby('异常类型').agg(
            total_count=('异常类型', 'size'),
            mean_score=('异常类型编码', 'mean'),
            std_score=('异常类型编码', 'std')
        ).reset_index()
        
            # 转换为 JSON 格式
        summary_predict_json = summary_predict.to_dict(orient='records')
        summary_predict_json = json.dumps(summary_predict_json, ensure_ascii=False, indent=4)
        
        anomalies_file_path, anomalies, plot_file_path = self.visualize_anomalies(pre_df, "pre")

        # 保存预测结果到 CSV 文件
        output_file = "predicted_results.csv"
        pre_df.to_csv(output_file, index=False, encoding=self.config["data_file_encoding"])
        logging.info(f"预测结果已保存到 {output_file}")

        return output_file, earliest_time, latest_time, summary_predict_json, plot_file_path

        
          
    def train(self, uploaded_file=None, rules_input=None):
        """
        执行训练流程
        :param uploaded_file: 上传的 CSV 数据文件
        :param rules_input: 输入的 JSON 格式规则配置
        :return: 文件路径、摘要、图表路径、模型路径、异常类型映射字典
        """
        logging.info("开始训练流程...")
        try:
            # 1. 初始化数据文件和规则
            self.__init_train_conf__(uploaded_file, rules_input)
            logging.info("训练配置初始化完成。")

            # 2. 数据加载
            df = self.load_train_data(data_file_type="origin", data_file=self.train_data_file)
            logging.info(f"数据加载完成，共 {len(df)} 行记录。")

            # 3. 数据清洗和特征提取
            pre_df = self.preprocess_data(df)
            logging.info("数据清洗和特征提取完成。")

            # 4. 应用异常检测规则并生成类型映射
            tag_df, type_to_id_map = self.apply_anomaly_rules(pre_df)


            # 校验映射结果
            if not isinstance(type_to_id_map, dict) or not type_to_id_map:
                raise ValueError("异常类型映射 (type_to_id_map) 格式不正确或为空，请检查规则配置。")
            logging.info(f"异常类型映射生成完成：{type_to_id_map}")

            # 5. 可视化异常数据
            anomalies_file_path, anomalies, plot_file_path = self.visualize_anomalies(tag_df, "train")
            summar_anomalie = anomalies[["用户ID", "登录时间", "登录地址", "登录资源", "登录结果", "登录失败次数", "登录成功率", "异常类型", "异常类型编码"]].head(100)

            logging.info(f"异常数据文件已保存：{anomalies_file_path}")
            logging.info(f"异常统计图表已保存：{plot_file_path}")

            # 6. 模型训练
            model, user_stats = self.build_model(tag_df)
            logging.info("模型训练完成。")

            # 7. 保存模型
            model_file = self.model_file
            joblib.dump(model, model_file)
            logging.info(f"模型已保存到 {model_file}")

            # 返回训练结果
            return anomalies_file_path, summar_anomalie, plot_file_path, model_file, type_to_id_map

        except Exception as e:
            # 捕获任何异常并记录日志
            logging.error(f"训练过程中发生错误：{str(e)}", exc_info=True)
            # 返回空值或错误提示，确保返回的值数量一致
            return None, f"训练失败：{str(e)}", None, None, None


# ------------------------------
# 主程序
# ------------------------------
if __name__ == "__main__":
    logging.info("启动训练流程...")
    
    model = ModelTrainer()
    anomalies_file_path, summar_anomalie, plot_file_path, model_file = model.train()
    logging.info(f"训练完成！\n异常文件路径: {anomalies_file_path}\n图表路径: {plot_file_path}\n模型路径: {model_file}")

