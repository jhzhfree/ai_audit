import json
import os
import logging
import pandas as pd
from utils import ip_in_subnet
import base64
import marshal
import yaml
import pickle  # 用于序列化和反序列化 Python 对象
import types   # 用于动态创建函数对象等

# 初始化配置
logging.basicConfig(
    level=logging.DEBUG,  # 设置为 DEBUG 级别以捕获调试信息
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]  # 输出到控制台
)

logger = logging.getLogger(__name__)

# ------------------------------
# 异常规则模块
# ------------------------------
class AnomalyRules:
    def __init__(self, rules_config_file="conf/rule.yaml"):
        """
        初始化异常规则管理
        """
        
        self.rules = []
        self.rules_config_file = rules_config_file
        if os.path.exists(rules_config_file):
            logging.info(f"规则配置文件 {rules_config_file} 存在，尝试加载...")
            self.load_rules_from_yaml(rules_config_file)
        else:
            logging.info(f"规则配置文件 {rules_config_file} 不存在，生成默认规则...")
            self.initialize_default_rules()
            self.save_rules_to_yaml(file_path=rules_config_file)

    def initialize_default_rules(self):
        """
        初始化默认规则
        """
        logging.info("初始化默认规则...")

        # 定义内网子网规则
        internal_subnets = ["192.168.0.0/16", "10.0.0.0/8", "172.16.0.0/12"]

        # 添加默认规则
        self.add_rule(
            name="频繁登录失败",
            func=lambda x: x["连续失败次数"] > 3,  # 失败间隔1分钟，连续失败3次
            description="每分钟失败比例超过 0.1",
            external_vars={}  # 无需外部变量
        )

        # 可选规则（已注释）
        # self.add_rule(
        #     name="高频登录",
        #     func=lambda x: x["登录频率"] > 0.5,
        #     description="短时间内高频登录",
        #     external_vars={}  # 无需外部变量
        # )

        self.add_rule(
            name="非正常源地址",
            func=lambda x, subnets: ~x["登录地址"].apply(lambda ip: ip_in_subnet(ip, subnets)),
            description="登录地址属于可疑网段",
            external_vars={"subnets": ["192.168.0.0/16", "10.0.0.0/8", "172.16.0.0/12"]}  # 传递外部变量
        )

        self.add_rule(
            name="未知用户登录",
            func=lambda x: x["用户ID"] == "unknown_user",
            description="未知用户尝试登录",
            external_vars={}  # 无需外部变量
        )

    
    def add_rule(self, name, func, description="", external_vars=None):
            """
            添加新的异常规则
            :param name: 规则名称
            :param func: 检测逻辑（lambda 函数）
            :param description: 规则描述
            :param external_vars: 外部依赖的变量（字典格式）
            """
            if not callable(func):
                raise ValueError(f"规则 {name} 的逻辑必须是一个可调用对象（函数）。")
            self.rules.append({
                "name": name,
                "func": func,
                "description": description,
                "external_vars": external_vars or {}  # 默认为空字典
            })
            logging.info(f"规则已添加: {name} - {description}")

    def apply_rules(self, df):
        """
        应用所有规则并生成 '是否异常' 标记
        :param df: 输入数据框
        :return: 更新后的数据框
        """
        try:
            logging.info("开始应用规则...")
            results = {}

            # 遍历所有规则
            for rule in self.rules:
                rule_name = rule.get("name", "未知规则")
                rule_description = rule.get("description", "")
                rule_func = rule.get("func")
                external_vars = rule.get("external_vars", {})

                # 检查规则的完整性
                if not callable(rule_func):
                    logging.warning(f"规则 '{rule_name}' 的逻辑无效，跳过此规则。")
                    continue

                logging.info(f"正在应用规则: {rule_name} ({rule_description})")

                try:
                    # 执行规则逻辑，将外部变量作为参数传入
                    results[rule_name] = rule_func(df, **external_vars)
                except Exception as e:
                    logging.error(f"规则 '{rule_name}' 应用失败: {e}")
                    results[rule_name] = pd.Series(False, index=df.index)

            # 如果没有规则成功应用，直接返回
            if not results:
                logging.warning("没有有效规则被应用。")
                df["异常类型"] = "正常"
                return df

            # 合并规则结果
            anomaly_flags = pd.DataFrame(results)
            logging.info(f"规则结果合并完成，共 {len(anomaly_flags.columns)} 个规则。")

            df["异常类型"] = anomaly_flags.apply(
                lambda row: ", ".join([name for name, val in row.items() if val]), axis=1
            )

            # 如果没有匹配的规则，设置 "异常类型" 为 "正常"
            df.loc[df["异常类型"] == "", "异常类型"] = "正常"

            # 统计规则匹配数量
            for rule_name, matches in results.items():
                match_count = matches.sum()
                match_ratio = match_count / len(df)
                logging.info(f"规则 '{rule_name}' 匹配数量: {match_count} ({match_ratio:.2%})")

                # 打印部分匹配记录
                matched_records = df.loc[matches, [
                    "用户ID", "登录时间", "登录地址", "登录资源", 
                    "登录结果", "登录失败次数", "登录成功率", 
                    "每分钟失败比例", "异常类型"
                ]].head(10)
                if not matched_records.empty:
                    logging.info(f"规则 '{rule_name}' 部分匹配记录:\n{matched_records}")

            logging.info("规则应用完成")
            return df

        except Exception as e:
            logging.error(f"应用规则时发生错误: {e}", exc_info=True)
            raise


    def load_rules_from_yaml(self, file_path="conf/rule.yaml"):
        """
        从 YAML 文件加载规则
        :param file_path: YAML 文件路径
        """
        logging.info(f"从 YAML 文件加载规则: {file_path}")
        decoded_rules = []  # 用于存储解码后的规则内容
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                rules_data = yaml.safe_load(f)
                for rule in rules_data:
                    name = rule.get("name")
                    description = rule.get("description", "")
                    condition_encoded = rule.get("condition")
                    external_vars = rule.get("external_vars", {})
                    
                    # 检查规则字段是否完整
                    if not name or not condition_encoded:
                        logging.warning(f"规则 {rule} 缺少必要的字段，已跳过。")
                        continue

                    # 解码 Base64 字符串并恢复函数
                    try:
                        code_object = marshal.loads(base64.b64decode(condition_encoded))
                        
                        func = types.FunctionType(code_object, globals())
                        
                    except Exception as decode_error:
                        logging.error(f"解析规则 {name} 的条件时发生错误: {decode_error}")
                        continue

                    self.add_rule(name, func, description, external_vars)
                    
                    # 添加解码后的规则到结果列表
                    decoded_rules.append({
                        "name": name,
                        "description": description,
                        "condition": func,  # 保存解码后的函数对象
                        "external_vars": external_vars
                    })

                logging.info(f"规则成功从 YAML 文件加载，规则数量: {len(self.rules)}")
        except FileNotFoundError:
            logging.warning(f"规则文件 {file_path} 不存在，加载默认规则。")
            self.initialize_default_rules()
            self.save_rules_to_yaml(file_path=file_path)
        except yaml.YAMLError as yaml_error:
            logging.error(f"YAML 文件解析错误: {yaml_error}")
        except Exception as e:
            logging.error(f"加载规则时发生未知错误: {e}")
        return decoded_rules

            
    def save_rules_to_yaml(self, file_path="conf/rule.yaml"):
        """
        将当前规则保存到 YAML 文件
        :param file_path: YAML 文件路径
        """
        logging.info(f"保存规则到 YAML 文件: {file_path}")
        try:
            rules_data = []
            for rule in self.rules:
                code_object = base64.b64encode(marshal.dumps(rule["func"].__code__)).decode("utf-8")
                #free_vars = {var: rule["func"].__globals__[var] for var in rule["func"].__code__.co_freevars}
                #free_vars = {var: rule["func"].__globals__.get(var) for var in rule["func"].__code__.co_freevars}
                external_vars = rule.get("external_vars", {})
                
                #all_vars = {**free_vars, **external_vars}
                
                #serialized_free_vars = base64.b64encode(pickle.dumps(all_vars)).decode("utf-8")
                
                rules_data.append({
                    "name": rule["name"],
                    "description": rule["description"],
                    "condition": code_object,
                    #"free_vars": serialized_free_vars ,
                    "external_vars": external_vars
                })
            os.makedirs(os.path.dirname(file_path), exist_ok=True)  # 确保目录存在
            with open(file_path, "w", encoding="utf-8") as f:
                yaml.dump(rules_data, f, allow_unicode=True, default_flow_style=False)
            logging.info("规则已成功保存到文件。")
        except Exception as e:
            logging.error(f"保存规则时发生错误: {e}")



