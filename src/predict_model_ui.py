import gradio as gr
import json
from train_data import ModelTrainer
from chat import ChatbotManager
import asyncio

class PredictModelUI:
    def __init__(self):
        """
        初始化 UI 界面
        """
        print("PredictModelUI initialized.")

    async def  handle_predict(self, uploaded_file, anomalies_map_str):
        """
        点击按钮后的处理逻辑
        :param uploaded_file: 上传的测试数据文件
        :param anomalies_map_str: JSON 字符串格式的异常映射信息
        :return: 预测结果文件路径和统计信息
        """
        try:
            # 将字符串解析为 JSON
            anomalies_map = json.loads(anomalies_map_str)
            
            # 检查是否是字典格式
            if not isinstance(anomalies_map, dict):
                raise ValueError("异常映射信息必须是 JSON 格式的字典。")
            
            # 创建 ModelTrainer 实例并调用 predict 方法
            model_trainer = ModelTrainer()
            anomalies_file,  earliest_time, latest_time, stats_summary, plot_file_path = model_trainer.predict(uploaded_file, anomalies_map)
            
            print(f"earliest_time {earliest_time}")
            print(f"latest_time {latest_time}")
            
            chatmanager = ChatbotManager()
            report_output =  chatmanager.create_report(earliest_time, latest_time, stats_summary)
            
            print(f"reportout {report_output}")
            
            return anomalies_file, report_output, plot_file_path
        except json.JSONDecodeError:
            # 如果 JSON 解析失败
            return None, "异常映射信息格式不正确，请输入有效的 JSON 格式字典。"
        except Exception as e:
            # 处理其他错误
            return None, f"失败：{str(e)}"

    def render(self):
        """
        渲染预测界面
        """
        with gr.Blocks() as predict_ui:
            # 上传文件
            with gr.Row():
                uploaded_file = gr.File(label="上传 CSV 文件", file_types=[".csv"])
            
            # 异常映射信息输入框
            with gr.Row():
                anomalies_map = gr.Textbox(
                    label="模型异常映射信息（JSON 格式）", 
                    placeholder='{"异常类型1": 1, "异常类型2": 2}', 
                    interactive=True
                )
            
            # 按钮
            with gr.Row():
                predict_button = gr.Button("开始预测并生成报告")
            
            # 预测结果输出
            with gr.Row():
                plot_output = gr.Image(label="训练数据集图例")
            # 预测结果输出
            with gr.Row():
                report_output = gr.Textbox(label="检测报告")
            # 预测结果输出
            with gr.Row():
                anomalies_output = gr.File(label="预测结果记录文件（CSV）")
            # 绑定按钮事件
            predict_button.click(
                fn=self.handle_predict,
                inputs=[uploaded_file, anomalies_map],
                outputs=[anomalies_output, report_output, plot_output]
            )
        
        return predict_ui
