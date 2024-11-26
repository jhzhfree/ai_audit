import gradio as gr
from train_data import ModelTrainer
from rule import AnomalyRules
import logging
import json

logging.basicConfig(
    level=logging.DEBUG,  # 设置为 DEBUG 级别以捕获调试信息
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]  # 输出到控制台
)


class TrainModelUI:
    def __init__(self):
        """
        初始化 UI 界面
        """
        print("TrainModelUI initialized.")

    def handle_training(self, uploaded_file, rules_input):
        """
        训练按钮的处理逻辑：调用 ModelTrainer 执行训练
        """
        try:
            # 创建 ModelTrainer 实例并调用训练逻辑
            print("beg train......")
            model_trainer = ModelTrainer()
            anomalies_file, stats_summary, plot_file, model_file, type_to_id_map = model_trainer.train(uploaded_file, rules_input)
            print("end train......")
            # 返回结果
            return anomalies_file, stats_summary, plot_file, model_file, json.dumps(type_to_id_map, indent=4, ensure_ascii=False)
        
        except Exception as e:
            # 捕获异常并返回 5 个值
            logging.error(f"训练过程中发生错误：{str(e)}", exc_info=True)
            return None, f"训练失败：{str(e)}", None, None, None


    def render(self):
        """
        渲染训练界面
        """
        # 加载默认规则配置
        anomaly_rules = AnomalyRules()
        json_rule = anomaly_rules.load_rules_from_yaml()

        with gr.Blocks(css="""
            /* 修改 Gradio 默认背景和样式 */
            .gr-box { 
                background-color: #f5f5f5 !important; /* 设置灰色背景 */
                border-radius: 8px; /* 设置圆角 */
                border: 1px solid #dcdcdc; /* 添加边框 */
                padding: 15px; /* 增加内边距 */
                box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1); /* 添加阴影效果 */
            }
            .gr-container {
                background-color: transparent !important; /* 确保整体背景一致 */
            }
            /* 对 Group 的子区域进行调整 */
            .gr-group {
                background-color: #ffffff !important;
                border-radius: 10px;
                padding: 10px;
                border: 1px solid #cccccc;
            }
        """) as train_ui:
            # 标题部分
            gr.Markdown("# 模型训练界面\n分步骤完成数据收集、规则配置、训练数据集查看和模型生成。")

            # **训练基础模块**
            with gr.Group():
                gr.Markdown("## 训练基础\n上传数据文件和配置规则")
                with gr.Row():
                    uploaded_file = gr.File(label="上传数据集文件（CSV 格式）", file_types=[".csv"])
                    rules_input = gr.Json(label="规则配置（JSON 格式）", value=json_rule)

            # **训练数据集模块**
            with gr.Group():
                gr.Markdown("## 训练数据集\n查看训练数据并下载结果文件")
                with gr.Row():
                    dataset_preview = gr.Textbox(label="数据集内容预览", interactive=False, lines=10)
                    plot_output = gr.Image(label="训练数据集图例")
                with gr.Row():
                    anomalies_output = gr.File(label="下载训练结果记录文件（CSV）")

            # **模型生成模块**
            with gr.Group():
                gr.Markdown("## 模型生成\n生成并下载模型，查看训练统计结果")
                with gr.Row():
                    model_file = gr.File(label="下载生成的模型文件")
                    model_result = gr.Textbox(label="异常映射表", interactive=False, lines=10)

            # 训练按钮
            train_button = gr.Button("开始训练")

            # 绑定按钮点击事件
            train_button.click(
                fn=self.handle_training,
                inputs=[uploaded_file, rules_input],
                outputs=[
                    anomalies_output,  # 文件输出
                    dataset_preview,   # 文本框显示的摘要信息
                    plot_output,       # 图片显示的图表
                    model_file,        # 文件输出
                    model_result       # 文本框显示异常类型映射
                ]
            )


        return train_ui

