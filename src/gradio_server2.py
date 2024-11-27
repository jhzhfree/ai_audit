import gradio as gr
from train_model_ui import TrainModelUI
from predict_model_ui import PredictModelUI
import logging

logging.basicConfig(
    level=logging.info,  # 全局日志级别
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]  # 输出到控制台
)

# import debugpy
# # 启动调试监听，默认 5678 端口
# debugpy.listen(('0.0.0.0', 5678))
# print("等待调试器连接...")
# debugpy.wait_for_client()  # 等待调试器连接

# 强制设置根记录器及其子记录器的级别为 DEBUG
logging.getLogger().setLevel(logging.INFO)

def main():

    # 定义 Gradio 界面
    with gr.Blocks() as demo:
        with gr.Tabs():
            with gr.TabItem("模型训练"):
                train_ui = TrainModelUI()
                train_ui.render()
            with gr.TabItem("模型预测"):
                predict_ui = PredictModelUI()
                predict_ui.render()

    # 启动服务
    demo.launch(share=True, server_name="0.0.0.0", server_port=7862)

if __name__ == "__main__":
    main()
