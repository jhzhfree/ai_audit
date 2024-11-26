import gradio as gr
from train_model_ui import TrainModelUI
from predict_model_ui import PredictModelUI

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
