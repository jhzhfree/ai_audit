import os
import os
import gradio as gr
from config import Config
from llm import create_llm_chain
from pydantic import BaseModel
import logging

logging.basicConfig(
    level=logging.INFO,  # 日志级别
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 初始化配置
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
    llm_chain = create_llm_chain(config)
    logger.info("LLMChain 初始化成功。")
except Exception as e:
    logger.error(f"LLMChain 初始化失败: {str(e)}")
    raise

# 自定义 Pydantic 数据模型以支持非标准类型
class CustomBaseModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True

# 获取 demo 目录中的项目列表
def get_project_list():
    """
    获取 demo 目录中所有项目名称（文件名去掉扩展名）
    需要同时存在 .flow 和 .cm 文件的项目。
    """
    base_dir = "demo"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    files = os.listdir(base_dir)
    flow_files = {os.path.splitext(f)[0] for f in files if f.endswith(".flow")}
    cm_files = {os.path.splitext(f)[0] for f in files if f.endswith(".cm")}
    return sorted(flow_files & cm_files)

# 从文件夹加载测试数据
def load_test_data(project_name):
    """
    根据项目名称加载 .flow 和 .cm 文件内容
    """
    base_dir = "demo"
    ticket_file = os.path.join(base_dir, f"{project_name}.flow")
    command_file = os.path.join(base_dir, f"{project_name}.cm")
    
    try:
        # 读取工单文件
        with open(ticket_file, "r", encoding="utf-8") as f:
            ticket_content = f.read()
        
        # 读取执行命令文件
        with open(command_file, "r", encoding="utf-8") as f:
            executed_commands = f.read()

        return ticket_content, executed_commands
    except Exception as e:
        return f"无法加载文件: {e}", ""

# Gradio 回调函数
def analyze_commands(ticket_content, executed_commands):
    """
    分析工单内容与实际执行命令的差异，并生成分析报告。
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        # 调试日志：记录输入内容
        logger.debug("开始分析命令...")
        logger.debug(f"输入的工单内容: {ticket_content.strip()}")
        logger.debug(f"输入的实际执行命令: {executed_commands.strip()}")
        
        # 调用 LangChain 进行推理
        result = llm_chain.run({
            "ticket_content": ticket_content.strip(),
            "executed_commands": executed_commands.strip()
        })

        # 调试日志：记录推理结果
        logger.debug("分析完成，生成结果成功。")
        logger.debug(f"生成的分析结果: {result}")
        
        return result
    except Exception as e:
        # 错误日志：记录异常信息
        logger.error(f"分析命令时发生错误: {str(e)}")
        return f"发生错误: {str(e)}"


# Gradio 界面设计
with gr.Blocks() as app:
    gr.Markdown("## 🛠️ 工单审计助手")
    gr.Markdown("使用大模型分析工单内容和实际执行命令的合规性")

    with gr.Tab('工单分析'):
        with gr.Row():
            ticket_input = gr.TextArea(
                label="原始工单内容",
                placeholder="请输入原始工单内容，包括升级背景、步骤和命令...",
                lines=10
            )
            command_input = gr.TextArea(
                label="实际执行命令",
                placeholder="请输入实际执行的命令列表，每行一个命令...",
                lines=10
            )

        output = gr.TextArea(
            label="分析结果",
            placeholder="分析结果将在这里显示...",
            lines=15
        )

        with gr.Row():
            project_selector = gr.Dropdown(
                label="选择项目名称",
                choices=get_project_list(),
                interactive=True
            )
            load_button = gr.Button("加载测试数据")

        analyze_button = gr.Button("开始分析")

        # 加载测试数据按钮逻辑
        def load_and_fill_data(project_name):
            ticket_content, executed_commands = load_test_data(project_name)
            return ticket_content, executed_commands

        load_button.click(
            fn=load_and_fill_data,
            inputs=[project_selector],
            outputs=[ticket_input, command_input]
        )

        # 分析按钮逻辑
        analyze_button.click(
            fn=analyze_commands,
            inputs=[ticket_input, command_input],
            outputs=output
        )
    # 添加系统配置的Tab
    with gr.Tab('系统配置'):
        gr.Markdown("### 配置系统参数")

        model_type_selector = gr.Radio(
            choices=['openai', 'ollama'],
            label='选择模型类型'
        )
        model_name_input = gr.Textbox(label='添加新模型名称')
        add_model_button = gr.Button("添加模型")

        def add_model(model_type, new_model):
            config.add_model(model_type, new_model)
            return f"成功添加模型: {new_model}"

        add_model_output = gr.Textbox()
        add_model_button.click(
            fn=add_model,
            inputs=[model_type_selector, model_name_input],
            outputs=add_model_output
        )

# 启动 Gradio 应用
if __name__ == "__main__":
    app.launch(share=True, server_name="0.0.0.0", server_port=7860)
