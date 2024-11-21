import os
import gradio as gr
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import openai
from pydantic import BaseModel

import getpass
import asyncio



def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"请输入您的 {var}")
        
        
_set_if_undefined("OPENAI_API_KEY")

# 配置 OpenAI 的 API 基础地址和密钥
openai.api_base = "http://172.16.3.115:2024/v1"
openai.api_key = os.getenv('OPENAI_API_KEY')

# 自定义 Pydantic 数据模型以支持非标准类型
class CustomBaseModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True

# 初始化 ChatOpenAI 模型
llm = ChatOpenAI(
    model="qwen2.5-14b",
    openai_api_base=openai.api_base,
    openai_api_key=openai.api_key,
    temperature=0.7,
    max_tokens=1500
)

# 提示词模板
prompt_template = PromptTemplate(
    input_variables=["ticket_content", "executed_commands"],
    template="""\
你是一名专业的IT运维审计助手，负责分析工单执行情况。以下是输入内容：
1. 原始工单内容：
{ticket_content}

2. 实际执行的命令：
{executed_commands}

请完成以下任务：
1. 对比工单内容和实际执行命令，分析命令是否完全一致。
2. 如果存在不一致，请明确指出工单中的命令与实际执行命令的差异，以及可能的原因。
3. 检查实际执行的命令是否完全按照工单的步骤顺序执行。
4. 检查是否存在额外的、不合规的命令操作，并详细说明原因及风险。
5. 最后，生成一份总结报告，内容包括：
   - 执行合规性分析（符合/不符合）。
   - 不一致命令及原因分析。
   - 不合规命令的风险说明。
   - 提供明确的审批是否通过的结论。
"""
)

# 定义 LangChain 分析链
llm_chain = LLMChain(llm=llm, prompt=prompt_template)

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
    try:
        # 调用 LangChain 进行推理
        result = llm_chain.run({
            "ticket_content": ticket_content.strip(),
            "executed_commands": executed_commands.strip()
        })
        return result
    except Exception as e:
        # 返回捕获的错误信息
        return f"发生错误: {str(e)}"

# Gradio 界面设计
with gr.Blocks() as app:
    gr.Markdown("## 🛠️ 工单审计助手")
    gr.Markdown("使用大模型分析工单内容和实际执行命令的合规性")

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

# 启动 Gradio 应用
if __name__ == "__main__":
    app.launch(share=True, server_name="0.0.0.0", server_port=7860)
