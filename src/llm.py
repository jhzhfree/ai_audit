from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
import openai
from config import Config
import getpass
import logging

# 初始化日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# 初始化配置
config = Config()

def _get_env_variable(var: str):
    """获取环境变量，如果未设置则从用户输入获取"""
    value = os.environ.get(var)
    if not value:
        logger.debug(f"环境变量 {var} 未设置，要求用户输入...")
        value = getpass.getpass(f"请输入您的 {var}: ")
        os.environ[var] = value
    else:
        logger.debug(f"环境变量 {var} 已设置，值为: {value[:4]}...")  # 部分显示，保护隐私
    return value

def create_openai_llm(config):
    """创建 OpenAI 的 LLM 实例"""
    logger.debug("创建 OpenAI LLM 实例...")
    _get_env_variable("OPENAI_API_KEY")
    openai.api_base = config.openai_api_url
    openai.api_key = os.getenv('OPENAI_API_KEY')
    
    logger.info(f"OpenAI API 基础 URL: {openai.api_base}")
    logger.info(f"OpenAI API Key: {openai.api_key[:4]}...")  # 部分显示 API Key
    
    # 确认配置项
    logger.debug(f"OpenAI 模型名称: {config.openai_model_name}")
    logger.debug(f"OpenAI 温度: {config.get('openai_temperature', 0.7)}")
    logger.debug(f"OpenAI 最大 Token 数量: {config.get('openai_max_tokens', 1500)}")
    
    return ChatOpenAI(
        model=config.openai_model_name,
        openai_api_base=openai.api_base,
        openai_api_key=openai.api_key,
        temperature=config.get('openai_temperature', 0.7),
        max_tokens=config.get('openai_max_tokens', 1500)
    )

def create_ollama_llm(config):
    """创建 Ollama 的 LLM 实例"""
    logger.debug("创建 Ollama LLM 实例...")
    
    # 确认配置项
    logger.debug(f"Ollama 模型名称: {config.ollama_model_name}")
    logger.debug(f"Ollama API 基础 URL: {config.ollama_api_url}")
    logger.debug(f"Ollama 温度: {config.get('ollama_temperature', 0.7)}")
    logger.debug(f"Ollama 最大 Token 数量: {config.get('ollama_max_tokens', 1500)}")
    
    return ChatOllama(
        model=config.ollama_model_name,
        temperature=config.get('ollama_temperature', 0.7),
        base_url=config.ollama_api_url
    )

def create_llm_chain(config):
    """根据配置创建 LLMChain 实例"""
    logger.debug("根据配置创建 LLMChain 实例...")
    
    # 打印配置项
    logger.info(f"使用的 LLM 模型类型: {config.llm_model_type}")
    if config.llm_model_type == 'openai':
        logger.info("选择了 OpenAI 模型")
        llm = create_openai_llm(config)
    elif config.llm_model_type == 'ollama':
        logger.info("选择了 Ollama 模型")
        llm = create_ollama_llm(config)
    else:
        logger.error(f"未知的模型类型: {config.llm_model_type}")
        raise ValueError(f"未知的模型类型: {config.llm_model_type}")
    
    # 打印提示词模板
    logger.debug("创建提示词模板...")
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

    logger.debug("返回创建的 LLMChain 实例...")
    # 返回创建的 LLMChain 实例
    return LLMChain(llm=llm, prompt=prompt_template)
