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
    logger.info("创建 Ollama LLM 实例...")
    
    # 确认配置项
    logger.info(f"Ollama 模型名称: {config.ollama_model_name}")
    logger.info(f"Ollama API 基础 URL: {config.ollama_api_url}")
    logger.info(f"Ollama 温度: {config.get('ollama_temperature', 0.7)}")
    logger.info(f"Ollama 最大 Token 数量: {config.get('ollama_max_tokens', 1500)}")
    
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


def create_logs_llm_chain(config):
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
    logger.info("创建提示词模板...")
    prompt_template = PromptTemplate(
        input_variables=["start_date", "end_date", "stats"],
        template="""\
# 异常行为分析报告生成

## 角色（Role）
你是一名专业的安全分析顾问，擅长从系统异常行为数据中提取关键信息，进行深入分析，并提出切实可行的改进建议。你的目标是帮助用户快速识别风险，提高系统的安全性和稳定性。

---

## 任务（Task）
根据以下输入数据，生成一份异常行为分析报告，具体任务包括：
1. **数据概览**：总结时间范围和总体异常统计情况。
2. **异常类型分析**：逐项分析每种异常行为的含义、典型场景和潜在风险，并提供针对性的改进建议。
3. **总结与总体建议**：概述主要分析结果，并提出整体优化方案以提高系统的安全性和稳定性。

---

## 输入数据
- **时间范围**：
  - 开始时间：{start_date}
  - 结束时间：{end_date}
- **异常统计数据**：
{stats}

---

## 输出格式（Format）
请严格按照以下格式组织输出内容：

### 异常行为分析报告

#### 1. 数据概览
- **时间范围**：
  - 开始时间：{start_date}
  - 结束时间：{end_date}
- **异常统计**：
{stats}

#### 2. 异常类型分析与建议
描述不同异常行为的分析和建议

#### 3. 总结与总体建议
- **总结**：概述主要异常行为的风险及影响。
- **总体建议**：提供提高系统安全性和稳定性的具体措施。

---

## 举例（Example）

### 异常行为分析报告

#### 1. 数据概览
- **时间范围**：
  - 开始时间：2024-01-01
  - 结束时间：2024-01-31
- **异常统计**：
  - 异常类型1：5 次
  - 异常类型2：3 次

#### 2. 异常类型分析与建议

##### 异常类型：异常登录行为
- **分析**：该行为指短时间内发生多次失败的登录尝试。典型场景包括密码暴力破解或用户多次尝试错误密码，可能导致系统账户被锁定。
- **改进建议**：建议启用账户锁定策略，增加多因素认证（MFA），并对频繁尝试的 IP 实施限制。

##### 异常类型：IP 地址异常
- **分析**：此行为指来自可疑或黑名单 IP 的登录尝试。典型场景包括来自非常规区域的大量请求，可能为恶意行为。
- **改进建议**：建议加强 IP 黑名单管理，启用实时监控异常请求的功能，并自动拦截可疑行为。

#### 3. 总结与总体建议
- **总结**：在统计时间范围内，共发现两类主要异常行为：异常登录行为和 IP 地址异常。这些行为可能导致账户安全和系统稳定性问题。
- **总体建议**：建议完善系统安全策略，包括启用多因素认证、实时监控功能以及 IP 地址拦截机制，以有效降低风险。

---

## 注意事项
1. **分析深度**：对每种异常行为提供清晰的含义、风险说明和改进建议。
2. **逻辑清晰**：确保输出内容组织有条理，格式符合要求。
3. **动态适配**：异常行为类型数量和内容会根据输入数据变化，请灵活处理。

请根据上述要求生成报告。
"""
    )

    logger.info("返回创建的 LLMChain logs 实例...")
    # 返回创建的 LLMChain 实例
    return LLMChain(llm=llm, prompt=prompt_template)
