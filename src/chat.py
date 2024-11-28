from config import Config
from llm import create_logs_llm_chain
import logging
import asyncio

logging.basicConfig(
    level=logging.INFO,  # 日志级别
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class ChatbotManager:
    
    def __init__(self):
        self.config =  Config()
        self.llm_chain = self.initialize_sales_bot()
        
    def initialize_sales_bot(self):
        try:
            logger.debug("正在初始化 LLMChain...")
            llmchain = create_logs_llm_chain(self.config)
            return llmchain
        except Exception as e:
            logger.error(f"LLMChain 初始化失败: {str(e)}")
            raise
    def create_report(self, start, end, status):
        """
        立即返回用户消息，更新页面。
        """
        logging.info(f"接收到用户问题: {status}")
        result = self.llm_chain.run({
            "start_date": start,
            "end_date": end,
            "stats": status
        })
        
        logging.info(f"llm_chain result: {result}")
        return result  # 立即返回用户消息
    
    
async def main():
    chatmanager = ChatbotManager()
    report_output = await chatmanager.create_report("2021", "2034", "stats_summary")
    
    print(report_output)

if __name__ == "__main__":
    asyncio.run(main())  # 使用 asyncio 运行异步 main 函数

    
    
    