import json

class Config:
    def __init__(self, config_file='config.json'):
        self.config_file = config_file
        self.load_config()

    def load_config(self):
        """加载配置文件"""
        try:
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)  # 将整个配置存储为一个属性
        except FileNotFoundError:
            print(f"配置文件 {self.config_file} 不存在。")
            self.config = {}
        except json.JSONDecodeError:
            print(f"配置文件 {self.config_file} 格式错误。")
            self.config = {}

        # 加载 LLM 相关配置
        llm_config = self.config.get('llm', {})
        self.llm_model_type = llm_config.get('model_type', 'openai')
        self.openai_model_name = llm_config.get('openai_model_name', 'gpt-4o-mini')
        self.openai_api_url = llm_config.get('openai_api_url', 'http://172.16.3.115:2024/v1')
        self.ollama_model_name = llm_config.get('ollama_model_name', 'llama3.2')
        self.ollama_api_url = llm_config.get('ollama_api_url', 'http://172.16.3.115:21434')

        # 加载模型列表
        self.openai_models = llm_config.get('openai_models', ['gpt-3.5-turbo', 'gpt-4o-mini', 'gpt-4o'])
        self.ollama_models = llm_config.get('ollama_models', ['gemma2:2b', 'llama3.2', 'llama3.1'])

    def save_config(self):
        """将当前配置保存到文件"""
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            print(f"配置文件 {self.config_file} 不存在，创建新的配置文件。")
            config = {}
        except json.JSONDecodeError:
            print(f"配置文件 {self.config_file} 格式错误，无法保存。")
            return

        # 更新配置
        config['llm'] = {
            'model_type': self.llm_model_type,
            'openai_model_name': self.openai_model_name,
            'ollama_model_name': self.ollama_model_name,
            'openai_api_url': self.openai_api_url,
            'ollama_api_url': self.ollama_api_url,
            'openai_models': self.openai_models,
            'ollama_models': self.ollama_models
        }

        # 保存配置文件
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=4)

    def get(self, key, default=None):
        """为配置信息提供一个统一的获取方法"""
        return self.config.get(key, default)

    # 模型管理相关方法
    def get_models(self, model_type):
        """根据模型类型返回模型列表"""
        if model_type == 'openai':
            return self.openai_models
        elif model_type == 'ollama':
            return self.ollama_models
        else:
            return []

    def add_model(self, model_type, new_model):
        """向指定的模型类型列表中添加模型"""
        if model_type == 'openai':
            if new_model not in self.openai_models:
                self.openai_models.append(new_model)
                self.save_config()
                return f"成功添加 OpenAI 模型: {new_model}"
            else:
                return f"模型 {new_model} 已存在于 OpenAI 模型列表中。"
        elif model_type == 'ollama':
            if new_model not in self.ollama_models:
                self.ollama_models.append(new_model)
                self.save_config()
                return f"成功添加 Ollama 模型: {new_model}"
            else:
                return f"模型 {new_model} 已存在于 Ollama 模型列表中。"
        else:
            return f"未知的模型类型: {model_type}"

    def delete_model(self, model_type, model_name):
        """从指定的模型类型列表中删除模型"""
        if model_type == 'openai' and model_name in self.openai_models:
            self.openai_models.remove(model_name)
            self.save_config()
            return f"成功删除 OpenAI 模型: {model_name}"
        elif model_type == 'ollama' and model_name in self.ollama_models:
            self.ollama_models.remove(model_name)
            self.save_config()
            return f"成功删除 Ollama 模型: {model_name}"
        else:
            return f"模型 {model_name} 不存在于 {model_type} 模型列表中。"

