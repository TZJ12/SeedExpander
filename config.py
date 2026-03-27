"""
统一配置管理模块
集中管理API密钥、模型参数、重试策略等配置
"""
import os
from typing import Dict, Any, Optional


class Config:
    """配置管理类"""
    
    # ==================== 主模型配置 ====================
    # 主要LLM服务配置
    MODEL_URL = "http://58.214.239.10:18080/app-2512170147-llm/v1/chat/completions"
    MODEL_KEY = "sk-390feb24-daee-11f0-a581-0242ac150003"
    MODEL_NAME = "Qwen3-30B"
    MAX_TOKENS = 12000
    TEMPERATURE = 0.7
    STREAM = False

    # ==================== 越狱模型配置（DeepSeek Ark） ====================
    JAILBREAK_MODEL_URL = 'http://58.214.239.10:18080/app-2512170147-llm/v1/chat/completions'
    JAILBREAK_MODEL_KEY = 'sk-390feb24-daee-11f0-a581-0242ac150003'
    JAILBREAK_MODEL_NAME = 'Qwen3-30B'
    JAILBREAK_MAX_TOKENS = 10000
    JAILBREAK_TEMPERATURE = 0.7
    JAILBREAK_STREAM = False
    
    # ==================== 重试策略配置 ====================
    # 超稳定配置 - 专为角色生成等复杂任务优化
    MAX_RETRIES = 3  # 最大重试次数，确保高成功率
    BASE_DELAY = 2.0  # 基础延迟，给服务器充分恢复时间
    MAX_DELAY = 90.0  # 最大延迟上限，应对严重网络问题
    BACKOFF_FACTOR = 1.5  # 温和的退避因子，避免延迟过快增长
    REQUEST_TIMEOUT = 60  # 请求超时时间（秒）- 专为复杂角色生成任务设计
    
    # ==================== 密码模块配置 ====================
    # cipher_module 专用配置
    CIPHER_MODEL_URL = "http://58.214.239.10:18080/app-2512170147-llm/v1/chat/completions"
    CIPHER_MODEL_KEY = "sk-390feb24-daee-11f0-a581-0242ac150003"
    CIPHER_MODEL_NAME = "Qwen3-30B"
    CIPHER_MAX_TOKENS = 12000
    CIPHER_TEMPERATURE = 0.7
    CIPHER_STREAM = False

    # ==================== 数据库配置 (MySQL) ====================
    DB_HOST = "127.0.0.1"
    DB_PORT = 3306
    DB_USER = "root"
    DB_PASSWORD = "Dabby@2024"
    DB_NAME = "seed_expander"
 
    # ==================== 翻译服务配置 ====================
    # 百度翻译API
    BAIDU_TRANSLATE_URL = 'https://fanyi-api.baidu.com/api/trans/vip/translate'
    BAIDU_APP_ID = os.getenv('BAIDU_APP_ID', '')
    BAIDU_SECRET_KEY = os.getenv('BAIDU_SECRET_KEY', '')
    
    # 老张翻译API
    LAOZHANG_TRANSLATE_URL = "https://api.laozhang.ai/v1/chat/completions"
    LAOZHANG_API_KEY = os.getenv('LAOZHANG_API_KEY', '')
    
    # ==================== 评估服务配置 ====================
    # 任务评估API基础URL (根地址)
    EVAL_URL = "http://127.0.0.1:8088"
    
    # 评估服务认证 Token
    EVAL_AUTH_TOKEN = "a_test_token"

    # 被测模型 (Victim) 配置 - 用于自动填充评估请求
    # 默认复用主模型配置，自动去除 /chat/completions 后缀
    EVAL_TARGET_BASE_URL = MODEL_URL.replace("/chat/completions", "")
    EVAL_TARGET_KEY = MODEL_KEY
    EVAL_TARGET_NAME = MODEL_NAME
    
    # 裁判模型 (Judge) 配置
    EVAL_JUDGE_BASE_URL = MODEL_URL.replace("/chat/completions", "")
    EVAL_JUDGE_KEY = MODEL_KEY
    EVAL_JUDGE_NAME = MODEL_NAME

    # ==================== 防御测试服务配置 ====================
    DEFENSE_TEST_URL = "http://10.30.3.141:8010/scan/injection"
    DEFENSE_TEST_TOKEN = "1tdhblkfcdhx2awjjzasztglsgdhzkglcsdbf"


    # ==================== Label Studio 服务配置 ====================   
    LABEL_STUDIO_URL = "http://localhost:8080"
    LABEL_STUDIO_TOKEN = "726a155618014b08d3694e7dc736e5492272fd9e"

    # ==================== 通用配置 ====================
    # 随机种子控制
    DEFAULT_SEED = None  # None表示使用随机种子
    
    # 输出格式配置
    OUTPUT_ENCODING = 'utf-8'
    
    # 日志配置
    LOG_LEVEL = 'INFO'
    ENABLE_DEBUG_LOG = False
    
    @classmethod
    def get_model_config(cls) -> Dict[str, Any]:
        """获取主模型配置"""
        return {
            'url': cls.MODEL_URL,
            'key': cls.MODEL_KEY,
            'name': cls.MODEL_NAME,
            'max_tokens': cls.MAX_TOKENS,
            'temperature': cls.TEMPERATURE,
            'stream': cls.STREAM
        }

    @classmethod
    def get_jailbreak_model_config(cls) -> Dict[str, Any]:
        """获取越狱模型配置（DeepSeek Ark）"""
        return {
            'url': cls.JAILBREAK_MODEL_URL,
            'key': cls.JAILBREAK_MODEL_KEY,
            'name': cls.JAILBREAK_MODEL_NAME,
            'max_tokens': cls.JAILBREAK_MAX_TOKENS,
            'temperature': cls.JAILBREAK_TEMPERATURE,
            'stream': cls.JAILBREAK_STREAM
        }
    
    @classmethod
    def get_retry_config(cls) -> Dict[str, Any]:
        """获取重试策略配置"""
        return {
            'max_retries': cls.MAX_RETRIES,
            'base_delay': cls.BASE_DELAY,
            'max_delay': cls.MAX_DELAY,
            'backoff_factor': cls.BACKOFF_FACTOR,
            'timeout': cls.REQUEST_TIMEOUT
        }
    
    @classmethod
    def get_cipher_model_config(cls) -> Dict[str, Any]:
        """获取密码模块模型配置"""
        return {
            'model_name': cls.CIPHER_MODEL_NAME,
            'model_url': cls.CIPHER_MODEL_URL,
            'model_key': cls.CIPHER_MODEL_KEY,
            'max_tokens': cls.CIPHER_MAX_TOKENS,
            'temperature': cls.CIPHER_TEMPERATURE,
            'stream': cls.CIPHER_STREAM
        }
    
    @classmethod
    def get_baidu_translate_config(cls) -> Dict[str, Any]:
        """获取百度翻译配置"""
        return {
            'url': cls.BAIDU_TRANSLATE_URL,
            'app_id': cls.BAIDU_APP_ID,
            'secret_key': cls.BAIDU_SECRET_KEY
        }
    
    @classmethod
    def get_laozhang_translate_config(cls) -> Dict[str, Any]:
        """获取老张翻译配置"""
        return {
            'url': cls.LAOZHANG_TRANSLATE_URL,
            'api_key': cls.LAOZHANG_API_KEY
        }
    
    @classmethod
    def update_config(cls, **kwargs) -> None:
        """动态更新配置项"""
        for key, value in kwargs.items():
            if hasattr(cls, key.upper()):
                setattr(cls, key.upper(), value)
            else:
                print(f"警告: 配置项 {key} 不存在")
    
    @classmethod
    def load_from_env(cls) -> None:
        """从环境变量加载配置"""
        # 主模型配置
        cls.MODEL_URL = os.getenv('MODEL_URL', cls.MODEL_URL)
        cls.MODEL_KEY = os.getenv('MODEL_KEY', cls.MODEL_KEY)
        cls.MODEL_NAME = os.getenv('MODEL_NAME', cls.MODEL_NAME)
        
        # 密码模块配置
        cls.CIPHER_MODEL_URL = os.getenv('CIPHER_MODEL_URL', cls.CIPHER_MODEL_URL)
        cls.CIPHER_MODEL_KEY = os.getenv('CIPHER_MODEL_KEY', cls.CIPHER_MODEL_KEY)
        cls.CIPHER_MODEL_NAME = os.getenv('CIPHER_MODEL_NAME', cls.CIPHER_MODEL_NAME)
        
        # 重试配置
        cls.MAX_RETRIES = int(os.getenv('MAX_RETRIES', cls.MAX_RETRIES))
        cls.BASE_DELAY = float(os.getenv('BASE_DELAY', cls.BASE_DELAY))
        cls.MAX_DELAY = float(os.getenv('MAX_DELAY', cls.MAX_DELAY))
        
        # 翻译服务配置
        cls.BAIDU_APP_ID = os.getenv('BAIDU_APP_ID', cls.BAIDU_APP_ID)
        cls.BAIDU_SECRET_KEY = os.getenv('BAIDU_SECRET_KEY', cls.BAIDU_SECRET_KEY)
        cls.LAOZHANG_API_KEY = os.getenv('LAOZHANG_API_KEY', cls.LAOZHANG_API_KEY)
        
        # 评估服务配置
        cls.EVAL_URL = os.getenv('EVAL_URL', cls.EVAL_URL)
        cls.EVAL_AUTH_TOKEN = os.getenv('EVAL_AUTH_TOKEN', cls.EVAL_AUTH_TOKEN)
        
        cls.EVAL_TARGET_BASE_URL = os.getenv('EVAL_TARGET_BASE_URL', cls.EVAL_TARGET_BASE_URL)
        cls.EVAL_TARGET_KEY = os.getenv('EVAL_TARGET_KEY', cls.EVAL_TARGET_KEY)
        cls.EVAL_TARGET_NAME = os.getenv('EVAL_TARGET_NAME', cls.EVAL_TARGET_NAME)
        
        cls.EVAL_JUDGE_BASE_URL = os.getenv('EVAL_JUDGE_BASE_URL', cls.EVAL_JUDGE_BASE_URL)
        cls.EVAL_JUDGE_KEY = os.getenv('EVAL_JUDGE_KEY', cls.EVAL_JUDGE_KEY)
        cls.EVAL_JUDGE_NAME = os.getenv('EVAL_JUDGE_NAME', cls.EVAL_JUDGE_NAME)

        # 其他配置
        cls.DEFAULT_SEED = os.getenv('DEFAULT_SEED', cls.DEFAULT_SEED)
        if cls.DEFAULT_SEED and cls.DEFAULT_SEED.isdigit():
            cls.DEFAULT_SEED = int(cls.DEFAULT_SEED)


# 自动加载环境变量配置
Config.load_from_env()

# 导出常用配置的快捷访问
MODEL_CONFIG = Config.get_model_config()
RETRY_CONFIG = Config.get_retry_config()
CIPHER_CONFIG = Config.get_cipher_model_config()
