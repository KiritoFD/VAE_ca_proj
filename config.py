import json
from pathlib import Path
from typing import Dict, Any


class Config:
    """配置管理类"""
    
    def __init__(self, config_path: str = "config.json"):
        """
        加载配置文件
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self._config = json.load(f)
    
    @property
    def model(self) -> Dict[str, Any]:
        """模型配置"""
        return self._config.get('model', {})
    
    @property
    def training(self) -> Dict[str, Any]:
        """训练配置"""
        return self._config.get('training', {})
    
    @property
    def data(self) -> Dict[str, Any]:
        """数据配置"""
        return self._config.get('data', {})
    
    @property
    def checkpoint(self) -> Dict[str, Any]:
        """检查点配置"""
        return self._config.get('checkpoint', {})
    
    @property
    def inference(self) -> Dict[str, Any]:
        """推理配置"""
        return self._config.get('inference', {})
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项"""
        return self._config.get(key, default)
    
    def save(self, config_path: str = None):
        """保存配置到文件"""
        save_path = Path(config_path) if config_path else self.config_path
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self._config, f, indent=2, ensure_ascii=False)
    
    def __repr__(self):
        return f"Config(config_path={self.config_path})"
    
    def print_config(self):
        """打印配置信息"""
        print("\n" + "="*50)
        print("Configuration")
        print("="*50)
        for section, params in self._config.items():
            print(f"\n[{section.upper()}]")
            if isinstance(params, dict):
                for key, value in params.items():
                    print(f"  {key}: {value}")
            elif isinstance(params, list):
                print(f"  (list with {len(params)} items)")
                for idx, item in enumerate(params):
                    print(f"    [{idx}]: {item}")
            else:
                print(f"  {params}")
        print("="*50 + "\n")
