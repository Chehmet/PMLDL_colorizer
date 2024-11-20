import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
PLOTS_DIR = os.path.join(PROJECT_ROOT, 'plots')

def get_root_path() -> str:
    return PROJECT_ROOT

def get_model_path(model_type: str, name: str = None) -> str:
    if name is None:
        return os.path.join(MODELS_DIR, model_type)
    return os.path.join(MODELS_DIR, model_type, name)

def get_plots_path(model_type: str) -> str:
    return os.path.join(PLOTS_DIR, model_type)

def get_main_config_path() -> str:
    return os.path.join(PROJECT_ROOT, 'config', 'vars_config.json')
