# __init__.py

import importlib

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Lista com os nomes dos seus arquivos (sem o .py)
MODULES = [
    "postprocessings",
    "audio_tools",
    "image_tools",
    "exports",
    "tools",
    "controls",
    "novelai",
    "shiro_test",
    "shiro_test2",
    "shiro_test3",
]

WEB_DIRECTORY = "./web"

# Carrega todos os módulos da lista e avisa no terminal se faltar algum
for module_name in MODULES:
    try:
        mod = importlib.import_module(f".{module_name}", package=__name__)
        
        if hasattr(mod, "NODE_CLASS_MAPPINGS"):
            NODE_CLASS_MAPPINGS.update(mod.NODE_CLASS_MAPPINGS)
        if hasattr(mod, "NODE_DISPLAY_NAME_MAPPINGS"):
            NODE_DISPLAY_NAME_MAPPINGS.update(mod.NODE_DISPLAY_NAME_MAPPINGS)
            
    except ImportError as e:
        print(f"\033[93m[Shiro Tools] Aviso: Erro ao carregar '{module_name}': {e}\033[0m")

# Tratamento especial para o ambiente de testes (ignora silenciosamente se não existir)
try:
    from . import shiro_test
    
    # Suporta tanto se o shiro_test usar NODE_CLASS_MAPPINGS quanto TEST_CLASS_MAPPINGS
    class_mappings = getattr(shiro_test, "TEST_CLASS_MAPPINGS", getattr(shiro_test, "NODE_CLASS_MAPPINGS", {}))
    name_mappings = getattr(shiro_test, "TEST_DISPLAY_NAME_MAPPINGS", getattr(shiro_test, "NODE_DISPLAY_NAME_MAPPINGS", {}))
    
    NODE_CLASS_MAPPINGS.update(class_mappings)
    NODE_DISPLAY_NAME_MAPPINGS.update(name_mappings)
except ImportError:
    pass

# Aplica o Monkey-Patch no VideoHelperSuite para corrigir o erro de float/int
try:
    from . import monkey_patch
except Exception as e:
    print(f"\033[93m[Shiro Tools] Aviso: Não foi possível aplicar o Monkey-Patch: {e}\033[0m")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]