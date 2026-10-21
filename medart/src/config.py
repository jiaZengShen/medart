from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
#CSM_ROOT = Path(r"I:\icdInformations\3claim\CSM-for-fetal-HC-measurement")
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

DEFAULT_TEST_JSON = Path(r"./measure_test.json")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT /  f"outputs_{timestamp}"
#DEFAULT_MODEL_PATH = Path(r"C:\Users\jzs\icd\model\google\medgemma-1.5-4b-it")
DEFAULT_LOCAL_DATA_ROOT = Path(r"I:\icdInformations\1数据集")
#DEFAULT_SERVER_DATA_ROOT = Path(r"C:\Users\jzs\icd")
DEFAULT_SERVER_DATA_ROOT = Path(r"I:\icdInformations\1数据集")
#DEFAULT_CSM_WEIGHTS = CSM_ROOT / "models" / "test_model.pth"
DEFAULT_DEVICE_MODE = "auto"
DEFAULT_QUANT_MODE = "4bit"
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TEMPERATURE = 0.1
DEFAULT_TOP_P = 0.9

# MedSAM segmentation server
MEDSAM_SERVER_URL = "http://192.168.1.100:6060"

# BiomedParse segmentation server
BIOMEDPARSE_SERVER_URL = "https://u319687-b1gp-28a7dea2.bjb2.seetacloud.com:8443"

# Doubao (豆包) Global Search API
DOUBAO_SEARCH_URL = "https://open.feedcoopapi.com/search_api/global_search"
DOUBAO_SEARCH_KYE = "vPqk98ylbFniZno4GaG36ZkSN75PsY5B"