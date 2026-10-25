import logging
from pathlib import Path

handlers = [logging.StreamHandler()]

# Also log to logs/app.log, unless the folder is read-only (some hosts)
try:
    Path('logs').mkdir(exist_ok=True)
    handlers.append(logging.FileHandler(Path('logs/app.log')))
except OSError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=handlers
)
logger = logging.getLogger(__name__)
