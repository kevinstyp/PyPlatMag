
import logging
import os

logger = logging.getLogger(__name__)


def create_mastercdf(master_cdf_path):
    command = []
    skeletoncdf = os.path.join(os.environ["CDF_BIN"], "skeletoncdf")
    if skeletoncdf is None:
        raise ValueError("CDF_BIN environment variable not set.")
    command.append(skeletoncdf)
    command.append(master_cdf_path[:-4] + ".skt")
    command.append("-cdf")
    command.append(master_cdf_path)
    logger.info(f"Executing command: {' '.join(command)}")

    os.system(' '.join(command))
