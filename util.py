import os
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter


def get_writers(name):
    tensorboard_dir = os.environ.get('TENSORBOARD_DIR') or '/tmp/tensorboard'
    # revision = os.environ.get("REVISION") or "%s" % datetime.now()
    message = os.environ.get('MESSAGE')

    train_writer = SummaryWriter(os.getcwd() + '/%s/train/%s' % (name,message))
    test_writer = SummaryWriter(os.getcwd() + '/%s/test/%s' % (name,message))
    return train_writer, test_writer
