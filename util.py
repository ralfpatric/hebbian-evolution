import os
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter


def get_writers(name):
    tensorboard_dir = os.environ.get('TENSORBOARD_DIR') or '/tmp/tensorboard'
    # revision = os.environ.get("REVISION") or "%s" % datetime.now()

    train_writer = SummaryWriter(os.getcwd() + '\\tensorboard\\%s\\train\\' % (name))
    #test_writer = SummaryWriter(os.getcwd() + '\\tensorboard\\%s\\test\\' % (name))

    return train_writer #, test_writer
