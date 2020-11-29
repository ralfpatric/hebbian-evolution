import os
from datetime import datetime

import pickle
import os.path
from google.cloud import storage
from google.cloud.storage import Bucket

from torch.utils.tensorboard import SummaryWriter

storage_client = storage.Client.from_service_account_json('credentials.json')
tensorboard_dir = "gs://modern-ai-store/tensorboard"

def get_writers(name):
    #tensorboard_dir = os.environ.get('TENSORBOARD_DIR') or '/tmp/tensorboard'
    # revision = os.environ.get("REVISION") or "%s" % datetime.now()

    train_writer = SummaryWriter(tensorboard_dir + "/train/%s/" % (name),10)
    #test_writer = SummaryWriter(os.getcwd() + '\\tensorboard\\%s\\test\\' % (name))

    return train_writer #, test_write



def upload_files(path,fname):
    bucket: Bucket = storage_client.bucket("modern-ai-store")
    bucket.blob(path).upload_from_filename(fname)