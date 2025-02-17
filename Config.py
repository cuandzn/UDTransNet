import os
import torch
import time
## PARAMETERS OF THE MODEL
save_model = True
tensorboard = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
use_cuda = torch.cuda.is_available()

seed = 666
os.environ['PYTHONHASHSEED'] = str(seed)

kfold = 5
cosineLR = True
n_channels = 3
n_labels = 4
epochs = 100
img_size = 512
img_size2 = 512
print_frequency = 1
save_frequency = 5000
vis_frequency = 40

task_name = 'GlaS'
# task_name = 'ISIC18'
# task_name = 'Synapse'

if task_name is 'GlaS':
    learning_rate = 0.0001
    batch_size = 4
    early_stopping_patience = 50
    print_frequency = 30
elif task_name is 'ISIC18':
    learning_rate = 1e-4
    batch_size = 9
    early_stopping_patience = 40
    print_frequency = 30
elif task_name is "Synapse":
    learning_rate = 1e-3
    early_stopping_patience = 40
    batch_size = 24
    n_labels = 3
    n_channels = 1
    print_frequency = 16


# model_name = 'UNet'
# model_name = 'R34_UNet'
model_name = 'UDTransNet'


if task_name is "ISIC18":
    if model_name is "UNet":
        test_session = "Test_session_"
    if model_name is "R34_UNet":
        test_session = "Test_session_"
    if model_name is "UDTransNet":
        test_session = "Test_session_"

elif task_name is "GlaS":
    if model_name is "UNet":
        test_session = "Test_session_"
    if model_name is "R34_UNet":
        test_session = "Test_session_"
    if model_name is "UDTransNet":
        test_session = "Test_session_08.16_09h16"

if task_name is "Synapse":
    if model_name is "UNet":
        test_session = "Test_session_"
    if model_name is "R34_UNet":
        test_session = "Test_session_"
    if model_name is "UDTransNet":
        test_session = "Test_session_"

if task_name == 'Synapse':
    train_dataset = './datasets/Synapse/train_npz/'
    test_dataset = './datasets/Synapse/test_vol_h5/'
else:
    train_dataset = '/root/data/UDtransnet_dataset/train/'
    test_dataset = '/root/data/UDtransnet_dataset/51-60/0select'

session_name       = 'Test_session' + '_' + time.strftime('%m.%d_%Hh%M')
save_path          = task_name +'_kfold/'+ model_name +'/' + session_name + '/'
model_path         = save_path + 'models/'
tensorboard_folder = save_path + 'tensorboard_logs/'
logger_path        = save_path + session_name + ".log"
visualize_path     = save_path + 'visualize_val/'