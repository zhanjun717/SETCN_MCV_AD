import torch.optim as optim
import sys
sys.path.append("../../")
from tcn import TCN
import argparse
from utils.biuld_dataset import *
from utils.util import *
from utils.pytool import EarlyStopping
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import time
import matplotlib.pyplot as plt
from sklearn import preprocessing
import torch
from torch.utils.data import DataLoader
# writer就相当于一个日志，保存你要做图的所有信息。第二句就是在你的项目目录下建立一个文件夹log，存放画图用的文件。刚开始的时候是空的
from tensorboardX import SummaryWriter
writer = SummaryWriter('log')  # 建立一个保存数据用的东西

######Globle variable#####
steps = 0

# ----------------------------------#
#         config                    #
# ----------------------------------#
parser = argparse.ArgumentParser(description='Sequence Modeling - (Permuted) Sequential SATCN')
parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='batch size (default: 64)')
parser.add_argument('--cuda', action='store_false', help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout applied to layers (default: 0.05)')
parser.add_argument('--clip', type=float, default=-1, help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--epochs', type=int, default=100, help='upper epoch limit (default: 20)')
parser.add_argument('--ksize', type=int, default=7, help='kernel size (default: 7)')
parser.add_argument('--levels', type=int, default=8, help='# of levels (default: 8)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate (default: 2e-3)')
parser.add_argument('--optim', type=str, default='Adam', help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=10, help='number of hidden units per layer (default: 25)')
parser.add_argument('--nclass', type=int, default=1, help='size of the output (default: 1)')
parser.add_argument('--seed', type=int, default=1111, help='random seed (default: 1111)')
parser.add_argument('--permute', action='store_true', help='use permuted MNIST (default: false)')
parser.add_argument('--stft', type=bool, default=True, help='Embedding STFT (default: True)')
parser.add_argument('--train_start_time', type=str, default='2018-07-01', help='Start time of train(default: 2018-07-01)')
parser.add_argument('--train_end_time', type=str, default='2018-07-01', help='End time of train DS(default: 2018-07-09)')
parser.add_argument('--test_start_time', type=str, default='2018-07-10', help='Start time of test(default: 2018-07-01)')
parser.add_argument('--test_end_time', type=str, default='2018-07-10', help='End time of test DS(default: 2018-07-10)')
args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

print(args)

# ----------------------------------#
#         load dataset              #
# ----------------------------------#
dataset = pd.read_parquet("./data/HD04/TB020_2018_train_normal.parquet",
                          columns=['x方向振动值', '机舱气象站风速', '轮毂转速', '叶片1角度', '5秒偏航对风平均值', '变频器发电机侧功率'])
dataset_train = dataset[args.train_start_time : args.train_end_time]
dataset_test = dataset[args.test_start_time : args.test_end_time]

lable_1hz = np.array(dataset_train["x方向振动值"]).reshape(-1, 1)

pd.DataFrame(dataset_train, columns=dataset_train.columns.values).to_parquet('./data/HD04/TB020_train_normal_fornormalizer.parquet')
dataset_normalizer = preprocessing.StandardScaler().fit(dataset_train)
dataset_train = dataset_normalizer.transform(dataset_train)
dataset_test = dataset_normalizer.transform(dataset_test)

pd.DataFrame(lable_1hz, columns=['label']).to_parquet('./data/HD04/TB020_train_label_fornormalizer.parquet')
y_train_normalizer = preprocessing.StandardScaler().fit(lable_1hz)
lable_normal_1hz = y_train_normalizer.transform(lable_1hz)

x_train, y_train, x_val, y_val = data_prepare(dataset_train, args.stft)
n_fearture = x_train.shape[2]

x_test, y_test = multivariate_data(dataset_test, dataset_test[:, 0],
                                   0, None, past_history,
                                   future_target, STEP,
                                   single_step=True,
                                   stft_flag=args.stft)

# ----------------------------------#
#         creating the dataset      #
# ----------------------------------#
trainset = TensorDataset(torch.tensor(x_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.float))
valset = TensorDataset(torch.tensor(x_val, dtype=torch.float), torch.tensor(y_val, dtype=torch.float))
testset = TensorDataset(torch.tensor(x_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.float))

# creating the dataloader
train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
size_data, seq_length, nb_feature = x_train.shape

val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

# ----------------------------------#
#               model               #
# ----------------------------------#
permute = torch.Tensor(np.random.permutation(60).astype(np.float64)).long()
channel_sizes = [args.nhid] * args.levels

model = TCN(n_fearture, args.nclass, channel_sizes, kernel_size=args.ksize, dropout=args.dropout)
if args.cuda:
    model.cuda()
    permute = permute.cuda()

lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)

def train(ep):
    global steps
    train_loss = 0
    model.train()
    criterion = torch.nn.MSELoss()

    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data = data.permute(0, 2, 1)
        if args.permute:
            data = data[:, :, permute]
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        train_loss += loss.item()
        steps += seq_length

        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            print('\r', '\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tSteps: {}'.format(
                ep, batch_idx * args.batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), steps), sep='', end='', flush=True)

    train_loss /= batch_idx
    end_time = time.time()
    print('Train set: Average loss: {:.6f}\ttime cost: {:.6f}'.format(train_loss,start_time-end_time))
    return train_loss

def val():
    model.eval()
    test_loss = 0
    criterion = torch.nn.MSELoss()
    batch_idx = 0

    with torch.no_grad():
        for data, target in val_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data = data.permute(0, 2, 1)
            # data = data.view(-1, input_channels, seq_length)
            if args.permute:
                data = data[:, :, permute]
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss
            batch_idx += 1

        test_loss /= batch_idx
        print('Val set: Average loss: {:.6f}'.format(test_loss))
        return test_loss

def predict(data_loader, model):
    model.eval()
    pridict_loss = 0
    criterion = torch.nn.MSELoss()
    batch_idx = 0
    pridict_output = []
    target_input = []
    total_loss = []

    with torch.no_grad():
        for data, target in data_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data = data.permute(0, 2, 1)
            # data = data.view(-1, input_channels, seq_length)
            if args.permute:
                data = data[:, :, permute]
            output = model(data)
            # pridict_output.append(output)

            pridict_output.append(output)
            target_input.append(target)

            loss = criterion(output, target)
            total_loss.append(loss.unsqueeze(-1))
            pridict_loss += loss
            batch_idx += 1

        pridict_loss /= batch_idx
        print('Pridict set: Average loss: {:.6f}'.format(pridict_loss))
        return pridict_loss, total_loss, pridict_output, target_input

def realtime_predict(data,model):
    model.eval()
    with torch.no_grad():
        if args.cuda:
            data = data.cuda()
        data = data.permute(0, 2, 1)
        # data = data.view(-1, input_channels, seq_length)
        if args.permute:
            data = data[:, :, permute]
        output = model(data)
    return output

if __name__ == "__main__":
    # EarlyStopping
    early_stopping = EarlyStopping(patience=10, verbose=True, delta=0.000001, store_path='./models')
    train_loss_list = []
    val_loss_list = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train(epoch)
        train_loss_list.append(train_loss)
        val_loss = val()
        val_loss_list.append(val_loss)

        if epoch % 10 == 0:
            lr /= 5
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print('Early stopping')
            break

    mdhms = time.strftime('%d%H%M', time.localtime(time.time()))
    plt.figure(0, figsize=(10, 6))
    plt.title('Train and Val loss')
    plt.plot(train_loss_list, color='blue', label='train_loss', linewidth=2)
    plt.plot(val_loss_list, color='yellow', label='test_loss', linewidth=2)
    plt.savefig("./figure" + '/loss_' + mdhms + '.png')

    pridict_loss, total_loss, pridict_output, target_input = predict(test_loader, model)
    test_out_val = torch.cat(pridict_output, dim=0).cpu().detach().numpy()
    test_Y_val = torch.cat(target_input, dim=0).cpu().detach().numpy()

    # Metrics
    print('Test dataset result:')
    # 输出在训练集上的R^2
    metrics_calculate(test_Y_val, test_out_val)
    print("在测试集上的R^2:", r2_score(test_Y_val, test_out_val))

    out_val_raw = y_train_normalizer.inverse_transform(test_out_val.reshape(-1,1))
    Y_val_raw = y_train_normalizer.inverse_transform(test_Y_val.reshape(-1, 1))

    print('Test raw dataset result:')
    # 输出在训练集上的R^2
    metrics_calculate(out_val_raw, Y_val_raw)
    print("在测试集上的R^2:", r2_score(out_val_raw, Y_val_raw))

    res = out_val_raw - Y_val_raw

    plt.figure(1, figsize=(10, 6))
    plt.subplot(311)
    plt.title('Normalized dataset')
    plt.plot(pd.DataFrame(test_Y_val), color='blue', label='lable', linewidth=2)
    plt.plot(pd.DataFrame(test_out_val), color='yellow', label='predict', linewidth=2)
    plt.legend()

    plt.subplot(312)
    plt.title('Raw dataset')
    plt.plot(pd.DataFrame(Y_val_raw), color='blue', label='lable', linewidth=2)
    plt.plot(pd.DataFrame(out_val_raw), color='yellow', label='predict', linewidth=2)
    plt.legend()

    plt.subplot(313)
    plt.title('Residual')
    plt.plot(res, color='blue', label='res', linewidth=2)
    plt.plot(pd.DataFrame(abs(res)).rolling(30).mean(), color='yellow', label='res_mean', linewidth=2)
    plt.legend()

    plt.savefig("./figure" + '/val_res_' + mdhms + '.png')
    plt.show()
