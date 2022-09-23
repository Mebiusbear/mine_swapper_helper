import torch
from torch.autograd import Variable
import logging

import src.setting_file as setting_file
import src.dataset_file as dataset_file
import src.net_file as net_file


# 数据准备部分
label_name = setting_file.label_name
label_dict = setting_file.label_dict
len_label_name = setting_file.len_label_name

# 模型部分
Batch_Net = net_file.Batch_Net
in_dim = setting_file.in_dim
n_hiddle_1 = setting_file.n_hiddle_1
n_hiddle_2 = setting_file.n_hiddle_2
n_hiddle_3 = setting_file.n_hiddle_3

# 训练部分
batch_size = setting_file.batch_size
epoch = setting_file.epoch
learning_rate = setting_file.learning_rate
model_name = setting_file.model_name

# 日志部分
make_log = setting_file.make_log


def train():

    make_log()
    logging.info("h1_%d_h2_%d_h3_%d_e_%d"%(n_hiddle_1,n_hiddle_2,n_hiddle_3,epoch))

    train_dataset = dataset_file.train_dataset(setting_file.DATASET_ROOT_DIR)

    criterion = torch.nn.CrossEntropyLoss()
    model = Batch_Net(in_dim,n_hiddle_1,n_hiddle_2,n_hiddle_3,out_dim=len_label_name)
    opitimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for i in range(epoch):
        loss_total = 0
        count = 0
        predict_total = 0
        for img, label in train_dataset:
            batch_len = len(img)
            count += batch_len
            img = img.view(batch_len, -1)
            img = Variable(img)
            label = Variable(label)
            # forward
            out = model(img)
            loss = criterion(out, label)

            out_index = torch.max(out.data,1)[1]
            predict_total += (out_index == label).sum()

            # backward
            opitimizer.zero_grad()
            loss.backward()
            opitimizer.step()
            loss_total += loss
            
        log_mes_1 = "\nepoch %d : "%i
        log_mes_2 = "loss : " + str(float(loss_total/count))
        log_mes_3 = "predict : " + str(float(predict_total/count))
        log_mes = log_mes_1 + "\n" + log_mes_2 + "\n" + log_mes_3 + "\n"
        logging.info(log_mes)

    torch.save(model.state_dict(),model_name)

if __name__ == "__main__":
    train()
