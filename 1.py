from tensorboardX import SummaryWriter
import time

writer = SummaryWriter(log_dir='./mv-hrnet_test')

for i in range(10):
    writer.add_scalar('loss', 10 - i, i)
    time.sleep(1)

writer.close()
