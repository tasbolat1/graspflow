import numpy as np
import torch
from robot_model import PandaRobot, RobotClassifier
from networks import losses
from scipy.spatial.transform import Rotation as R
from robot_model import PandaIKModel, PandaRobot
from pathlib import Path
import time
from torch.utils.tensorboard import SummaryWriter
from networks.utils import save_model
from torch.optim.lr_scheduler import MultiStepLR

# VARIABLES
print_train_freq = 10
print_test_freq = 10
save_freq = 1000
epochs = 100000
device = 0
batch_size = 1024
alpha_t = 0.9

# INITIALIZATION
model_name= int( time.time()*100 )
model_save_path = f'saved_models/IK/{model_name}/'
Path(model_save_path).mkdir(parents=True, exist_ok=True)

writer = SummaryWriter(model_save_path)

lower_limits = torch.FloatTensor([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]).to(device)
upper_limits = torch.FloatTensor([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]).to(device)
l=lower_limits.repeat(batch_size, 1).to(device)
u=upper_limits.repeat(batch_size, 1).to(device)

# RUNNING THE MODEL

ik = PandaIKModel().to(device)

ik.load_state_dict(torch.load("saved_models/IK/167095731284/38000.pt"))

fk = PandaRobot(device=device)
criteria = torch.nn.MSELoss()

optimizer = torch.optim.RMSprop(params=ik.parameters(), lr=1e-6, momentum=0.9)
scheduler = MultiStepLR(optimizer, milestones=[10, 100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000], gamma =0.5)

def train(epoch):

    optimizer.zero_grad()

    # random sample feasible joint configuration
    theta = l + (u-l)*torch.randn([batch_size,7]).to(l.device)

    # solve forward kinematics
    t,q=fk(theta)

    # solve for IK
    theta_pred = ik(t,q)

    # solve for new fk
    t_pred, q_pred = fk(theta_pred)

    # find loss
    # loss = criteria(theta, theta_pred)
    loss_t = criteria(t_pred, t)
    loss_q = losses.quat_chordal_squared_loss(q_pred, q)

    loss = alpha_t*loss_t + (1-alpha_t)*loss_q

    # # print
    # if epoch % print_train_freq == 0:
    #     print(f'Train loss {loss.data}')

    writer.add_scalar('Train loss (translation)', loss_t.data, epoch)
    writer.add_scalar('Train loss (quaternion)', loss_q.data, epoch)
    writer.add_scalar('Train loss', loss.data, epoch)

    loss.backward()

    optimizer.step()


def test(epoch):

    with torch.no_grad():

        # random sample feasible joint configuration
        theta = l + (u-l)*torch.randn([batch_size,7]).to(device)

        # solve forward kinematics
        t,q=fk(theta)

        # solve for IK
        theta_pred = ik(t,q)

        # find loss
        # loss = criteria(theta, theta_pred)
        # print(f'Test loss {loss.data}')

        # solve for new fk
        t_pred, q_pred = fk(theta_pred)

        loss_t = criteria(t_pred, t)
        loss_q = losses.quat_chordal_squared_loss(q_pred, q)

        loss = alpha_t*loss_t + (1-alpha_t)*loss_q

        writer.add_scalar('Test loss (translation)', loss_t.data, epoch)
        writer.add_scalar('Test loss (quaternion)', loss_q.data, epoch)
        writer.add_scalar('Test loss', loss.data, epoch)


if __name__ == "__main__":
    

    for epoch in range(epochs):
        ik.train()
        train(epoch)
        if epoch % print_test_freq == 0:
            ik.eval()
            test(epoch)

        if epoch % save_freq == 0:
            save_model(ik, model_save_path, epoch=epoch)

        # scheduler.step()
        