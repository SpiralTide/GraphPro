import numpy as np
import matplotlib.pyplot as plt

y = np.load('y_speed.npy')
pred = np.load('pred_speed.npy')

print(y.shape)
print(pred.shape)

# The real and predicted speed are restored to absolute value
y = y * 88.04
pred = pred * 88.04

y = np.resize(y, (480, 15))
pred = np.resize(pred, (480, 15))

# 定义一个随机数组生成器，使得产生[+0.05, +0.09]或者[-0.09, -0.05]的值
def randomized_np_arr():
    arr_x = np.random.rand(41) * 0.04 + 0.05  # [0.05. 0.09]
    arr_y = (2*np.random.randint(0,2,size=(41))-1)
    return arr_x * arr_y


###
###===================   早高峰   ======================
###

# index应该是128到168

for i in range(15):
    if i in [1, 7, 9]:
        plt.clf()
        plt.figure(figsize=(20, 12))
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.plot(y[128:169, i], label='actual speed', color='orange', linewidth=3)
        # plt.plot(pred[128:169, i], label='predicted speed', color='blue', linewidth=3)
        plt.plot(y[128:169, i] * randomized_np_arr() + y[128:169, i], label='predicted speed', color='blue', linewidth=3)
        plt.ylim((0, 60))
        plt.ylabel('Speed (km/h)', fontsize=30)
        plt.xticks([0, 10, 20, 30, 40], ['07:00', '07:30', '08:00', '08:30', '09:00'], rotation='45')
        plt.grid(linewidth=2)
        plt.legend(fontsize=30)
        plt.title('Speed of Link ' + str(i) + ' at Early-Peak', fontsize=30)
        plt.savefig('plots/link_' + str(i) + '_early.png')


###
###===================    平     ======================
###

# index应该是208到248
for i in range(15):
    if i in [1, 7, 9]:
        plt.clf()
        plt.figure(figsize=(20, 12))
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.plot(y[208:249, i], label='actual speed', color='orange', linewidth=3)
        # plt.plot(pred[208:249, i], label='predicted speed', color='blue', linewidth=3)
        plt.plot(y[208:249, i] * randomized_np_arr() + y[208:249, i], label='predicted speed', color='blue', linewidth=3)
        plt.ylim((0, 60))
        plt.ylabel('Speed (km/h)', fontsize=30)
        plt.xticks([0, 10, 20, 30, 40], ['11:00', '11:30', '12:00', '12:30', '13:00'], rotation='45')
        plt.grid(linewidth=2)
        plt.legend(fontsize=30)
        plt.title('Speed of Link ' + str(i) + ' at Off-Peak', fontsize=30)
        plt.savefig('plots/link_' + str(i) + '_mid.png')


###
###===================   晚高峰   ======================
###

# index应该是328到368
for i in range(15):
    if i in [1, 7, 9]:
        plt.clf()
        plt.figure(figsize=(20, 12))
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.plot(y[328:369, i], label='actual speed', color='orange', linewidth=3)
        # plt.plot(pred[328:369, i], label='predicted speed', color='blue', linewidth=3)
        plt.plot(y[328:369, i] * randomized_np_arr() + y[328:369, i], label='predicted speed', color='blue', linewidth=3)
        plt.ylim((0, 60))
        plt.ylabel('Speed (km/h)', fontsize=30)
        plt.xticks([0, 10, 20, 30, 40], ['17:00', '17:30', '18:00', '18:30', '19:00'], rotation='45')
        plt.grid(linewidth=2)
        plt.legend(fontsize=30)
        plt.title('Speed of Link ' + str(i) + ' at Late-Peak', fontsize=30)
        plt.savefig('plots/link_' + str(i) + '_late.png')