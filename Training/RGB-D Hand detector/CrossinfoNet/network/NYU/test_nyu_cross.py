import sys
sys.path.append('../../')#add root directory
# import matplotlib
# matplotlib.use('Agg')
from data.importers import NYUImporter
from data.transformations import transformPoints2D
from util.handdetector import HandDetector
import numpy as np
from util.preprocess import augmentCrop,norm_dm
import matplotlib.pyplot  as plt
import tensorflow as tf



rng=np.random.RandomState(23455)
from netlib.basemodel import basenet2

visual=True

train_root='/home/dumyy/data/nyu/dataset/'
di_1 = NYUImporter(train_root, cacheDir='../../cache/NYU/',refineNet=None,allJoints=False)
Seq_train = di_1.loadSequence('test', rng=rng, shuffle=False, docom=False,cube=(250,250,250))

test_num=len(Seq_train.data)
print test_num
cubes = np.asarray([d.cube for d in Seq_train.data], 'float32')
coms = np.asarray([d.com for d in Seq_train.data], 'float32')
Ms = np.asarray([d.T for d in Seq_train.data], dtype='float32')
gt3Dcrops = np.asarray([d.gt3Dcrop for d in Seq_train.data], dtype='float32')
imgs = np.asarray([d.dpt.copy() for d in Seq_train.data], 'float32')
test_data=np.ones_like(imgs)
test_label=np.ones_like(gt3Dcrops)


for i in range(test_num):

    test_data[i]=norm_dm(imgs[i],coms[i],cubes[i])
    test_label[i]=gt3Dcrops[i]/(cubes[i][0]/2.)
    #print cubes[i]
    # plt.imshow(test_data[i],cmap='gray')
    # print i
    # plt.pause(0.0001)
    # plt.cla()

test_data=np.expand_dims(test_data,3)
test_label=np.reshape(test_label,(-1,42))
# for i in range(test_num):
#     print cubes[i]
inputs=tf.placeholder(dtype=tf.float32,shape=(None,96,96,1))
label=tf.placeholder(dtype=tf.float32,shape=(None,42))


batch_size=128

import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layers

fn=layers.l2_regularizer(1e-5)
fn0=tf.no_regularizer

with slim.arg_scope([slim.conv2d, slim.fully_connected],
                    weights_regularizer=fn,
                    biases_regularizer=fn0,
                    normalizer_fn=slim.batch_norm):
    with slim.arg_scope([slim.batch_norm],
                        is_training=False,
                        updates_collections=None,
                        decay=0.9,
                        center=True,
                        scale=True,
                        epsilon=1e-5):
        pred_comb_ht, pred_comb_hand, pred_hand, pred_ht=basenet2(inputs,kp=1,is_training=False)


pred_out=pred_hand


import time

pred_norm=[]
saver=tf.train.Saver(max_to_keep=5)
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    saver.restore(sess,'../../model/crossInfoNet_NYU.ckpt')
    loopv = test_num // batch_size
    other = test_data[loopv * batch_size:]
    a=time.time()
    for i in xrange(loopv + 1):
        if i < loopv:
            start = i * batch_size
            end = (i + 1) * batch_size
            feed_dict = {inputs: test_data[start:end]}
        else:
            feed_dict = {inputs: other}
        [pred_]=sess.run([pred_out],feed_dict=feed_dict)
        pred_norm.append(pred_)
    b=time.time()
    print b-a

norm_hands=np.concatenate(pred_norm,0).reshape(-1,14,3)
pred_hands=norm_hands*np.tile(np.expand_dims(cubes/2.,1),(1,14,1))+np.tile(np.expand_dims(coms,1),(1,14,1))
gt_hands=test_label.reshape(-1,14,3)*np.tile(np.expand_dims(cubes/2.,1),(1,14,1))+np.tile(np.expand_dims(coms,1),(1,14,1))

def getJointMeanError(jointID, gt, joints):
    return np.nanmean(np.sqrt(np.square(gt[:, jointID, :] - joints[:, jointID, :]).sum(axis=1)))
def getMeanError(gt, joints):
    return np.nanmean(np.nanmean(np.sqrt(np.square(gt - joints).sum(axis=2)), axis=1))
meane= getMeanError(gt_hands,pred_hands)
sub1= getMeanError(gt_hands[0:2440],pred_hands[0:2440])
sub2= getMeanError(gt_hands[2440:],pred_hands[2440:])
print "meane is {}".format(meane)
print "sub1 is {}".format(sub1)
print "sub2 is {}".format(sub2)


print [getJointMeanError(j,gt_hands,pred_hands) for j in range(14)]

f = open('../../results/end_nyu.txt', 'a+')
for i in range(pred_hands.shape[0]):
    uvds=di_1.joints3DToImg(pred_hands[i])
    uvds=np.reshape(uvds,(1,42))
    for j in range(42):
        f.write(str(round(uvds[0,j],4)))
        f.write(' ')
    f.write('\n')

f.close()
plt.get_cmap()
hand_edges=[[0,1],[2,3],[4,5],[6,7],[8,9],[9,10],[13,10],[13,1],[13,3],[13,5],[13,7],[13,11],[13,12]]
if visual:

    for i in range(0,8252,10):


        plt.imshow(np.squeeze(test_data[i]), cmap='gray')
        jtIp = transformPoints2D(di_1.joints3DToImg(pred_hands[i]), Ms[i])
        plt.scatter(jtIp[:, 0], jtIp[:, 1], c='r')

        jtIt = transformPoints2D(di_1.joints3DToImg(gt_hands[i]), Ms[i])
        plt.scatter(jtIt[:, 0], jtIt[:, 1], c='b')

        for edge in hand_edges:
            plt.plot(jtIp[:, 0][edge], jtIp[:, 1][edge], c='r')
            plt.plot(jtIt[:, 0][edge], jtIt[:, 1][edge], c='b')
        plt.pause(0.1)
        plt.text(0, 0, str(i))

        plt.pause(0.01)
        #plt.savefig("../image/NYU/img_{}.png".format(i))
        plt.cla()









