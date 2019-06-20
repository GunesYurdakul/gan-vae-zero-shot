from __future__ import division
import os
import time
import tensorflow as tf
import argparse
from Datasets import *

# Real samples
import seaborn as sns

np.random.seed(1234)
tf.set_random_seed(1234)

"""parsing and configuration"""

def parse_args():
    desc = "Tensorflow implementation of GAN"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--nhidden', type=int, default=64, help='number of hidden neurons')
    parser.add_argument('--nlayers', type=int, default=4, help='number of hidden layers')
    parser.add_argument('--missing_mixt', type=int, default=0, help='label of the missing mixtures')
    parser.add_argument('--label', action='store_true', default=False, help='if data labeled')
    parser.add_argument('--notebook', action='store_true', default=False, help='if you are running in python notebook')
    parser.add_argument('--fig_name', type=str, default='loss-fig', help='file name of loss plot')
    parser.add_argument('--lambda_d', type=float, default=0.5, help='weight value of classification loss of discriminator')
    parser.add_argument('--lambda_g', type=float, default=0.5, help='weight value of creativity loss of generator')
    parser.add_argument('--niters', type=int, default=3001, help='number of iterations')
    parser.add_argument('--batch_size', type=int, default=400, help='batch size')
    parser.add_argument('--lrg', type=float, default=3e-3, help='lr for G')
    parser.add_argument('--lrd', type=float, default=9e-3, help='lr for D')
    parser.add_argument('--dataset', type=str, default='8Gaussians', help='dataset to use: 8Gaussians | 25Gaussians | swissroll | mnist')
    parser.add_argument('--scale', type=float, default=2., help='data scaling')
    parser.add_argument('--loss', type=str, default='gan', help='gan | wgan')
    parser.add_argument('--optim', type=str, default='SGD', help='optimizer to use')
    parser.add_argument('--minibatch_discriminate', action='store_true', default=False,help='minibatch_discriminate flag')

    return parser.parse_args()


def plot_loss(prefix, g_loss_list, d_loss_list, d_loss_fake_list, d_loss_real_list):
    f, ax = plt.subplots(1)
    g_loss_array = np.array(g_loss_list)
    d_loss_array = np.array(d_loss_list)
    d_loss_fake_array = np.array(d_loss_fake_list)
    d_loss_real_array = np.array(d_loss_real_list)
    if len(g_loss_list):
        ax.plot(g_loss_array[:, 0], g_loss_array[:, 1], color="k", label='g_loss')
    ax.plot(d_loss_array[:, 0], d_loss_array[:, 1], color="r", label='d_loss')
    ax.plot(d_loss_fake_array[:, 0], d_loss_fake_array[:, 1], color="g", label='d_loss_fake_array')
    ax.plot(d_loss_real_array[:, 0], d_loss_real_array[:, 1], color="b", label='d_loss_real_array')
    plt.title('GAN Metrics (2D Gaussians)')
    plt.xlabel('Step')
    plt.ylabel('Metrics')
    plt.legend()
    plt.savefig(prefix + 'metrics.png')


def draw_density(samps, scale, fname):
    fig = plt.figure(frameon=False, dpi= 160)
    fig.set_size_inches(5, 5)
    ax = fig.add_subplot(1, 1, 1)

    sns.kdeplot(samps[:, 0], samps[:, 1], shade=True, cmap='Greys', gridsize=200, n_levels=100)

    ax.set_xlim((-scale, scale))
    ax.set_ylim((-scale, scale))
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.spines['bottom'].set_color('0.5')
    ax.spines['top'].set_color('0.5')
    ax.spines['right'].set_color('0.5')
    ax.spines['left'].set_color('0.5')
    plt.savefig('figs/density/critic_' + fname, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

eps = 1e-20
def inception_score(X):
    kl = X * ((X+eps).log()-(X.mean(0)+eps).log().expand_as(X))
    score = np.exp(kl.sum(1).mean())

    return score
def mode_score(X, Y):
    kl1 = X * ((X+eps).log()-(X.mean(0)+eps).log().expand_as(X))
    kl2 = X.mean(0) * ((X.mean(0)+eps).log()-(Y.mean(0)+eps).log())
    score = np.exp(kl1.sum(1).mean() - kl2.sum())

    return score
class STYLEGAN(object):
    model_name = "STYLEGAN"  # name for checkpoint

    def __init__(self, sess, args, data, noise):
        self.sess = sess
        self.nhidden = args.nhidden
        self.nlayers = args.nlayers
        self.niters = args.niters
        self.batch_size = args.batch_size
        self.labeled = args.label
        self.lambda_g= args.lambda_g
        self.lambda_d = args.lambda_d
        self.notebook = args.notebook
        self.z_dim = 2
        self.label_dim=8
        self.x_dim = 2
        self.fig_name = args.fig_name
        self.missing_mixt = args.missing_mixt
        self.lrg = args.lrg
        self.lrd = args.lrd
        self.data = data
        self.noise = noise
        self.scale = args.scale
        self.minibatch_discriminate = args.minibatch_discriminate

    def minibatch(self, x, num_kernels=5, kernel_dim=3):
        net = tf.layers.dense(inputs=x, units=num_kernels * kernel_dim, name='minibatch')
        activation = tf.reshape(net, (-1, num_kernels, kernel_dim))
        diffs = tf.expand_dims(activation, 3) - \
                tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
        abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
        minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), self.z_dim)
        return tf.concat([x, minibatch_features], 1)

    def discriminator(self, x, is_training=True, reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse):
            net = tf.layers.dense(inputs=x[:,0:2], units=self.nhidden, activation=None, name='d_fc1')
            # net = tf.contrib.layers.batch_norm(net, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=is_training)
            net = tf.nn.relu(net, name='d_rl1')
            for i in range(self.nlayers - 2):
                net = tf.layers.dense(inputs=net, units=self.nhidden, activation=None, name='d_fc' + str(i + 2))
                # net = tf.contrib.layers.batch_norm(net, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=is_training)
                net = tf.nn.relu(net, name='d_rl' + str(i + 2))
            if self.minibatch_discriminate:
                net = self.minibatch(net)
            out_logit = tf.layers.dense(inputs=net, units=1, name='d_fc' + str(self.nlayers))
            out_class_logit = tf.layers.dense(inputs=net, units=8, name='d_fc_c' + str(self.nlayers))

            out = tf.nn.sigmoid(out_logit)
            return out, out_logit, out_class_logit

    def generator(self, z, is_training=True, reuse=False):
        with tf.variable_scope("generator", reuse=reuse):
            net = tf.layers.dense(inputs=z, units=self.nhidden, activation=None, name='g_fc1')
            # net = tf.contrib.layers.batch_norm(net, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=is_training)
            net = tf.nn.relu(net, name='g_rl1')
            for i in range(self.nlayers - 2):
                net = tf.layers.dense(inputs=net, units=self.nhidden, activation=None, name='g_fc' + str(i + 2))
                # net = tf.contrib.layers.batch_norm(net, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=is_training)
                net = tf.nn.relu(net, name='g_rl' + str(i + 2))
            out = tf.layers.dense(inputs=net, units=self.z_dim, activation=None, name='g_fc' + str(self.nlayers))
            return out

    def build_model(self):
        """ Graph Input """
        # images
        self.inputs = tf.placeholder(tf.float32, [None, self.x_dim], name='real_placeholder')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim+self.label_dim], name='z_placeholder')

        if self.labeled:
            self.labels = tf.placeholder(tf.float32, [None, self.label_dim], name='label_placeholder')
            self.fake_labels = self.z[:,2:10]
        # noises
        """ Loss Function """

        # output of D for real images
        D_real, D_real_logits, D_class_real_logits = self.discriminator(self.inputs, is_training=True, reuse=False)
        # output of D for fake images
        self.generates = self.generator(self.z[:,0:2], is_training=True, reuse=False)
        D_fake, D_fake_logits, D_class_fake_logits = self.discriminator(self.generates, is_training=True, reuse=True)
        if self.labeled:
            self.d_classification_loss_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=D_class_real_logits, labels=self.labels))
            print(self.inputs[:, 2:10].shape)
            self.d_classification_loss_fake = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=D_class_fake_logits,labels=self.fake_labels))

            uniform_one_hot_indices = np.random.uniform(0, 8, self.batch_size)
            tf.one_hot(uniform_one_hot_indices, 8)
            self.g_loss_creativity = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=D_class_fake_logits,labels= tf.one_hot(uniform_one_hot_indices, 8)))

        else:
            self.lambda_d=0
            self.lambda_g=0
        # get loss for discriminator
        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake)))
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake)))
        self.d_loss = self.d_loss_real + self.d_loss_fake


        if self.labeled:
            self.d_loss+= self.lambda_d*self.d_classification_loss_fake + self.lambda_d * self.d_classification_loss_real
            self.g_loss+= self.lambda_g*self.g_loss_creativity

        # get loss for generator
        """ Training """
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]

        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_optim = tf.train.AdamOptimizer(self.lrd, beta1=0.5).minimize(self.d_loss, var_list=d_vars)
            self.g_optim = tf.train.AdamOptimizer(self.lrg, beta1=0.5).minimize(self.g_loss, var_list=g_vars)

        """" Testing """
        # for test
        self.fake_samples = self.generator(self.z[:,0:2], is_training=False, reuse=True)
        self.fake_sigmoid, self.fake_logit, self.class_fake_logit  = self.discriminator(self.fake_samples, is_training=False, reuse=True)
        self.fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_logit, labels=tf.ones_like(self.fake_sigmoid)))
        self.fake_saliency = tf.gradients(self.fake_loss, self.fake_samples)[0]

        """ Summary """
        d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)
        d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)

        # final summary operations
        self.g_sum = tf.summary.merge([d_loss_fake_sum, g_loss_sum])
        self.d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum])

        # saver to save model
        self.saver = tf.train.Saver()

    def train(self, prefix, teacher=None, sess_teacher=None):
        # initialize all variables
        tf.global_variables_initializer().run()

        # summary writer
        self.writer = tf.summary.FileWriter(prefix)
        g_loss_list, d_loss_list, d_loss_fake_list, d_loss_real_list = [], [], [], []

        # graph inputs for visualize training results
        self.sample_z = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))
        n_rows = int(np.ceil(self.niters / 500))

        if self.notebook:
            fig = plt.figure(figsize=(20, n_rows*4), dpi= 160)
            fig.subplots_adjust(hspace=0.4, wspace=0.4)
        else:
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))

        plot_idx=0
        # loop for epoch
        start_time = time.time()
        fake=[]
        real=[]
        for it in range(self.niters):

            noise_batch = self.noise.next_batch(self.batch_size)
            # update D network

            real_batch, labels = self.data.next_batch(self.batch_size, missing_mixt=self.missing_mixt)

            _, d_loss, d_loss_fake, d_loss_real, summary_str = self.sess.run(
                [self.d_optim, self.d_loss, self.d_loss_fake, self.d_loss_real, self.d_sum],
                feed_dict={self.inputs: real_batch, self.z: noise_batch, self.labels: labels})

            self.writer.add_summary(summary_str, it)
            d_loss_list.append((it, d_loss))
            d_loss_fake_list.append((it, d_loss_fake))
            d_loss_real_list.append((it, d_loss_real))

            _, g_loss, summary_str = self.sess.run([self.g_optim, self.g_loss, self.g_sum],
                                                   feed_dict={self.z: noise_batch})

            self.writer.add_summary(summary_str, it)
            g_loss_list.append((it, g_loss))

            fake_batch, fake_saliency, fake_logit = self.sess.run(
                [self.fake_samples, self.fake_saliency, self.fake_logit], feed_dict={self.z: noise_batch})

            # display training status
            if it % 100 == 0:
                print("Iter: %d, d_loss: %.8f, g_loss: %.8f" % (it, d_loss, g_loss))
                plot_idx+=1
                if self.notebook:
                    ax = fig.add_subplot(n_rows, 5, plot_idx)
                else:
                    ax.clear()

                ax.scatter(real_batch[:, 0], real_batch[:, 1], s=2, c='k')
                ax.scatter(fake_batch[:, 0], fake_batch[:, 1], s=2, c='g', marker='o')
                ax.set_xlim(-self.scale,self.scale)
                ax.set_ylim(-self.scale, self.scale)
                ax.set_title("It #{:d}: g = {:.4f}, d = {:.4f}".format(it, g_loss, d_loss),fontsize=10)
                plt.savefig(prefix + 'fig_%05d.png' % it, bbox_inches='tight')

                if not self.notebook:

                    plt.draw()
                    plt.pause(1e-6)
                    #plt.show()
                fake=np.asarray(fake_batch)
                real=np.asarray(real_batch)
                draw_density(real,2,'real'+str(it))
                draw_density(fake,2,'generated'+str(it))
        plt.savefig('figs/'+self.fig_name, bbox_inches='tight')
        #plt.show()
        # self.saver.save(self.sess, self.dir_model + 'gan')
        plot_loss('figs/'+self.fig_name+'-', g_loss_list, d_loss_list, d_loss_fake_list, d_loss_real_list)

    def visualize_results(self, epoch):
        z_sample = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))
        samples = self.sess.run(self.fake_samples, feed_dict={self.z: z_sample})


"""main"""


def main(args):
    # open session
    tf.reset_default_graph()

    with tf.Session() as sess:
        # declare instance for GAN
        data = ToyDataset(distr=args.dataset, scale=args.scale, labeled=args.label)
        noise = NoiseDataset(labeled=args.label)
        gan = STYLEGAN(sess, args, data, noise)

        # build graph
        gan.build_model()

        # train
        prefix = 'figs/tf_default_lrd_' + str(args.lrd) + '_lrg_' + str(args.lrg) + '/'
        if not os.path.exists(prefix):
            os.makedirs(prefix)

        gan.train(prefix)
        print(" [*] Training finished!")

if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)
    # parse arguments
    args = parse_args()
    print(args)
    main(args)
