import numpy as np

Z_DIM = 1
G_HIDDEN = 10
X_DIM = 10
D_HIDDEN = 10
step_size_G = 0.01
step_size_D = 0.01
ITER_NUM = 50000
GRADIENT_CLIP = 0.2
WEIGHT_CLIP = 0.25

# real sine samples
def get_samples(random=True):
    if random:
        x0 = np.random.uniform(0,1)
        freq = np.random.uniform(1.2, 1.5)
        mult =  np.random.uniform(0.5, 0.8)
    else:
        x0 = 0
        freq = 0.2
        mult = 1
    signal = [mult * np.sin(x0+freq*i) for i in range(X_DIM)]
    return np.array(signal)

#define activation functions
def ReLU(x):
    return np.maximum(x,0.)

def dReLU(x):
    return ReLU(x)

def LeakyReLU(x, k=0.2):
    return np.where(x>0, x, x*k)

def dLeakyReLU(x, k=0.2):
    return np.where(x>0, 1, x*k)

def Tanh(x):
    return np.tanh(x)
def dTanh(x):
    return 1 - np.tanh(x)**2

def Sigmoid(x):
    return 1. /(1. +np.exp(-x))
def dSigmoid(x):
    return Sigmoid(x)*(1-Sigmoid(x))

#initilaize layer parameters
def weight_initializers(in_channels, out_channels):
    scale = np.sqrt(2. / (in_channels + out_channels))
    return np.random.uniform(-scale, scale, (in_channels, out_channels))

# loss function binary_cross_entropy
class LossFunc(object):
    def __init__(self):
        self.logit = None
        self.label = None
    def forward(self, logit, label):
        if logit[0,0] < 1e-7:
            logit[0,0] = 1e-7
        if 1. - logit[0,0] < 1e-7:
            logit[0,0] = 1. - 1e-7
        self.logit = logit
        self.label = label
        out = -label * np.log(logit) + (1. - label) * np.log(1. - logit)
        return out
    def backward(self):
        return (1- self.label)/(1-self.logit) - self.label/self.logit

# define generator network
class Generator(object):
    def __init__(self):
        self.z = None
        self.w1 = weight_initializers(Z_DIM, G_HIDDEN)
        self.b1 = weight_initializers(1, G_HIDDEN)
        self.x1 = None
        self.w2 = weight_initializers(G_HIDDEN, G_HIDDEN)
        self.b2 = weight_initializers(1, G_HIDDEN)
        self.x2 = None
        self.w3 = weight_initializers(G_HIDDEN, X_DIM)
        self.b3 = weight_initializers(1, X_DIM)
        self.x3 = None
        self.x = None

    def forward(self, inputs):
        self.z = inputs.reshape(1, Z_DIM)
        self.x1 = np.matmul(self.z, self.w1) + self.b1
        self.x1 = ReLU(self.x1)
        self.x2 = np.matmul(self.x1, self.w2) + self.b2
        self.x2 = ReLU(self.x2)
        self.x3 = np.matmul(self.x2, self.w3) + self.b3
        self.x = Tanh(self.x3)
        return self.x
    # calculate derivatives
    def backward(self, outputs):
        # derivatives with respect to output
        delta = outputs
        delta *= dTanh(self.x)

        # derivatives with respect to w3, b3
        d_w3 = np.matmul(np.transpose(self.x2), delta)
        d_b3 = delta.copy()

        # derivatives with respect to x2
        delta = np.matmul(delta, np.transpose(self.w3))

        # update w3
        if(np.linalg.norm(d_w3) > GRADIENT_CLIP):
            d_w3 = GRADIENT_CLIP / np.linalg.norm(d_w3) * d_w3
        self.w3 -= step_size_G * d_w3
        self.w3 = np.maximum(-WEIGHT_CLIP, np.minimum(WEIGHT_CLIP, self.w3))
        #update b3
        self.b3 -= step_size_G * d_b3
        self.b3 = np.maximum(-WEIGHT_CLIP, np.minimum(WEIGHT_CLIP, self.b3))
        delta *=  dReLU(self.x2)

        # Update the parameters in the second layer and pass the gradients to the first layer:
        # derivatives w.r.t w2
        d_w2 = np.matmul(np.transpose(self.x1), delta)
        d_b2 = delta.copy()

        #derivatives w.r.t x1
        delta = np.matmul(delta, np.transpose(self.w2))
        if(np.linalg.norm(d_w2) > GRADIENT_CLIP):
            d_w2 = GRADIENT_CLIP / np.linalg.norm(d_w2) * d_w2
        self.w2 -= step_size_G * d_w2
        self.w2 = np.maximum(-WEIGHT_CLIP, np.minimum(WEIGHT_CLIP, self.w2))

        # update b2
        self.b2 -= step_size_G * d_b2
        self.b2 = np.maximum(-WEIGHT_CLIP, np.minimum(WEIGHT_CLIP, self.b2))
        delta *= dReLU(self.x1)

        # Update the parameters in the first layer:
        # Derivative with respect to w1
        d_w1 = np.matmul(np.transpose(self.z), delta)
        # Derivative with respect to b1
        d_b1 = delta.copy()
        # No need to calculate derivative with respect to z
        # Update w1
        if (np.linalg.norm(d_w1) > GRADIENT_CLIP):
            d_w1 = GRADIENT_CLIP / np.linalg.norm(d_w1) * d_w1
        self.w1 -= step_size_G * d_w1
        self.w1 = np.maximum(-WEIGHT_CLIP, np.minimum(WEIGHT_CLIP, self.w1))
        # Update b1
        self.b1 -= step_size_G * d_b1
        self.b1 = np.maximum(-WEIGHT_CLIP, np.minimum(WEIGHT_CLIP, self.b1))

class Discriminator(object):
    def __init__(self):
        self.x = None
        self.w1 = weight_initializers(X_DIM, D_HIDDEN)
        self.b1 = weight_initializers(1, D_HIDDEN)
        self.y1 = None
        self.w2 = weight_initializers(D_HIDDEN, D_HIDDEN)
        self.b2 = weight_initializers(1, D_HIDDEN)
        self.y2 = None
        self.w3 = weight_initializers(D_HIDDEN, 1)
        self.b3 = weight_initializers(1, 1)
        self.y3 = None
        self.y = None

    def forward(self, inputs):
        self.x = inputs.reshape(1, X_DIM)
        self.y1 = np.matmul(self.x, self.w1) + self.b1
        self.y1 = LeakyReLU(self.y1)
        self.y2 = np.matmul(self.y1, self.w2) + self.b2
        self.y2 = LeakyReLU(self.y2)
        self.y3 = np.matmul(self.y2, self.w3) + self.b3
        self.y = Sigmoid(self.y3)
        return self.y

    def backward(self, outputs, apply_grads=True):
        # Derivative with respect to output
        delta = outputs
        delta *= dSigmoid(self.y)
        # Derivative with respect to w3
        d_w3 = np.matmul(np.transpose(self.y2), delta)
        # Derivative with respect to b3
        d_b3 = delta.copy()
        # Derivative with respect to y2
        delta = np.matmul(delta, np.transpose(self.w3))
        if apply_grads:
            # Update w3
            if np.linalg.norm(d_w3) > GRADIENT_CLIP:
                d_w3 = GRADIENT_CLIP / np.linalg.norm(d_w3) * d_w3
            self.w3 += step_size_D * d_w3
            self.w3 = np.maximum(-WEIGHT_CLIP, np.minimum(WEIGHT_CLIP, self.w3))
            # Update b3
            self.b3 += step_size_D * d_b3
            self.b3 = np.maximum(-WEIGHT_CLIP, np.minimum(WEIGHT_CLIP, self.b3))
        delta *= dLeakyReLU(self.y2)
        # Derivative with respect to w2
        d_w2 = np.matmul(np.transpose(self.y1), delta)
        # Derivative with respect to b2
        d_b2 = delta.copy()
        # Derivative with respect to y1
        delta = np.matmul(delta, np.transpose(self.w2))
        if apply_grads:
            # Update w2
            if np.linalg.norm(d_w2) > GRADIENT_CLIP:
                d_w2 = GRADIENT_CLIP / np.linalg.norm(d_w2) * d_w2
            self.w2 += step_size_D * d_w2
            self.w2 = np.maximum(-WEIGHT_CLIP, np.minimum(WEIGHT_CLIP, self.w2))
            # Update b2
            self.b2 += step_size_D * d_b2
            self.b2 = np.maximum(-WEIGHT_CLIP, np.minimum(WEIGHT_CLIP, self.b2))
        delta *= dLeakyReLU(self.y1)
        # Derivative with respect to w1
        d_w1 = np.matmul(np.transpose(self.x), delta)
        # Derivative with respect to b1
        d_b1 = delta.copy()
        # Derivative with respect to x
        delta = np.matmul(delta, np.transpose(self.w1))
        # Update w1
        if apply_grads:
            if np.linalg.norm(d_w1) > GRADIENT_CLIP:
                d_w1 = GRADIENT_CLIP / np.linalg.norm(d_w1) * d_w1
            self.w1 += step_size_D * d_w1
            self.w1 = np.maximum(-WEIGHT_CLIP, np.minimum(WEIGHT_CLIP, self.w1))
            # Update b1
            self.b1 += step_size_D * d_b1
            self.b1 = np.maximum(-WEIGHT_CLIP, np.minimum(WEIGHT_CLIP, self.b1))
        return delta
if __name__ == '__main__':
    G = Generator()
    D = Discriminator()
    criterion = LossFunc()
    real_label = 1
    fake_label = 0
    for itr in range(ITER_NUM):
        # update D with real data
        real_data = get_samples(True)
        y_real = D.forward(real_data)
        loss_D_r =  criterion.forward(y_real, real_label)
        d_loss_D = criterion.backward()
        D.backward(d_loss_D)

        # Update D with fake data
        z_noise = np.random.randn(Z_DIM)
        x_fake = G.forward(z_noise)
        y_fake = D.forward(x_fake)
        loss_D_f = criterion.forward(y_fake, fake_label)
        d_loss_D = criterion.backward()
        D.backward(d_loss_D)

        # Update G with fake data
        y_fake_r = D.forward(x_fake)
        loss_G = criterion.forward(y_fake_r, real_label)
        d_loss_G = D.backward(loss_G, apply_grads=False)
        G.backward(d_loss_G)
        loss_D = loss_D_r + loss_D_f
        if itr % 100 == 0:
            print('{} {} {}'.format(loss_D_r.item((0, 0)), loss_D_f.item((0, 0)), loss_G.item((0, 0))))
