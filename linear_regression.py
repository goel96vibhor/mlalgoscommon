class LinearReg:

    def __init__(self, x_train, y_train, alpha=0.01, iters=10):
        self.xmean = np.mean(x_train)
        self.xstd = np.mean(x_train)
#         self.xnorm = zscore_normalize(x_train, self.xmean, self.xstd)
        self.xnorm = x_train
        self.y_train = y_train.reshape(-1,1)
        print("xnorm, y_train shape", self.xnorm.shape, self.y_train.shape)
        m, n = self.xnorm.shape
        self.m = m
        self.w = np.zeros((n,1))
        self.b = np.array([0])
        self.alpha = alpha
        self.iters = iters
    
    def compute_loss(self, x, y):
        assert(x.shape[-1] == self.w.shape[-1])
        assert(x.shape[0] == y.shape[0])
        example_losses = (np.matmul(x, self.w) + self.b - y)**2
        loss = np.sum(example_losses)/(2*self.m)
        return loss
    
    def get_gradients(self, x, y):
        
        assert(x.shape[-1] == self.w.shape[-1])
        assert(x.shape[0] == y.shape[0])
#         print("w shape, b shape, x shape, y shape", self.w.shape, self.b.shape, x.shape, y.shape)
        example_grad = np.multiply(np.matmul(x, self.w) + self.b - y, x)
        djw = np.sum(example_grad, axis=0, keepdims=True)/self.m
        djb = np.sum(np.matmul(x, self.w) + self.b - y, axis=0, keepdims=True)/self.m
#         print("djw shape, djb shape", djw.shape, djb.shape)
        return djw, djb
    
    def fit(self):
        for i in range(self.iters):
            loss = self.compute_loss(self.xnorm, self.y_train)
            djw, djb = self.get_gradients(self.xnorm, self.y_train)
            self.w = self.w - self.alpha * djw
            self.b = self.b - self.alpha * djb
            print("iter %f loss %f" %(i, loss))
    
    def predict(self, x_eval):
        assert(x_eval.shape()[-1] == self.w.shape[-1])
        x_eval_norm = zscore_normalize(x_eval, self.xmean, self.xstd)
        return np.dot(self.w, x_eval_norm) + b
