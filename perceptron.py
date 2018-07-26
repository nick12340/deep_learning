import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

class perceptron(object):
    def __init__(self,input_num,activator):
        self.activator = activator
        self.weights = [0.0 for _ in range(input_num)]
        self.bias = 0.0
    def __str__(self):
        return 'weights\t:%s\nbias\t:%f\n' % (self.weights,self.bias)

    def predict(self,input_vec):
        pair = zip(input_vec , self.weights)
        multipy = map(lambda (a,b): a*b, pair)
        def add(x,y): return x + y
        sum = reduce(add, multipy,0.0)
        return self.activator (sum +self.bias)

    def train(self,input_vecs,labels,iteration,rate):
        for i in range(iteration):
            self.iterate(input_vecs,labels,rate)

    def update_weight(self,input_vec,output,label,rate):
        delta = label - output
        self.weights =map(lambda (x,w): w + rate * delta * x,zip(input_vec,self.weights))
        self.bias = self.bias + rate*delta

    def iterate(self,input_vecs,labels,rate):
        samples = zip(input_vecs,labels)
        for (input_vec,label) in samples:
            output = self.predict(input_vec)
            self.update_weight(input_vec,output,label,rate)

def f(x):
    return 1 if x > 0 else 0

def get_data():
    input_vecs = [[1,1],[0,0],[1,0],[0,1]]
    labels = [1,0,0,0]
    return input_vecs,labels
def train_perceptron():
    p = perceptron(2,f)
    input_vecs,labels = get_data()
    p.train(input_vecs,labels,10,0.1)
    return p

if __name__ == '__main__':
    sb = train_perceptron()
    print sb
    print '1 and 1 = %d' % sb.predict([1, 1])
    print '0 and 0 = %d' % sb.predict([0, 0])
    print '1 and 0 = %d' % sb.predict([1, 0])
    print '0 and 1 = %d' % sb.predict([0, 1])
