class Perceptron:
    def __init__(self,input_num,activator):
        self.activator = activator
        self.weights = [0.0] * input_num
        self.bias = 0.0

    def __str__(self):
        return 'weights\t:%s\nbias\t:%f\n' % (self.weights,self.bias)
    
    def predict(self,input_vec):
        weighted_input = map(lambda w,x:w*x,self.weights,input_vec)
        #print (list(weighted_input))
        return (self.activator(sum(weighted_input)+self.bias))
    
    def update_weights(self,input_vec,output,label,rate):
        delta = label - output
        self.weights = list(map(lambda w,x: w+rate*delta*x,self.weights,input_vec))
        self.bias = self.bias + rate * delta

    def iterate(self,input_vecs,label,rate):
        sample = zip(input_vecs,label)
        for input_vec,label in sample:
            output = self.predict(input_vec)
            self.update_weights(input_vec,output,label,rate)

    def train(self,input_vecs,label,iteration,rate):
        for i in range(iteration):
            self.iterate(input_vecs,label,rate)

def f(x):
    return 1 if x>0 else 0
def get_train_data():
    input_vecs = [[1,1],[0,0],[1,0],[0,1]]
    labels = [1,0,0,0]
    return input_vecs,labels

def train():
    p = Perceptron(2,f)
    input_vecs,labels = get_train_data()
    p.train(input_vecs,labels,10,0.1)
    return p

if __name__ == '__main__':
    p = train()
    print (p)
    print ('1 and 1 = %d' % p.predict([1, 1]))
    print ('0 and 0 = %d' % p.predict([0, 0]))
    print ('1 and 0 = %d' % p.predict([1, 0]))
    print ('0 and 1 = %d' % p.predict([0, 1]))
