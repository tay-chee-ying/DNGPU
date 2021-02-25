import numpy as np
def np_add(a,b):
  bitnum = a.shape[0]
  c = np.zeros([bitnum])
  carry = 0
  for i in range(bitnum):
    ind = bitnum-1-i
    sum = a[ind]+b[ind]+carry
    if sum==1:
      c[ind] = 1
      carry = 0
    elif sum == 2:
      c[ind] = 0
      carry = 1
    elif sum == 3:
      c[ind] = 1
      carry = 1
    elif sum == 0:
      c[ind] = 0
  return c
def np_multiply(a,b):
  template = np.zeros([a.shape[0]+b.shape[0]])
  #iterate for lenght b
  for c in range(b.shape[0]):
    multiple = b[b.shape[0]-1-c]*a
    #buffer a
    acat = np.concatenate([np.zeros([b.shape[0]-c]),multiple,np.zeros([c])])
    template = np_add(acat,template)
  return template
class make_multiply():
  def __init__(self,max_len = 5):
    self.select_std = 0.2
    self.buffer_std = 0.2
    #self.sublen_std = 0.0
    self.max_len = max_len
  def make_one(self):
    #make distance proprotions
    a = np.random.normal(loc = 1,scale = self.select_std,size = 1)[0]
    v1 = round(a*self.max_len)
    v2 = self.max_len*2 - v1
    if v1>self.max_len*2-1:
      v1 = self.max_len*2 - 1
      v2 = 1
    if v1<1:
      v1 = 1
      v2 = self.max_len*2 - 1
    #make s1 and s2
    s1 = np.random.randint(0,2,size = [v1])
    s2 = np.random.randint(0,2,size = [v2])
    #override
    v1b = int(np.random.normal(loc = 0, scale = self.buffer_std)*v1)
    v2b = int(np.random.normal(loc = 0, scale = self.buffer_std)*v2)
    if v1b<0:
        v1b = 0
    elif v1b>v1:
        v1b = v1
    if v2b<0:
        v2b = 0
    elif v2b>v2:
        v2b = v2
    s1[0:v1b] = 0
    s2[0:v2b] = 0
    temp = np_multiply(s1,s2)
    return temp,s1,s2
  def make_batch(self,batch_size):
    Y = np.zeros([batch_size,self.max_len*2+1,3])
    X = np.zeros([batch_size,self.max_len*2+1,4])
    for c in range(batch_size):
      y,x1,x2 = self.make_one()
      Y[c,0:y.shape[0],0] = y
      Y[c,0:y.shape[0],1] = 1-y
      Y[c,y.shape[0]:self.max_len*2+1,2] = 1
      X[c,0:x1.shape[0],0] = x1
      X[c,0:x1.shape[0],1] = 1-x1
      X[c,x1.shape[0],2] = 1
      X[c,x1.shape[0]+1:x2.shape[0]+x1.shape[0]+1,0] = x2
      X[c,x1.shape[0]+1:x2.shape[0]+x1.shape[0]+1,1] = 1-x2
      X[c,x1.shape[0]+x2.shape[0]+1:self.max_len+1,3] = 1
    return X,Y
if __name__ == "__main__":
    mm = make_multiply(max_len = 2)
    x,y = mm.make_batch(1)
    print(x)
    print(y)