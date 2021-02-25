import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv1D, SeparableConv1D, LSTM, LSTMCell
from tensorflow.keras import Model

#
#Silver dynamic pointwise conv
#

class _dconv(Model):
    def __init__(self,h):
        super(_dconv,self).__init__()
        self.h = h
        self.enc_lstm = LSTM(self.h,return_sequences = True,return_state = True)
        self.dec_y = LSTMCell(self.h)
        self.Rev_gate = Dense(self.h,activation = "sigmoid")
        self.For_gate = Dense(self.h,activation = "sigmoid")
        self.Rev_lin = Dense(self.h,activation = "tanh")
        self.For_lin = Dense(self.h,activation = "tanh")
        self.m = tf.Variable(tf.random.normal([1,1,self.h],mean = 0.0, stddev = 1/(self.h**0.5)),trainable = True)
        return
    def call(self,input_seq):
        b = input_seq.shape[0]
        t = input_seq.shape[1]
        #make the encoder sequence
        y,enc_h,enc_c = self.enc_lstm(input_seq)
        F = tf.zeros([b,0,self.h])
        dec_state = (enc_h,enc_c)
        for c in range(t-1):
            dec_input = tf.concat([input_seq[:,c,:],input_seq[:,t-c-1,:]],1)
            #dec_input = tf.zeros([b,self.h])
            _y,dec_state = self.dec_y(dec_input,dec_state)
            _y = tf.expand_dims(_y,1)
            F = tf.concat([F,_y],axis = 1)
        F1 = self.For_lin(F)*self.For_gate(F)
        #F1 = self.For_lin(F)
        R = tf.reverse(self.Rev_lin(F)*self.Rev_gate(F),[1])
        #R = tf.reverse(self.Rev_lin(F),[1])
        D = tf.concat([R,tf.tile(tf.nn.tanh(self.m),[b,1,1]),F1],axis = 1)
        #D = tf.concat([R,tf.tile(tf.nn.relu(self.m),[b,1,1]),F1],axis = 1)
        #make conv mat
        e_conv = tf.zeros([b,0,t,self.h])
        for c1 in range(t):
            temp_e_conv = D[:,(t-1-c1):2*t-c1-1,:]
            #tempe_e_conv = tf.concat([R[:,t-1-c1:t-1,:],F])
            temp_e_conv = tf.expand_dims(temp_e_conv,1)
            e_conv = tf.concat([e_conv,temp_e_conv],1)
        return e_conv

class _apply_dconv(Model):
    def __init__(self,h):
        super(_apply_dconv,self).__init__()
        self.h = h
        self.weight_lin = Dense(self.h,activation = None)
        self.x_lin = Dense(self.h,activation = "tanh")
        self.bias = tf.Variable(tf.zeros([1,1,self.h]),trainable = True)
        return
    def call(self,input_seq,dconv):
        b = input_seq.shape[0]
        t = input_seq.shape[1]
        c_dconv = self.weight_lin(dconv)
        #c_dconv = dconv
        input_seq = tf.tile(tf.expand_dims(self.x_lin(input_seq),1),[1,t,1,1])
        #input_seq = tf.tile(tf.expand_dims((input_seq),1),[1,t,1,1])
        product = tf.reduce_sum(input_seq*c_dconv,axis = 2) + tf.tile(self.bias,[b,t,1])
        return product

#
# ATTENTIVE FULLY CONNECTED CONVOLUTION
#

class _apply_aconv(Model):
    def __init__(self,h,f):
        super(_apply_aconv,self).__init__()
        self.h = h
        self.f = f
        #self.x_lin = Dense(self.h,activation = "relu")
        self.weight_lin = Dense(self.f,activation = None)
        self.conv_lin = Dense(self.h,activation = None)
    def call(self,input_seq,dconv):
        b = input_seq.shape[0]
        t = input_seq.shape[1]
        aconv = tf.nn.softmax(self.weight_lin(dconv),2)
        #[b, t, t, f]
        aconv = tf.tile(tf.expand_dims(aconv,3),[1,1,1,self.h,1])
        #input_seq = self.x_lin(input_seq)
        input_seq = tf.tile(tf.expand_dims(input_seq,1),[1,t,1,1])
        input_seq = tf.tile(tf.expand_dims(input_seq,4),[1,1,1,1,self.f])
        product = tf.reduce_sum(aconv*input_seq,axis = 2)
        #[b,t,h,f]
        product = self.conv_lin(tf.reshape(product,[b,t,self.h*self.f]))
        return product
#
#Make Recurrent Units
#

class _cell_pw(Model):
    def __init__(self,h):
        super(_cell_pw,self).__init__()
        self.h = h
        self.u_forget_conv = _apply_dconv(self.h)
        self.z_reset_conv = _apply_dconv(self.h)
        self.candidate = _apply_dconv(self.h)
        return
    def call(self,hidden_state,dconv):
        u_forget = tf.nn.sigmoid(self.u_forget_conv(hidden_state,dconv))
        z_reset = tf.nn.sigmoid(self.z_reset_conv(hidden_state,dconv))
        y = u_forget*hidden_state + (1 - u_forget)*tf.nn.tanh(self.candidate(z_reset*hidden_state,dconv))
        return y
    
class _cell_attend(Model):
    def __init__(self,h):
        super(_cell_attend,self).__init__()
        self.h = h
        self.f = 7
        self.u_forget_conv = _apply_aconv(self.h,self.f)
        self.z_reset_conv = _apply_aconv(self.h,self.f)
        self.candidate = _apply_aconv(self.h,self.f)
        return
    def call(self,hidden_state,dconv):
        u_forget = tf.nn.sigmoid(self.u_forget_conv(hidden_state,dconv))
        z_reset = tf.nn.sigmoid(self.z_reset_conv(hidden_state,dconv))
        y = u_forget*hidden_state + (1 - u_forget)*tf.nn.tanh(self.candidate(z_reset*hidden_state,dconv))
        return y

#
#Vanilla NGPU and PW NGPU
#

class NGPU_cell(Model):
    def __init__(self,h):
        super(NGPU_cell,self).__init__()
        self.h = h
        self.u_forget_conv = Conv1D(h,3,padding = "same")
        self.z_reset_conv = Conv1D(h,3,padding = "same")
        self.candidate = Conv1D(h,3,padding = "same")
        return
    def call(self,hidden_state):
        u_forget = tf.nn.sigmoid(self.u_forget_conv(hidden_state))
        z_reset = tf.nn.sigmoid(self.z_reset_conv(hidden_state))
        y = u_forget*hidden_state + (1 - u_forget)*tf.nn.tanh(self.candidate(z_reset*hidden_state))
        return y

class PW_NGPU_cell(Model):
    def __init__(self,h):
        super(PW_NGPU_cell,self).__init__()
        self.h = h
        self.u_forget_conv = SeparableConv1D(h,3,padding = "same")
        self.z_reset_conv = SeparableConv1D(h,3,padding = "same")
        self.candidate = SeparableConv1D(h,3,padding = "same")
        return
    def call(self,hidden_state):
        u_forget = tf.nn.sigmoid(self.u_forget_conv(hidden_state))
        z_reset = tf.nn.sigmoid(self.z_reset_conv(hidden_state))
        y = u_forget*hidden_state + (1 - u_forget)*tf.nn.tanh(self.candidate(z_reset*hidden_state))
        return y

class NGPU(Model):
    def __init__(self,h,o):
        super(NGPU,self).__init__()
        self.h = h
        self.o = o
        self.cell = PW_NGPU_cell(self.h)
        #self.cell = NGPU_cell(self.h)
        self.h_lin = Dense(self.h,activation = "tanh")
        self.o_lin = Dense(self.o,activation = None)
        return
    def call(self,input_sequence,dropout = 0.0):
        hidden = self.h_lin(input_sequence)
        for c in range(input_sequence.shape[1]):
            hidden = self.cell(hidden)
            hidden = tf.nn.dropout(hidden,dropout)*(1-dropout)
        o = self.o_lin(hidden)
        return o
#
#Used Model
#

class _recur(Model):
    def __init__(self,h,o):
        super(_recur,self).__init__()
        self.h = h
        self.o = o
        self.dconv_gen = _dconv(self.h)
        #self.cell = _cell_attend(self.h)
        self.cell = _cell_pw(self.h)
        self.h_lin = Dense(self.h,activation = "tanh")
        self.o_lin = Dense(self.o,activation = None)
        return
    def call(self,input_sequence,dropout = 0.0):
        dconv = self.dconv_gen(input_sequence)
        hidden = self.h_lin(input_sequence)
        for c in range(input_sequence.shape[1]):
            hidden = self.cell(hidden,dconv)
            hidden = tf.nn.dropout(hidden,dropout)*(1-dropout)
        o = self.o_lin(hidden)
        return o

if __name__ == "__main__":
    #dconvg = silver_dconv(10)
    #sadconv = silver_apply_dconv(10)
    #test_input = tf.random.normal([1,3,4])
    #a = dconvg(test_input)
    #test_input2 = tf.random.normal([1,3,10])
    #y = sadconv(test_input2,a)
    #print(y.shape)
    test_recur = _recur(5,4)
    #test_recur = NGPU(5,4)
    input_sequence = tf.random.normal([2,6,3])
    o = test_recur(input_sequence)
    print(o.shape)