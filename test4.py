import gym
from gym.spaces import *

#import tensorflow as tf

import numpy as np

import ray
import ray.rllib.agents.ppo.ppo as ppo
from ray.rllib.policy.policy import Policy
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.utils import try_import_tf
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy,postprocess_ppo_gae
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG,execution_plan,validate_config
from ray.rllib.policy.sample_batch import SampleBatch

import matplotlib.pyplot as plt

ray.init(ignore_reinit_error=True)
tf = try_import_tf()

#tf.enable_eager_execution()
global nb_saccades
nb_saccades = 6

mode = "test" #train -> evalue sur train_dataset // test -> evalue sur test_dataset

nb_epochs = 75

db = tf.keras.datasets.mnist.load_data()
def get_data():
    return np.reshape(db[0][0],(60000,28,28,1))/255,np.reshape(db[1][0],(10000,28,28,1))/255,db[0][1],db[1][1]

class Env(gym.Env):
    def __init__(self,*args):
        super().__init__()

        self.observation_space = Box(low=-1,high=9,shape=(9*9*1+1,))
        self.action_space = Box(low=-1,high=1,shape=(10+2,)) #Classification : 10 // Saccades : 2

        self.train_img,self.test_img,self.train_lbl,self.test_lbl = get_data()
        self.index = 0

        #self.new_image()
        self.crop_pos = np.array([0.5,0.5])
        self.crop_shape = (9,9)
        self.nb_crop = 0
        self.nb_saccades = nb_saccades
        #self.new_crop()

        self.stats = np.array([[0 for i in range(5)] for j in range(5)])
        self.stats_count = 0

        self.test = False
        self.test_count = 0

    def new_image(self):
        #self.index = (self.index+1)%60000
        if not(self.test) or mode=="train":
            self.index = np.random.randint(60000)
            self.image = self.train_img[self.index]
            self.label = self.train_lbl[self.index]
        else:
            self.index = np.random.randint(10000)
            self.image = self.test_img[self.index]
            self.label = self.test_lbl[self.index]
        self.test_count += 1
        #print(self.test_count,6*256)
        #print(self.test_count)
        if self.test_count > 30721:
            if not(self.test):
                print("TEST MODE",mode)
            self.test = True
        
    def new_crop(self):
        self.nb_crop += 1
        (image_h,image_w,_) = self.image.shape
        (crop_h,crop_w) = self.crop_shape
        [x,y] = self.crop_pos
        #[x,y] = [np.random.random(),np.random.random()]
        x = int(x*(image_w-1)+0.5)+(crop_w-1)//2
        y = int(y*(image_h-1)+0.5)+(crop_h-1)//2
        xmin = x-(crop_w-1)//2
        xmax = x+(crop_w-1)//2
        ymin = y-(crop_h-1)//2
        ymax = y+(crop_h-1)//2
        padlist = [[(crop_h-1)//2,(crop_h-1)//2],[(crop_w-1)//2,(crop_w-1)//2]]
        padout = np.pad(np.reshape(self.image,(28,28)),padlist,'constant')
        out = padout[ymin:ymax+1,xmin:xmax+1]
        out = np.reshape(out,(9,9,1))
        out = np.ndarray.flatten(out)
        out = np.concatenate([out,[self.label]])
        self.crop = out
        
    def step(self,action):
        if self.test:
            return self.__step(action)
        else:
            return self._step(action)

    def reset(self):
        if self.test:
            return self.__reset()
        else:
            return self._reset()

    #Train mode
    def _step(self,action):
        self.crop_pos = (action[10:12]+1)/2
        [x,y] = self.crop_pos
        #Print
        if x == 1:
            x = 0.999
        if y == 1:
            y = 0.999
        self.stats[int(y*5)][int(x*5)] += 1
        self.stats_count += 1
        if self.stats_count == 256*8*10:
            self.stats = self.stats/np.sum(self.stats)*100
            self.stats = self.stats.astype(int)
            print("---",self.crop_pos)
            print(self.stats)
            self.stats = self.stats*0
            self.stats_count = 0
            
        #self.crop_pos = [0.5,0.5]
        cl = action[:10]
        cl = np.clip(cl,1e-15,1)
        index = np.argmax(cl)
        self.new_crop()
        done = self.nb_crop==self.nb_saccades
        #reward = int(index == self.label)
        reward = -np.log(cl[self.label])
        reward = np.clip(reward,1e-15,1e15)
        reward = np.tanh(1/reward)
        #print(self.nb_crop,index,self.label,reward)
        return self.crop,reward,done,{}

    def _reset(self):
        self.nb_crop = 0
        self.crop_pos = np.array([0.5,0.5])
        self.new_image()
        self.new_crop()
        return self.crop

    #Test mode
    def __step(self,action):
        self.crop_pos = (action[10:12]+1)/2
        cl = action[:10]
        index = np.argmax(cl)
        self.new_crop()
        done = self.nb_crop==self.nb_saccades
        reward = int(index == self.label)*done
        #print(self.nb_crop,index,self.label,reward)
        return self.crop,reward,done,{"label":self.label}

    def __reset(self):
        self.nb_crop = 0
        self.crop_pos = np.array([0.5,0.5])
        self.new_image()
        self.new_crop()
        return self.crop

from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
def make_model(policy,obs_space,action_space,model_config):
    from ray.rllib.utils.annotations import override
    from ray.rllib.models.modelv2 import ModelV2
    class KerasModel(RecurrentNetwork):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            inp = tf.keras.layers.Input(shape=(None,9*9*1))
            
            inp1_h = tf.keras.layers.Input(shape=(256,))
            lstm1,s1h = tf.keras.layers.SimpleRNN(256,activation=tf.nn.relu,name='d1',return_state=True,return_sequences=True)(inp,initial_state=[inp1_h])
            
            inp2_h = tf.keras.layers.Input(shape=(128,))
            lstm2,s2h = tf.keras.layers.SimpleRNN(128,activation=tf.nn.relu,name='d2',return_state=True,return_sequences=True)(lstm1,initial_state=[inp2_h])
            
            #Classifier output
            c_o1 = tf.keras.layers.Dense(10,activation="softmax",name='dc_o1')(lstm2)
            c_o2 = tf.keras.layers.Dense(10,activation="softmax",name='dc_o2')(lstm2)
            
            #Saccader output
            clstm2 = tf.stop_gradient(lstm2)
            lstm2 = tf.keras.layers.Dense(64,activation=tf.nn.relu,name='d3')(clstm2)
            dense = tf.keras.layers.Dense(2,activation="tanh",name='ds_o1')
            s_o1 = dense(lstm2)
            s_o2 = tf.keras.layers.Dense(2,activation="tanh",name='ds_o2')(lstm2)
            
            #Value output
            v = tf.keras.layers.Dense(1,activation=None,name='dv')(lstm2)
            
            self.model = tf.keras.Model(inputs=[inp,inp1_h,inp2_h],outputs=[ tf.concat([c_o1,s_o1,c_o2,s_o2],2) , v ,s1h,s2h])
            w1,w2 = dense.get_weights()
            dense.set_weights([w1/100,w2/100])
            self.register_variables(self.model.variables) #required by rllib

        #def forward(input,...):
        #    input -> reshape [batch_size*timestep,input_sahep] -> [batchsize,timestep,input_shape]
        #    forward_rnn
        #    output -> reshape [batchsize,timestep,input_shape] -> [batch_size*timestep,input_sahep]

        @override(RecurrentNetwork)
        def forward_rnn(self,input_dict,state,seq_lens):
            #inp -> [batch_size,timesteps,*input_shape] (only in this function, otherwise : [batch_size*timesteps,input_shape])
            inp = input_dict
            inp = inp[:,:,:-1]
            #Compute the model
            m_out,self.m_val,*state = self.model([inp,*state]) #Store the value output separatly
            c_o1 = m_out[:,:,0:10]
            s_o1 = m_out[:,:,10:12]
            c_o2 = m_out[:,:,12:22]
            s_o2 = m_out[:,:,22:24]
            #Cut logstd
            c_o2 = c_o2*0-1e1
            s_o2 = s_o2*0-1e1
            #Prepare output
            m_out = tf.concat([c_o1,s_o1,c_o2,s_o2],axis=2)
            return m_out,state
        @override(ModelV2)
        def get_initial_state(self):
            return [np.zeros(256,np.float32),#np.zeros(256,np.float32),
                np.zeros(128,np.float32)]#np.zeros(128,np.float32)]
        @override(ModelV2)
        def value_function(self): #Rllib wants the value output separately
            return tf.reshape(self.m_val,[-1]) #reshape is copy pasted from rllib code
    num_outputs = 24
    return KerasModel(obs_space,action_space,num_outputs,model_config,"keras_model")

#Bidouillage de loss
#ray.rllib.agents -> agents rl
#                                -> ppo -> ppo.py -> trainer
#                                       -> ppo_tf_policy.py -> policy

def cut_gradient(tensor,slices):
    t_list = []
    for begin,end,stop in slices:
        t = tensor[:,begin:end]
        if stop:
            _t = tf.stop_gradient(t)*0
        else:
            _t = t
        t_list.append(_t)
    return tf.concat(t_list,axis=1)

from ray.rllib.agents.ppo.ppo_tf_policy import *
# ---- Here you can define modified stuff for ppo ----

class KLCoeffMixin:
    def __init__(self, config):
        # KL Coefficient
        self.kl_coeff_val = config["kl_coeff"]
        self.kl_target = config["kl_target"]
        self.kl_coeff = tf.get_variable(
            initializer=tf.constant_initializer(self.kl_coeff_val),
            name="kl_coeff",
            shape=(),
            trainable=False,
            dtype=tf.float32)

    def update_kl(self, sampled_kl):
        #coeff kl diverges -> removed the update
        return self.kl_coeff_val

def _ppo_surrogate_loss(policy, model, dist_class, train_batch):
    logits, state = model.from_batch(train_batch)
    logits = cut_gradient(logits,[(0,10,True),(10,12,False),(12,22,True),(22,24,True)]) #We redefined ppo_surogate_loss to cut the gradient
    action_dist = dist_class(logits, model)

    mask = None
    if state:
        max_seq_len = tf.reduce_max(train_batch["seq_lens"])
        mask = tf.sequence_mask(train_batch["seq_lens"], max_seq_len)
        mask = tf.reshape(mask, [-1])

    policy.loss_obj = PPOLoss(
        dist_class,
        model,
        train_batch[Postprocessing.VALUE_TARGETS],
        train_batch[Postprocessing.ADVANTAGES],
        train_batch[SampleBatch.ACTIONS],
        train_batch[SampleBatch.ACTION_DIST_INPUTS],
        train_batch[SampleBatch.ACTION_LOGP],
        train_batch[SampleBatch.VF_PREDS],
        action_dist,
        model.value_function(),
        policy.kl_coeff,
        mask,
        entropy_coeff=policy.entropy_coeff,
        clip_param=policy.config["clip_param"],
        vf_clip_param=policy.config["vf_clip_param"],
        vf_loss_coeff=policy.config["vf_loss_coeff"],
        use_gae=policy.config["use_gae"],
    )

    return policy.loss_obj.loss

#model.from_batch is computed both in ppo_loss and categorical loss separately. We could also compute it first and pass it to both functions
def branch_loss(policy,model,dist_class,train_batch): #[crop1,crop2,crop3,...,crop6,crop1-2,crop2-2,...] -> 9x9 images -> 81 + label -> 82 // [batch_size,timesteps,*input_shape]
    #PPO Loss [batch_size*timesteps,*input_shape] -> [batch_size,timesteps,*input_shape] -> [batch_size*timesteps,*input_shape] -> sum([batch_size,*input_shape]) / sum(batch_size*timesteps,*input_shape)
    ppo_loss = _ppo_surrogate_loss(policy,model,dist_class,train_batch)
    #Classifier Loss
    labels = train_batch[SampleBatch.CUR_OBS][:,-1]
    out, state = model.from_batch(train_batch) #Batches come from the workers -> we don't have the graph -> we recompute it
    #dist = dist_class(out,model) #dist_class applies a gaussian to the output but we don't really need it for the classification
    #out = dist.sample()
    pred = out[:,:10]
    """
    #Mnih backprop ---------
    labels = labels[nb_saccades-1::nb_saccades]
    pred = pred[nb_saccades-1::nb_saccades]
    #-----------------------
    """
    cl_loss = tf.keras.losses.sparse_categorical_crossentropy(labels,pred)
    cl_loss = tf.reduce_mean(cl_loss)
    #cl_loss = tf.Print(cl_loss,[cl_loss,ppo_loss],"--",summarize=-1)
    #cl_loss = cl_loss*0.9999 #To print the loss you need to multiply by something like 0.9999 but not 1.0
    return cl_loss + ppo_loss # d loss / d w -> d (cl_loss + ppo_loss) / dw 
    
BranchPolicy = build_tf_policy(
    name="PPOTFPolicy",
    make_model = make_model, #Give the model
    get_default_config=lambda: ray.rllib.agents.ppo.ppo.DEFAULT_CONFIG, #Default ppo config
    loss_fn=branch_loss, #This is the only change from the standard ppo policy
    stats_fn=kl_and_loss_stats,
    extra_action_fetches_fn=vf_preds_fetches,
    postprocess_fn=postprocess_ppo_gae,
    gradients_fn=clip_gradients,
    before_init=setup_config,
    before_loss_init=setup_mixins,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        ValueNetworkMixin
    ])

config = {'gamma': 0.9,
          #'lr': 1e-3,
          'num_workers': 4,
          'train_batch_size': 8*256*4,
          #"explore": True,
          
          #"exploration_config": {
          #    "type": "EpsilonGreedy",
          #    "initial_epsilon":1.0,  # default is 1.0
          #    "final_epsilon":0.05,  # default is 0.05
          #},
          
}
ppo_config= DEFAULT_CONFIG.copy()
#ppo_config["lambda"] = 1e-3
ppo_config["rollout_fragment_length"] = 8*256 #batch length for one worker
#Both config are needed -> but they are probably mixed -> for example changing lambda in either of them changes the result
_trainer = build_trainer(name="BranchTrainer",
                        #default_policy=PPOTFPolicy,
                        default_policy=BranchPolicy,
                        validate_config=validate_config,
                        execution_plan=execution_plan,
                        default_config=ppo_config)
trainer = _trainer(config=config,env=Env)#"CartPole-v1")

def crop_batch(batch,pos,crop_shape=(9,9)):
    return tf.image.extract_glimpse(batch,crop_shape,pos)

def compute_accuracy(batch,lbls,policy):
    batch_size = batch.shape[0]
    c = crop_batch(batch,tf.zeros((batch_size,2))+0.5)
    inputs = tf.reshape(c,(batch_size,81))
    state1 = np.zeros((batch_size,256),np.float32)
    state2 = np.zeros((batch_size,128),np.float32)
    state = [state1,state2]
    for i in range(6):
        inputs = tf.concat([inputs,np.ones((batch_size,1))],axis=1)
        m_out,state,_ = policy.compute_actions(inputs.eval(),state)
        pred = m_out[:,:10]
        sacc = (m_out[:,10:12]+1)/2
        plabel = tf.argmax(pred,axis=1).eval()
        c = crop_batch(batch,sacc)
        inputs = tf.reshape(c,(batch_size,81))
    issues = tf.cast(tf.equal(plabel,lbls),tf.int64)
    accuracy = tf.reduce_sum(issues)
    print(accuracy.eval())
    print(accuracy.eval()/batch_size*100)

if __name__ == "__main__":
    #print(trainer.workers._local_worker.policy_map["default_policy"].get_weights()["default_policy/dc_o1/kernel"][0,0])
    for i in range(nb_epochs):
        res = trainer.train() #-> goes through ppo_config["train_batch_size"]
        print(i,int(res["episode_reward_mean"]*100))
    res = trainer.train()
    print("---",int(res["episode_reward_mean"]*100))

"""
    with trainer.workers._local_worker.tf_sess.as_default():
        with trainer.workers.local_worker().tf_sess.graph.as_default():
            policy = trainer.get_policy()
            #policy = trainer.workers.remote_workers()[0].policy_map["default_policy"]
            compute_accuracy(get_data()[0][:1000],get_data()[1][:1000],policy)
"""
"""
            crops = crop_batch(get_data()[0][:100],tf.zeros((100,2))+0.5)
            crops = tf.reshape(crops,(100,81))
            crops = tf.concat([crops,tf.zeros((100,1))],axis=1)
            crops = crops.eval()
            actions,states,_ = trainer.workers._local_worker.policy_map["default_policy"].compute_actions(crops,[np.zeros((100,256)),np.zeros((100,128))])
            print(actions.shape)
"""
"""
    print(trainer.workers._local_worker.policy_map["default_policy"].get_weights()["default_policy/d1/kernel"][0,0])
    print(trainer.workers._local_worker.policy_map["default_policy"].get_weights()["default_policy/d1/recurrent_kernel"][0,0])
    print(trainer.workers._local_worker.policy_map["default_policy"].get_weights()["default_policy/d1/bias"][0,0])
    print(trainer.workers._local_worker.policy_map["default_policy"].get_weights()["default_policy/d2/kernel"][0,0])
    print(trainer.workers._local_worker.policy_map["default_policy"].get_weights()["default_policy/d1/recurrent_kernel"][0,0])
    print(trainer.workers._local_worker.policy_map["default_policy"].get_weights()["default_policy/d2/bias"][0,0])
    print(trainer.workers._local_worker.policy_map["default_policy"].get_weights()["default_policy/d3/kernel"][0,0])
    print(trainer.workers._local_worker.policy_map["default_policy"].get_weights()["default_policy/d3/bias"][0,0])
    print(trainer.workers._local_worker.policy_map["default_policy"].get_weights()["default_policy/dc_o1/kernel"][0,0])
    print(trainer.workers._local_worker.policy_map["default_policy"].get_weights()["default_policy/dc_o1/bias"][0,0])
    print(trainer.workers._local_worker.policy_map["default_policy"].get_weights()["default_policy/ds_o1/kernel"][0,0])
    print(trainer.workers._local_worker.policy_map["default_policy"].get_weights()["default_policy/ds_o1/bias"][0,0])
"""
