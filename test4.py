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

db = tf.keras.datasets.mnist.load_data()
def get_data():
    return np.reshape(db[0][0],(60000,28,28,1))/255,db[0][1]

class Env(gym.Env):
    def __init__(self,*args):
        super().__init__()

        self.observation_space = Box(low=0,high=9,shape=(9*9*1+1,))
        self.action_space = Box(low=-1,high=1,shape=(10+2,)) #Classification : 10 // Saccades : 2

        self.train_img,self.train_lbl = get_data()
        self.index = 0

        #self.new_image()
        self.crop_pos = np.array([0.5,0.5])
        self.crop_shape = (9,9)
        self.nb_crop = 0
        self.nb_saccades = 6
        #self.new_crop()

        self.stats = np.array([[0 for i in range(5)] for j in range(5)])
        self.stats_count = 0

    def new_image(self):
        self.index = (self.index+1)%60000
        self.index = np.random.randint(60000)
        self.image = self.train_img[self.index]
        self.label = self.train_lbl[self.index]
        
    def new_crop(self):
        self.nb_crop += 1
        (image_h,image_w,_) = self.image.shape
        (crop_h,crop_w) = self.crop_shape
        [x,y] = self.crop_pos
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
        self.crop_pos = (action[10:12]+1)/2
        [x,y] = self.crop_pos
        if x == 1:
            x = 0.999
        if y == 1:
            y = 0.999
        self.stats[int(y*5)][int(x*5)] += 1
        self.stats_count += 1
        if self.stats_count == 256*20:
            self.stats = self.stats/np.sum(self.stats)*100
            self.stats = self.stats.astype(int)
            print("---",self.crop_pos)
            print(self.stats)
            self.stats = self.stats*0
            self.stats_count = 0
        #self.crop_pos = [0.5,0.5]
        cl = action[:10]
        index = np.argmax(cl)
        self.new_crop()
        done = self.nb_crop==self.nb_saccades
        reward = int(index == self.label)*done
        return self.crop,reward,done,{"label":self.label}
        
    def reset(self):
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
            s_o1 = tf.keras.layers.Dense(2,activation="tanh",name='ds_o1')(tf.stop_gradient(lstm2))
            s_o2 = tf.keras.layers.Dense(2,activation="tanh",name='ds_o2')(tf.stop_gradient(lstm2))
            
            #Value output
            v = tf.keras.layers.Dense(1,activation=None,name='dv')(lstm2)
            
            self.model = tf.keras.Model(inputs=[inp,inp1_h,inp2_h],outputs=[ tf.concat([c_o1,s_o1,c_o2,s_o2],2) , v ,s1h,s2h])
            self.register_variables(self.model.variables)

        @override(RecurrentNetwork)
        def forward_rnn(self,input_dict,state,seq_lens):
            inp = input_dict
            inp = inp[:,:,:-1]
            #Compute the model
            m_out,self.m_val,*state = self.model([inp,*state])
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
        def value_function(self):
            return tf.reshape(self.m_val,[-1])
    num_outputs = 24
    return KerasModel(obs_space,action_space,num_outputs,model_config,"keras_model")

from ray.rllib.agents.ppo.ppo_tf_policy import *
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
        return self.kl_coeff_val
    
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
      

def _ppo_surrogate_loss(policy, model, dist_class, train_batch):
    logits, state = model.from_batch(train_batch)
    logits = cut_gradient(logits,[(0,10,True),(10,12,False),(12,22,True),(22,24,True)])
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

def branch_loss(policy,model,dist_class,train_batch):
    #PPO Loss
    ppo_loss = _ppo_surrogate_loss(policy,model,dist_class,train_batch)
    #Classifier Loss
    labels = train_batch[SampleBatch.CUR_OBS][:,-1]
    out, state = model.from_batch(train_batch)
    dist = dist_class(out,model)
    out = dist.sample()
    pred = out[:,:10]
    #Mnih backprop ---------
    labels = labels[nb_saccades-1::nb_saccades]
    pred = pred[nb_saccades-1::nb_saccades]
    #-----------------------
    
    cl_loss = tf.keras.losses.sparse_categorical_crossentropy(labels,pred)
    cl_loss = tf.reduce_mean(cl_loss)
    #cl_loss = tf.Print(cl_loss,[cl_loss,ppo_loss],"--",summarize=-1)
    cl_loss = cl_loss*0.5
    return cl_loss + ppo_loss
    
BranchPolicy = build_tf_policy(
    name="PPOTFPolicy",
    make_model = make_model, #Give the model
    get_default_config=lambda: ray.rllib.agents.ppo.ppo.DEFAULT_CONFIG, #Default ppo config
    loss_fn=branch_loss,
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
          #'lr': 1e-2,
          'num_workers': 4,
          'train_batch_size': 256*4*8,
}
ppo_config= DEFAULT_CONFIG.copy()
ppo_config["rollout_fragment_length"] = 256*2
trainer = build_trainer(name="BranchTrainer",
                        #default_policy=PPOTFPolicy,
                        default_policy=BranchPolicy,
                        validate_config=validate_config,
                        execution_plan=execution_plan,
                        default_config=ppo_config)
trainer = trainer(config=config,env=Env)#"CartPole-v1")

if __name__ == "__main__":
    print(trainer.workers._local_worker.policy_map["default_policy"].get_weights()["default_policy/dc_o1/kernel"][0,0])
    for i in range(1000):
        for j in range(10):
            res = trainer.train()
            print(i*10+j,int(res["episode_reward_mean"]*100))
        print("saved - ",trainer.save())
    print(trainer.workers._local_worker.policy_map["default_policy"].get_weights()["default_policy/dc_o1/kernel"][0,0])
