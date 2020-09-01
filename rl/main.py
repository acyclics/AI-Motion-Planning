import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys
import time
import datetime
import numpy as np
from mpi4py import MPI
from copy import deepcopy

from env import make_env
from navigation import Navigation


comm = MPI.COMM_WORLD
rank = comm.Get_rank()


actors = [[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]]
learners = [15]


if rank >= 0 and rank <= 14:
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
elif rank == 15:
    pass
import tensorflow as tf


# Training parameters
batch_size = nsteps = 256
model_save_path = os.path.join("data", "nav_model")
optimizer_save_path = os.path.join("data", "optimizer")
record_dir = os.path.join("data", "record")


def actor():
    print(f"STARTING ACTOR with rank {rank}")
    sys.stdout.flush()

    # GAE hyper-parameters
    lam = 0.95
    gamma = 0.99

    # Build network architecture
    nav = Navigation(1, training=False)
    nav.call_build()

    # Get agent type
    agent_type = np.where(np.array(actors) == rank)[0][0]

    # Setup environment
    env = make_env()
    obs = env.reset()
    dones = False

    while True:
        weights = comm.recv(source=learners[agent_type])
        nav.set_weights(weights)

        mb_rewards = np.zeros([nsteps, 1], dtype=np.float32)
        mb_values = np.zeros([nsteps, 1], dtype=np.float32)
        mb_neglogpacs = np.zeros([nsteps, 1], dtype=np.float32)
        mb_dones = np.zeros([nsteps, 1], dtype=np.float32)
        mb_obs = np.zeros([nsteps, 16], dtype=np.float32)
        mb_actions = {
            'x1': np.zeros([nsteps, 1], dtype=np.int32),
            'x2': np.zeros([nsteps, 1], dtype=np.int32),
            'y1': np.zeros([nsteps, 1], dtype=np.int32),
            'y2': np.zeros([nsteps, 1], dtype=np.int32),
            'w1': np.zeros([nsteps, 1], dtype=np.int32),
            'w2': np.zeros([nsteps, 1], dtype=np.int32)
        }
        mb_logits = {
            'x1': np.zeros([nsteps, 21], dtype=np.float32),
            'x2': np.zeros([nsteps, 21], dtype=np.float32),
            'y1': np.zeros([nsteps, 21], dtype=np.float32),
            'y2': np.zeros([nsteps, 21], dtype=np.float32),
            'w1': np.zeros([nsteps, 21], dtype=np.float32),
            'w2': np.zeros([nsteps, 21], dtype=np.float32)
        }

        for i in range(nsteps):
            # Get actions of training agent
            actions, neglogp, entropy, value, logits = nav(np.expand_dims(obs, axis=0))

            mb_values[i]        = value
            mb_neglogpacs[i]    = neglogp
            mb_obs[i]           = obs
            for k in actions.keys():
                mb_actions[k][i]       = actions[k]
                mb_logits[k][i]        = logits[k]
            mb_dones[i]         = dones

            # Take actions in env and look at the results
            actions = {k: (v[0] - 10) / 10 for k, v in actions.items()}
            agent_actions = np.array([[actions['x1'], actions['y1'], actions['w1']],
                                      [actions['x2'], actions['y2'], actions['w2']]])
            obs, rewards, dones, infos = env.step(agent_actions)

            # Handle rewards
            mb_rewards[i]       = rewards
            
            if dones:
                obs = env.reset()

        # get last value for bootstrap
        _, _, _, last_values, _ = nav(np.expand_dims(obs, axis=0))

        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        # perform GAE calculation
        for t in reversed(range(nsteps)):
            if t == nsteps - 1:
                nextnonterminal = 1.0 - dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values

        # Send trajectory to learner
        mb_values = np.squeeze(mb_values, axis=-1)
        mb_rewards = np.squeeze(mb_rewards, axis=-1)
        mb_neglogpacs = np.squeeze(mb_neglogpacs, axis=-1)
        mb_returns = np.squeeze(mb_returns, axis=-1)
        mb_dones = np.squeeze(mb_dones, axis=-1)

        trajectory = {
            'mb_obs': mb_obs,
            'mb_actions': mb_actions,
            'mb_logits': mb_logits,
            'mb_returns': mb_returns,
            'mb_dones': mb_dones,
            'mb_values': mb_values,
            'mb_neglogpacs': mb_neglogpacs,
            'mb_rewards': mb_rewards
        }

        comm.send(trajectory, dest=learners[agent_type])


def learner():
    print(f"STARTING LEARNER with rank {rank}")
    sys.stdout.flush()

    # Truly-ppo hyperparameters
    KLRANGE = 0.03
    slope_rollback = -5
    slope_likelihood = 1
    target_kl = 0.05

    # Learner hyperparameters
    ent_coef = 0.01
    vf_coef = 0.5
    CLIPRANGE = 0.2
    max_grad_norm = 5.0
    noptepochs = 4
    nbatch = batch_size * len(actors[0])
    batch_scale = 2
    nbatch_steps = nbatch // batch_scale

    # Build network architecture
    nav = Navigation(nbatch_steps, training=True)
    nav.call_build()
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4, beta_1=0.9, beta_2=0.99, epsilon=1e-5)

    if os.path.isfile(model_save_path + ".npy"):
        weights = np.load(model_save_path + ".npy", allow_pickle=True)
        optimizer_weights = np.load(optimizer_save_path + ".npy", allow_pickle=True)

        grad_vars = nav.trainable_weights
        zero_grads = [tf.zeros_like(w) for w in grad_vars]
        optimizer.apply_gradients(zip(zero_grads, grad_vars))

        optimizer.set_weights(optimizer_weights)
        nav.set_weights(weights)
    
    # Get weights from file
    weights = nav.get_weights()

    # Get agent type
    agent_type = np.where(np.array(learners) == rank)[0][0]

    # Send agent to actor
    for actor in actors[agent_type]:
        comm.send(weights, dest=actor)
  
    # Truly PPO RL optimization loss function
    @tf.function
    def t_ppo_loss(b_obs, b_actions, b_logits, b_returns, b_dones, b_values, b_neglogpacs):
        # Stochastic selection
        inds = tf.range(nbatch)
        # Buffers for recording
        losses_total = []
        approxkls = []
        entropies = []
        # Start SGD
        for _ in range(noptepochs):
            inds = tf.random.shuffle(inds)
            for start in range(0, nbatch, nbatch_steps):
                end = start + nbatch_steps
                # Gather mini-batch
                mbinds = inds[start:end]
                mb_obs = tf.gather(b_obs, mbinds)
                mb_actions = {k: tf.gather(b_actions[k], mbinds) for k, v in b_actions.items()}
                mb_logits = {k: tf.gather(b_logits[k], mbinds) for k, v in b_logits.items()}
                mb_returns = tf.gather(b_returns, mbinds)
                mb_dones = tf.gather(b_dones, mbinds)
                mb_values = tf.gather(b_values, mbinds)
                mb_neglogpacs = tf.gather(b_neglogpacs, mbinds)
                with tf.GradientTape() as tape:
                    p_actions, p_neglogp, p_entropy, vpred, p_logits = nav(mb_obs)
                    # Batch normalize the advantages
                    advs = mb_returns - mb_values
                    advs = (advs - tf.reduce_mean(advs)) / (tf.math.reduce_std(advs) + 1e-8)
                    # Calculate neglogpac
                    #neglogpac = nav.diagguass.neglogp(p_mean, p_logstd, mb_actions)
                    neglogpac = (
                        nav.categoricalPd.neglogp(p_logits['x1'], mb_actions['x1']) +
                        nav.categoricalPd.neglogp(p_logits['x2'], mb_actions['x2']) +
                        nav.categoricalPd.neglogp(p_logits['y1'], mb_actions['y1']) +
                        nav.categoricalPd.neglogp(p_logits['y2'], mb_actions['y2']) +
                        nav.categoricalPd.neglogp(p_logits['w1'], mb_actions['w1']) +
                        nav.categoricalPd.neglogp(p_logits['w2'], mb_actions['w2'])
                    )
                    # Calculate the entropy
                    entropy = tf.reduce_mean(p_entropy)
                    # Get the predicted value
                    vpredclipped = mb_values + tf.clip_by_value(vpred - mb_values, -CLIPRANGE, CLIPRANGE)
                    # Unclipped value
                    vf_losses1 = tf.square(vpred - mb_returns)
                    # Clipped value
                    vf_losses2 = tf.square(vpredclipped - mb_returns)
                    vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
                    # KL
                    all_logits = tf.concat([p_logits['x1'], p_logits['x2'], p_logits['y1'], p_logits['y2'], p_logits['w1'], p_logits['w2']], axis=-1)
                    all_taken_logits = tf.concat([mb_logits['x1'], mb_logits['x2'], mb_logits['y1'], mb_logits['y2'], mb_logits['w1'], mb_logits['w2']], axis=-1)
                    kl = nav.categoricalPd.kl(all_logits, all_taken_logits)
                    #kl = nav.diagguass.kl(p_mean, p_logstd, mb_mean, mb_logstd)
                    approxkl = tf.reduce_mean(kl)
                    # Early stopping
                    #if approxkl > 1.5 * target_kl:
                    #    break
                    # Calculate ratio (pi current policy / pi old policy)
                    ratio = tf.exp(mb_neglogpacs - neglogpac)
                    # Defining Loss = - J is equivalent to max J
                    pg_targets = tf.where(
                        tf.logical_and( kl >= KLRANGE, ratio * advs > 1 * advs),
                        slope_likelihood * ratio * advs + slope_rollback * kl,
                        ratio * advs
                    )
                    pg_loss = -tf.reduce_mean(pg_targets)
                    # Total loss
                    loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
                # 1. Get the model parameters
                var = nav.trainable_variables
                grads = tape.gradient(loss, var)
                # 3. Calculate the gradients
                grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
                grads_and_var = zip(grads, var)
                # zip aggregate each gradient with parameters associated
                optimizer.apply_gradients(grads_and_var)
                losses_total.append(loss)
                approxkls.append(approxkl)
                entropies.append(entropy)
        losses_total = tf.reduce_mean(losses_total)
        approxkls = tf.reduce_mean(approxkls)
        entropies = tf.reduce_mean(entropies)
        return losses_total, approxkls, entropies

    # Normal trajectory
    trajectory = {
                'mb_obs':                               np.empty((nbatch, 16), dtype=np.float32),
                'mb_actions':                           {
                    'x1': np.empty((nbatch, 1), dtype=np.int32),
                    'x2': np.empty((nbatch, 1), dtype=np.int32),
                    'y1': np.empty((nbatch, 1), dtype=np.int32),
                    'y2': np.empty((nbatch, 1), dtype=np.int32),
                    'w1': np.empty((nbatch, 1), dtype=np.int32),
                    'w2': np.empty((nbatch, 1), dtype=np.int32),
                },
                'mb_logits':                            {
                    'x1': np.empty((nbatch, 21), dtype=np.float32),
                    'x2': np.empty((nbatch, 21), dtype=np.float32),
                    'y1': np.empty((nbatch, 21), dtype=np.float32),
                    'y2': np.empty((nbatch, 21), dtype=np.float32),
                    'w1': np.empty((nbatch, 21), dtype=np.float32),
                    'w2': np.empty((nbatch, 21), dtype=np.float32),
                },
                'mb_returns':                           np.empty((nbatch,), dtype=np.float32),
                'mb_dones':                             np.empty((nbatch,), dtype=np.float32),
                'mb_values':                            np.empty((nbatch,), dtype=np.float32),
                'mb_neglogpacs':                        np.empty((nbatch,), dtype=np.float32),
                'mb_rewards':                           np.empty((nbatch,), dtype=np.float32)
    }
    
    # Start learner process loop
    start_time = time.time()
    itrs = 0
    while True:
        # Collect enough rollout to fill batch_size
        traj_size = 0
        for idx, actor in enumerate(actors[agent_type]):
            # Append to trajectory
            a_trajectory = comm.recv(source=actor)
            a_traj_size = a_trajectory['mb_returns'].shape[0]
            for k, v in trajectory.items():
                if k == 'mb_actions' or k == 'mb_logits':
                    for k2 in trajectory[k].keys():
                        trajectory[k][k2][traj_size:min(traj_size+a_traj_size, nbatch)] = a_trajectory[k][k2][0:max(0, nbatch-traj_size)]
                else:
                    trajectory[k][traj_size:min(traj_size+a_traj_size, nbatch)] = a_trajectory[k][0:max(0, nbatch-traj_size)]
            traj_size += min(a_traj_size, nbatch-traj_size)

        # Update Navigation when conditions are met
        b_obs = trajectory['mb_obs']
        b_actions = trajectory['mb_actions']
        b_logits = trajectory['mb_logits']
        b_dones = trajectory['mb_dones']
        b_values = trajectory['mb_values']
        b_neglogpacs = trajectory['mb_neglogpacs']
        b_returns = trajectory['mb_returns']
        b_rewards = trajectory['mb_rewards']

        # Start SGD and optimize model via Adam
        losses, approxkls, entropies = t_ppo_loss(b_obs, b_actions,
                                                  b_logits,
                                                  b_returns, b_dones,
                                                  b_values, b_neglogpacs)
    
        # Send agent to actor
        weights = nav.get_weights()
        for actor in actors[agent_type]:
            comm.send(weights, dest=actor)
        
        model_save_path_tmp = model_save_path + "_tmp"
        np.save(model_save_path_tmp, weights)
        if os.path.exists(model_save_path + ".npy"):
            os.remove(model_save_path + ".npy")
        os.rename(model_save_path_tmp + ".npy", model_save_path + ".npy")

        optimizer_save_path_tmp = optimizer_save_path + "_tmp"
        np.save(optimizer_save_path_tmp, optimizer.get_weights())
        if os.path.exists(optimizer_save_path + ".npy"):
            os.remove(optimizer_save_path + ".npy")
        os.rename(optimizer_save_path_tmp + ".npy", optimizer_save_path + ".npy")

        itrs += 1

        if itrs % 1000 == 0:
            curTime = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            np.save(os.path.join(record_dir, curTime), weights)
        
        print(f"{itrs}: approxkl = {approxkls} ; loss = {losses} ; entropy = {entropies} ; reward = {np.mean(b_rewards, axis=0)} ; time = {time.time() - start_time}")
        sys.stdout.flush()


if rank >= 0 and rank <= 14:
    actor()
elif rank >= 15 and rank <= 15:
    learner()
