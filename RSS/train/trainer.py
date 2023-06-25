import numpy as np


def train(log_fw, params, env, agent, buffer):
    EP_LEN = params.ep_len
    TRAIN_EP = params.train_ep
    START_REPLAY = params.start_replay
    TEST_INTERVAL = params.test_interval
    ADJUST_LR = params.lr_adjust
    TRAIN_INTERVAL = params.train_interval
    SAVE_HIS = params.save_his
    summary_step = 0
    loss_step = 1
    train_flag = False
    actor_mean_loss = 0.
    critic_mean_loss = 0.
    actor_mean_grad = 0.
    critic_mean_grad = 0.
    for ep_cnt in range(TRAIN_EP):
        evaluation_flag = train_flag and (ep_cnt % TEST_INTERVAL == 0)
        if evaluation_flag:
            test_mean_r, test_mean_return, test_a = test(params, env, agent)
            if ADJUST_LR:
                agent.adjust_lr()
            lr_a = agent.actor_opt.state_dict()['param_groups'][0]['lr']
            log_fw.add_scalar("lr_a", lr_a, summary_step)
            log_fw.add_scalar("mean_r", test_mean_r, summary_step)
            log_fw.add_scalar("mean_return", test_mean_return, summary_step)
            log_fw.add_scalar("critic_mean_loss", critic_mean_loss, summary_step)
            log_fw.add_scalar("critic_mean_grad", critic_mean_grad, summary_step)
            log_fw.add_scalar("actor_mean_loss", actor_mean_loss, summary_step)
            log_fw.add_scalar("actor_mean_grad", actor_mean_grad, summary_step)
            print(f"Episode: {ep_cnt} | Test Reward: {test_mean_r}")
            summary_step += 1
            actor_mean_loss = 0.
            critic_mean_loss = 0.
            loss_step = 1
            actor_mean_grad = 0.
            critic_mean_grad = 0.
        atm1 = env.init_a()
        ot = env.reset()
        st = ot
        agent.clear_inner_history()
        h = None
        for step_cnt in range(EP_LEN):
            if train_flag:
                at, h = agent.choose_action(atm1, ot, htm1=h, env=env)
            else:
                at = (np.random.rand(env.a_dim) - 0.5) * 2.
                at = (at, agent.a_filter.mapper(at))
            stp1, otp1, rt = env.step(st, at[1])
            buffer.push(atm1, ot, rt)
            ot = otp1
            st = stp1
            atm1 = at[0]
            if len(buffer) >= START_REPLAY and (step_cnt % TRAIN_INTERVAL == 0):
                batch = buffer.sample()
                log_a = SAVE_HIS and (step_cnt == 0) and evaluation_flag
                c_loss, a_loss, a_grad, c_grad = agent.train(batch, env, log_a, log_fw, summary_step)
                critic_mean_loss += (c_loss - critic_mean_loss) / loss_step
                actor_mean_loss += (a_loss - actor_mean_loss) / loss_step
                critic_mean_grad += (c_grad - critic_mean_grad) / loss_step
                actor_mean_grad += (a_grad - actor_mean_grad) / loss_step
                loss_step += 1
                train_flag = True
        buffer.store_episode()


def test(params, env, agent):
    TEST_EP = params.test_ep
    EP_LEN = params.ep_len
    test_r = np.empty((TEST_EP, EP_LEN), float)
    test_a = np.zeros((EP_LEN, env.a_dim), float)

    for i in range(TEST_EP):
        atm1 = env.init_a()
        ot = env.reset()
        st = ot
        htm1 = None
        agent.clear_inner_history()

        for j in range(EP_LEN):
            at, htm1 = agent.evaluate_action(atm1, ot, htm1=htm1, env=env)
            stp1, otp1, rt = env.step(st, at[1])
            test_a[j] = at[1].copy()
            ot = otp1
            st = stp1
            atm1 = at[0]
            test_r[i][j] = rt

    mean_reward = test_r.mean()
    mean_return = (test_r * agent.gamma_arr).sum(-1).mean()
    return mean_reward, mean_return, test_a.squeeze()