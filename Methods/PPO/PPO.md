1. state, action, reward, old policy의 batch를 초기화
```py
batch_state, batch_action, batch_reward = [], [], []
batch_log_old_policy_pdf = []
```

2. episode 및 env 초기화
```py
time, episode_reward, done = 0, 0, False
state = self.env.reset()
```

3. 이전 정책의 평균, 표준편차를 계산하고 행동 샘플링
```py
mu_old, std_old, action = self.get_policy_action(tf.convert_to_tensor([state], dtype=tf.float32))
```

4. 행동 범위 클리핑
```py
action = np.clip(action, -self.action_bound, self.action_bound)
```

5. 이전 정책의 로그 확률밀도함수 계산
```py
var_old = std_old ** 2
log_old_policy_pdf = -0.5 * (action - mu_old) ** 2 / var_old - 0.5 * np.log(var_old * 2 * np.pi)
log_old_policy_pdf = np.sum(log_old_policy_pdf)
```

