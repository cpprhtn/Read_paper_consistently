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

6. action을 실행하여 reword와 next state x를 얻는다, done=1이 되면 episode가 종료
```py
next_state, reward, done, _ = self.env.step(action)
```

7. 학습용 reward 범위 조정
```py
train_reward = (reward + 8) / 8
```

8. 배치에 저장
```py
batch_state.append(state)
batch_action.append(action)
batch_reward.append(train_reward)
batch_log_old_policy_pdf.append(log_old_policy_pdf)
```

9. 배치가 채워질 때까지 학습하지 않고 저장만 계속
```py
if len(batch_state) < self.BATCH_SIZE:
    # 상태 업데이트
    state = next_state
    episode_reward += reward[0]
    time += 1
    continue
```

10. 배치가 채워지면, 데이터를 추출 후 배치는 초기화
```py
states = self.unpack_batch(batch_state)
actions = self.unpack_batch(batch_action)
rewards = self.unpack_batch(batch_reward)
log_old_policy_pdfs = self.unpack_batch(batch_log_old_policy_pdf)

batch_state, batch_action, batch_reward, = [], [], []
batch_log_old_policy_pdf = []
```
