# Reinforcement Learning-Based Adaptive Traffic Signal Control

An educational college mini-project that lets a tabular Q-Learning agent interact with a small, stochastic four-way traffic simulation. It is designed for a live demonstration and academic viva—not for real-road deployment.

## Problem statement

Fixed-time traffic lights use predetermined cycles even when traffic demand varies by approach. This mini-project asks whether an agent can learn which legal traffic phase to request from compact queue observations, using experience from repeated simulated interaction. The project reports measurements against a fixed-time reference and makes no assumption that Q-Learning will always win.

## Objectives and proposed solution

The simulation draws one top-down four-way intersection with a separate straight-through lane and queue for North, South, East, and West. Vehicle arrivals are stochastic and selectable at low, medium, or high density. A tabular Q-Learning agent observes queue categories plus the current phase, selects one of four directional actions, receives a weighted reward, and updates a 162 × 4 table. An environment-owned safety layer maps those actions to two legal axes and enforces all-red clearance, minimum green, and maximum green. A non-learning, predetermined fixed-time controller is included only as an evaluation reference.

## Technology stack

- Python 3.10 or later (3.11 recommended)
- Pygame for the 2D application and responsive control panel
- NumPy for the 162 × 4 Q-table and numerical operations
- Python standard library for stochastic arrivals, evaluation seeds, and CSV output
- Optional Matplotlib if you want an additional exported graph; the Pygame interface already draws its own learning trace

No API, database, cloud service, prediction model, or external RL library is needed.

## Reinforcement Learning definitions

- The simulator advances at 30 Hz (`DT = 1/30` second). Training and evaluation still make decisions once per simulated second; each training episode is 90 seconds and each matched comparison episode is 120 seconds.
- **Agent:** the tabular Q-Learning controller.
- **Environment:** the stochastic four-approach intersection simulator.
- **State:** each approach's queue category (Low 0–2 vehicles, Medium 3–5, High 6+) plus the current phase (N–S or E–W). There are 3⁴ × 2 = 162 states.
- **Actions:** 0 = North, 1 = South, 2 = East, 3 = West. North or South requests the shared North–South green phase; East or West requests the East–West phase. Four queues remain independent, so the observation still shows directional imbalance.
- **Reward:** `r = 2.5 × vehicles served − 0.03 × waiting seconds − 0.125 × queue vehicle-seconds − 0.8 × phase switches − 0.025 × excess-starvation exposure`. Queue and starvation rate penalties are integrated over each 1/30-second update, preserving the previous penalty per simulated second; an approach's longest current wait above 15 seconds contributes to excess-starvation exposure. Positive throughput adds reward; waiting, congestion, switching, and starvation subtract reward. These are educational coefficients, not calibrated traffic-engineering values.

The learning update is:

```text
Q(s, a) ← Q(s, a) + α [r + γ max_a′ Q(s′, a′) − Q(s, a)]
```

`Q(s,a)` is the current return estimate; `α` controls update size; `r` is the observed reward; `γ` discounts later reward; and `max Q(s′,a′)` is the best next-state estimate. At a terminal training step, the future-value term is zero. During training, an epsilon-greedy policy explores a random action with probability `ε` and otherwise selects the action with the highest current value; `ε` decays toward its configured minimum. During evaluation, the policy is greedy (`ε = 0` for action selection); evaluation does not update the table.

## Safety and phase design

The environment permits only two green phases: N–S green with E–W red, or E–W green with N–S red. Actions are requests; the environment, not the agent, decides when a requested change can take effect. The current phase must meet a 5-second minimum, a phase is limited to 17 seconds before the environment requests the other axis, and a one-second all-red interval is inserted during axis changes. Vehicles already inside the intersection are allowed to clear during all-red; new entrants must have the green for their axis. Vehicles keep a minimum same-lane gap and cannot cross a red stop line.

## How to run the project

### Quick start (Windows, PowerShell)

1. Install Python 3.10+ (3.11 recommended) from [python.org](https://www.python.org/downloads/windows/) and tick "Add python.exe to PATH" / enable the `py` launcher. Check with `py --version`.
2. Open PowerShell in the project folder (the one containing `main.py`).
3. Create a virtual environment and install the dependencies (one time only):

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

4. Start the app:

```powershell
python main.py
```

On later runs you only need steps 2, `.venv\Scripts\Activate.ps1`, and 4.

If PowerShell blocks activation ("running scripts is disabled"), either run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, use Command Prompt with `.venv\Scripts\activate.bat`, or skip activation and call `.venv\Scripts\python.exe main.py` directly.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 main.py
```

### Using the app

A 1280 × 840 Pygame window opens. Press **Esc** or close the window to quit.

Suggested first session:

1. Pick a traffic density (**LOW / MEDIUM / HIGH**).
2. Click **START** for a live demo with no learning.
3. Click **TRAIN AGENT** (default 80 episodes; adjust with **EPISODES −/+**). Use **SPEED +** to train faster. The reward trace updates after each episode.
4. After training, click **RUN LEARNED POLICY** or **FIXED-TIME BASELINE**. Either one starts the 5-seed comparison (10 runs); the results table fills in when it finishes.

The agent stores its table in `models/q_table.npy` (plus `models/agent_config.json`), and the app loads it automatically on the next start. Training history is written to `results/episode_history.csv` and comparison runs to `results/comparison.csv`. To start from scratch, delete the files in `models/` and `results/`.

### Deploy to the web (Vercel)

The project can run in a browser through Pygbag (Python compiled to WebAssembly), served as a static site.

```powershell
pip install pygbag
python build_web.py
```

This writes the site to `public/`. Commit it (`git add . && git commit`), push to GitHub, then on vercel.com choose **Add New > Project**, import the repo and deploy. `vercel.json` already sets `public` as the output directory with no build step. Re-run `python build_web.py` and push again whenever the code changes. The first load in a browser downloads the Python runtime and NumPy (about 20 MB), so give it a minute. In the browser build the Q-table and CSVs live in memory only, so they reset on page reload.

### Run the tests (no window needed)

From the project folder:

```powershell
python -m unittest discover -s tests -v
```

### Troubleshooting

- `ModuleNotFoundError: pygame` or `numpy`: the virtual environment is not active, or `pip install -r requirements.txt` was not run.
- `python` / `py` not recognised: reinstall Python with PATH / the launcher enabled.
- Window does not open on a remote or headless machine: Pygame needs a display; the tests above run without one.
- Optional Matplotlib graph support: `pip install matplotlib` (not required).

## Train and evaluate

Choose the episode count in the sidebar and select **Train Agent**. Training advances a small batch from the Pygame event loop so the interface can continue to draw and accept events. The panel shows progress, current episode reward, exploration rate, queues, served vehicles, wait, and an episode-reward graph. Training defaults to 80 episodes; change the count in the interface to explore. The trace is measurement, not proof of convergence.

**Run Learned Policy** uses the greedy Q-table without exploration. **Run Fixed-Time Baseline** alternates the N–S and E–W axes every 12 seconds and does not adapt to queues. Either evaluation starts an incremental comparison: both controllers are measured for five 120-second episodes using the same seeded arrival events within each pair. The table reports mean waiting time per served vehicle, time-averaged queue per approach, vehicles served per episode, and actual phase switches. Per-episode rows are written to `results/comparison.csv`; observed values may favor either controller and depend on the learned table, traffic density, and chosen training budget.

## File organization

```text
EXPLAINED.txt           Plain-language overview of the whole project
build_web.py            Builds the Pygbag browser version into public/
vercel.json             Vercel static-site config (serves public/)
main.py                 Pygame interface and event/render loop
simulation.py           Vehicle arrival, queues, movement, state, rewards' inputs
vehicle.py              One vehicle's state
traffic_signal.py       Safe legal phases, minimum/maximum green, all-red clearance
q_learning_agent.py     Visible NumPy table, epsilon-greedy choice and update
training.py             Incremental training session and reward function
evaluation.py           Matched-seed evaluation and fixed-time reference
requirements.txt        Pygame and NumPy
models/                 Local q_table.npy (created when training saves)
results/                episode_history.csv and comparison.csv
```

## Experimental results and limitations

The interface and CSV files report the measurements generated on your machine. There are intentionally **no claimed or pre-filled performance improvements** in this source. A reproducible seed makes each paired arrival stream comparable, but five runs remain a small sample. Results may be poor with short training, directional imbalance, high arrival rates, coarse state bins, or a reward function whose coefficients do not reflect the desired trade-off.

The model deliberately omits turns, lane choice, pedestrians, cyclists, emergency vehicles, multiple intersections, realistic vehicle dynamics, detector failures, and legal signal certification. The fixed-time schedule and reward coefficients are educational simplifications. Results from this simulation do not demonstrate that the learned table is safe or effective on public roads. **This is not a real-world traffic-signal controller.**

## Viva preparation

- **Why is RL suitable?** Signal actions affect later queue and delay outcomes; the simulator provides interaction and reward without labeled target actions.
- **Why Q-Learning?** The discretized environment is small enough for a transparent table and straightforward update equation, appropriate for an undergraduate demonstration.
- **What does the table represent?** For each observed queue-category/phase state and each of four directional actions, it stores an estimate of discounted future return.
- **What is epsilon-greedy exploration?** It chooses a random action with probability ε and a best-known action otherwise. Decay shifts behavior from exploration toward the current greedy policy.
- **Why have a reward?** It explicitly communicates the trade-off among throughput, waiting, queues, starvation, and switching; different weights can change what the agent learns.
- **How might it prioritize congestion?** Queue buckets are part of the state, and queue/wait/throughput terms provide feedback after a phase choice. The Q-table may learn different preferred actions for different congestion patterns; this is not guaranteed after a short run.
- **Why might it perform poorly?** Arrival randomness, state aggregation, insufficient training, reward-weight choices, and omitted traffic complexity can all limit the learned policy.
- **Why is it not deployment-ready?** A simplified educational simulator cannot establish real-world safety, regulatory compliance, accurate demand response, sensor reliability, or signal-timing validity.
