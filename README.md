# Reinforcement Learning-Based Adaptive Traffic Signal Control

An educational college mini-project that lets a tabular Q-Learning agent interact with a small, stochastic four-way traffic simulation. It runs **in your web browser** (served from localhost by a one-line Python command) and is designed for a live demonstration and academic viva—not for real-road deployment.

> New to the project? Read [EXPLAINED.txt](EXPLAINED.txt) for a plain-English walkthrough of how everything works.

## Quick start

From the project folder:

```bash
python -m backend.app
```

Your browser opens at <http://localhost:8000>. Press `Ctrl+C` in the terminal to stop. No packages need installing for the web version (Python 3.10+ standard library only). Useful options:

```bash
python -m backend.app --port 5000     # choose a port (the next free one is used if it is busy)
python -m backend.app --no-browser    # start the server without opening a tab
```

## Problem statement

Fixed-time traffic lights use predetermined cycles even when traffic demand varies by approach. This mini-project asks whether an agent can learn which legal traffic phase to request from compact queue observations, using experience from repeated simulated interaction. The project reports measurements against a fixed-time reference and makes no assumption that Q-Learning will always win.

## Objectives and proposed solution

The simulation draws one top-down four-way intersection with a separate straight-through lane and queue for North, South, East, and West. Vehicle arrivals are stochastic and selectable at low, medium, or high density. A tabular Q-Learning agent observes queue categories plus the current phase, selects one of four directional actions, receives a weighted reward, and updates a 162 × 4 table. An environment-owned safety layer maps those actions to two legal axes and enforces all-red clearance, minimum green, and maximum green. A non-learning, predetermined fixed-time controller is included only as an evaluation reference.

## Technology stack

- **Web app (primary):** HTML5 canvas + vanilla JavaScript in a single file, [web/index.html](web/index.html). No framework, no build step, no external requests.
- **Server:** [backend/app.py](backend/app.py), a tiny Python standard-library static server (`python -m backend.app`).
- **Original Python implementation (reference):** `main.py` (Pygame window), `simulation.py`, `traffic_signal.py`, `vehicle.py`, `q_learning_agent.py`, `training.py`, `evaluation.py`, with unit tests in `tests/`. The JavaScript in `web/index.html` is a direct port of these modules.
- Python 3.10 or later (3.11 recommended). Pygame and NumPy are needed only for the desktop version and the tests.

No API, database, cloud service, prediction model, or external RL library is needed.

## Reinforcement Learning definitions

- **Agent:** the tabular Q-Learning controller.
- **Environment:** the stochastic four-approach intersection simulator.
- **State:** each approach's queue category (Low 0–2 vehicles, Medium 3–5, High 6+) plus the current phase (N–S or E–W). There are 3⁴ × 2 = 162 states.
- **Actions:** 0 = North, 1 = South, 2 = East, 3 = West. North or South requests the shared North–South green phase; East or West requests the East–West phase. Four queues remain independent, so the observation still shows directional imbalance.
- **Reward:** `r = 2.5 × vehicles served − 0.03 × waiting seconds − 0.025 × total queued vehicles − 0.8 × phase switches − 0.005 × excess starvation seconds`. An approach's longest current wait above 15 seconds contributes excess starvation. Positive throughput adds reward; waiting, congestion, switching, and starvation subtract reward. These are educational coefficients, not calibrated traffic-engineering values.

The learning update is:

```text
Q(s, a) ← Q(s, a) + α [r + γ max_a′ Q(s′, a′) − Q(s, a)]
```

`Q(s,a)` is the current return estimate; `α` controls update size; `r` is the observed reward; `γ` discounts later reward; and `max Q(s′,a′)` is the best next-state estimate. At a terminal training step, the future-value term is zero. During training, an epsilon-greedy policy explores a random action with probability `ε` and otherwise selects the action with the highest current value; `ε` decays toward its configured minimum. During evaluation, the policy is greedy (`ε = 0` for action selection); evaluation does not update the table.

## Safety and phase design

The environment permits only two green phases: N–S green with E–W red, or E–W green with N–S red. Actions are requests; the environment, not the agent, decides when a requested change can take effect. The current phase must meet a 5-second minimum, a phase is limited to 17 seconds before the environment requests the other axis, and a one-second all-red interval is inserted during axis changes. Vehicles already inside the intersection are allowed to clear during all-red; new entrants must have the green for their axis. Vehicles keep a minimum same-lane gap and cannot cross a red stop line.

## Using the web app

- **START** runs a live demonstration. In this mode the agent is not choosing; the environment alternates axes using its maximum-green rule.
- **PAUSE / RESUME** (or the space bar) freezes and resumes the current experiment. **RESET** starts a clean intersection and keeps the saved Q-table.
- **LOW / MEDIUM / HIGH** change vehicle arrival rates. **SPEED − / +** change playback speed (0.5×–4×).
- **TRAIN AGENT** trains for the chosen number of episodes (**EPISODES − / +**, default 80). The trace at the bottom plots reward per episode.
- **RUN LEARNED POLICY** drives the intersection with the greedy Q-table. **FIXED-TIME BASELINE** alternates the N–S and E–W axes every 12 seconds. Either one also starts a matched comparison: both controllers are measured for five 120-second episodes using the same seeded arrival events within each pair. The table reports mean waiting time per served vehicle, time-averaged queue per approach, vehicles served per episode, and actual phase switches.
- **CLEAR SAVED Q-TABLE** forgets everything stored in this browser.
- **Download CSV** links export the episode history and the per-episode comparison rows.

The Q-table, training history, and last comparison are stored in your browser's `localStorage`, so they survive page reloads on the same browser and address (note that `localhost:8000` and `localhost:5000` have separate storage).

## Optional: the original Pygame desktop version

```bash
pip install -r requirements.txt
python main.py
```

This opens the 1280 × 840 Pygame window and saves its table to `models/q_table.npy` and CSVs to `results/`. It stays in the repository as the reference implementation, but the browser version is smoother and is the recommended way to demo.

Run the unit tests (headless) with:

```bash
python -m unittest discover tests
```

## File organization

```text
backend/app.py          Static localhost server: python -m backend.app
web/index.html          The complete browser app (simulation, Q-learning, training, evaluation, UI)
EXPLAINED.txt           Plain-English explanation of the project
main.py                 Original Pygame interface and event/render loop
simulation.py           Vehicle arrival, queues, movement, state, rewards' inputs
vehicle.py              One vehicle's state
traffic_signal.py       Safe legal phases, minimum/maximum green, all-red clearance
q_learning_agent.py     Visible NumPy table, epsilon-greedy choice and update
training.py             Incremental training session and reward function
evaluation.py           Matched-seed evaluation and fixed-time reference
tests/test_core.py      Headless tests for the Python core
requirements.txt        Pygame and NumPy (desktop version / tests only)
models/, results/       Output folders used by the Pygame version
```

## Experimental results and limitations

The interface reports the measurements generated on your machine. There are intentionally **no claimed or pre-filled performance improvements** in this source. A reproducible seed makes each paired arrival stream comparable, but five runs remain a small sample. Results may be poor with short training, directional imbalance, high arrival rates, coarse state bins, or a reward function whose coefficients do not reflect the desired trade-off. In practice a short training run often does **not** beat the fixed-time baseline; that is an honest result, not a bug.

The browser version uses its own pseudo-random generator, so exact numbers differ from the Python version even with the same seeds; the behaviour and rules are the same.

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
