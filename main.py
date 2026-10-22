"""Pygame entry point for the educational adaptive traffic-signal lab."""
from __future__ import annotations

import asyncio
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pygame

from evaluation import MatchedComparison
from q_learning_agent import QLearningAgent
from simulation import APPROACHES, DT, QUEUE_STOP, IntersectionSimulator
from training import TrainingSession, reward_for_transition

# Pygbag (browser/WebAssembly) build: no system fonts, in-memory filesystem.
WEB = sys.platform == "emscripten"
# Glyphs missing from Pygame's built-in web font are replaced with ASCII equivalents.
ASCII_FALLBACK = str.maketrans({
    "·": "|", "–": "-", "−": "-", "▶": ">", "Ⅱ": "||", "↓": "v", "↑": "^",
    "←": "<", "→": ">", "×": "x", "ε": "e",
})


def display_text(text: str) -> str:
    return text.translate(ASCII_FALLBACK) if WEB else text


ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
WIDTH, HEIGHT = 1280, 840
FPS = 60
SPEED_LEVELS = (0.5, 1.0, 2.0, 4.0)

PAPER = (242, 243, 237)
INK = (32, 39, 34)
MUTED = (111, 122, 112)
FAINT = (154, 164, 152)
LINE = (217, 222, 213)
PANEL = (251, 252, 248)
DARK = (29, 36, 32)
ROAD = (57, 70, 62)
ROAD_DEEP = (42, 55, 47)
ROAD_LINE = (118, 132, 120)
LIME = (183, 239, 102)
GREEN = (105, 213, 140)
RED = (239, 113, 107)
AMBER = (241, 191, 85)
CAR_COLORS = [(243, 179, 110), (135, 192, 238), (233, 130, 130), (211, 201, 120), (195, 158, 226)]


def load_agent() -> QLearningAgent:
    table_path = MODELS / "q_table.npy"
    config_path = MODELS / "agent_config.json"
    if not table_path.exists():
        return QLearningAgent()
    try:
        settings = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        agent = QLearningAgent.load(table_path, **{
            key: settings[key] for key in ("alpha", "gamma", "epsilon", "epsilon_decay", "min_epsilon")
            if key in settings
        })
        agent.updates = int(settings.get("updates", 0))
        return agent
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"Could not load {table_path}: {exc}. A fresh Q-table will be used.")
        return QLearningAgent()


def save_agent(agent: QLearningAgent) -> None:
    try:
        MODELS.mkdir(exist_ok=True)
        agent.save(MODELS / "q_table.npy")
        (MODELS / "agent_config.json").write_text(json.dumps({
            "alpha": agent.alpha,
            "gamma": agent.gamma,
            "epsilon": agent.epsilon,
            "epsilon_decay": agent.epsilon_decay,
            "min_epsilon": agent.min_epsilon,
            "updates": agent.updates,
        }, indent=2), encoding="utf-8")
    except OSError as exc:
        print(f"Could not save the Q-table: {exc}")


def write_episode_history(history: list[dict[str, Any]]) -> None:
    if not history:
        return
    try:
        RESULTS.mkdir(exist_ok=True)
        path = RESULTS / "episode_history.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
    except OSError as exc:
        print(f"Could not write episode history: {exc}")


def write_comparison(comparison: MatchedComparison) -> None:
    fields = ["density", "controller", "seed", "simulation_seconds", "vehicles_served", "average_wait", "average_queue", "switches", "reward"]
    try:
        RESULTS.mkdir(exist_ok=True)
        path = RESULTS / "comparison.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for controller in ("rl", "fixed"):
                for record in comparison.records[controller]:
                    writer.writerow({
                        "density": comparison.density,
                        "controller": controller,
                        "seed": record["seed"],
                        "simulation_seconds": record["simulation_seconds"],
                        "vehicles_served": record["vehicles_served"],
                        "average_wait": record["average_wait"],
                        "average_queue": record["average_queue"],
                        "switches": record["switches"],
                        "reward": record["reward"],
                    })
    except OSError as exc:
        print(f"Could not write comparison results: {exc}")


def font(size: int, mono: bool = False, bold: bool = False) -> pygame.font.Font:
    if WEB:
        return pygame.font.Font(None, size + 6)
    candidates = ("consolas", "dejavusansmono", "monospace") if mono else ("segoeui", "arial", "dejavusans")
    for name in candidates:
        candidate = pygame.font.SysFont(name, size, bold=bold)
        if candidate:
            return candidate
    return pygame.font.Font(None, size)


class Button:
    def __init__(self, label: str, rect: pygame.Rect, key: str, style: str = "outline") -> None:
        self.label = label
        self.rect = rect
        self.key = key
        self.style = style
        self.enabled = True
        self.selected = False

    def draw(self, surface: pygame.Surface, text_font: pygame.font.Font) -> None:
        if self.style == "primary":
            fill, border, text_color = DARK, DARK, (248, 250, 244)
        elif self.selected:
            fill, border, text_color = (235, 243, 222), (173, 201, 139), (62, 86, 48)
        else:
            fill, border, text_color = PANEL, LINE, (57, 68, 59)
        if not self.enabled:
            fill, text_color = (237, 239, 232), (151, 158, 150)
        pygame.draw.rect(surface, fill, self.rect, border_radius=5)
        pygame.draw.rect(surface, border, self.rect, width=1, border_radius=5)
        label = text_font.render(display_text(self.label), True, text_color)
        surface.blit(label, label.get_rect(center=self.rect.center))


class TrafficLabApp:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Adaptive Traffic Lab · Tabular Q-Learning")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.fonts = {
            "title": font(21, bold=True), "heading": font(15, bold=True),
            "body": font(12), "small": font(10), "tiny": font(8, mono=True),
            "mono": font(12, mono=True), "metric": font(23, mono=True),
            "big": font(42, bold=True), "equation": font(10, mono=True),
            "button": font(10, bold=True),
        }
        self.agent = load_agent()
        self.sim = IntersectionSimulator(density="medium", seed=18421)
        self.training: TrainingSession | None = None
        self.comparison: MatchedComparison | None = None
        self.mode = "READY"
        self.is_running = False
        self.paused = False
        self.density = "medium"
        self.speed_index = 1
        self.episode_count = 80
        self.reward_history: list[dict[str, Any]] = self._read_history()
        self.last_comparison: dict[str, Any] | None = self._read_comparison()
        self.accumulator = 0.0
        self.decision_timer = 0.0
        self.live_reward = 0.0
        self.last_action = 0
        self.message = "Start a live demo, train Q-Learning, or compare both controllers."
        self.message_until = 0.0
        self.buttons = self._make_buttons()

    def _read_history(self) -> list[dict[str, Any]]:
        path = RESULTS / "episode_history.csv"
        if not path.exists():
            return []
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))[-240:]
            for row in rows:
                for key in ("episode", "reward", "average_wait", "average_queue", "vehicles_served", "switches", "epsilon"):
                    row[key] = float(row[key])
            return rows
        except (OSError, ValueError, KeyError):
            return []

    def _read_comparison(self) -> dict[str, Any] | None:
        path = RESULTS / "comparison.csv"
        if not path.exists():
            return None
        try:
            rows = list(csv.DictReader(path.open(encoding="utf-8")))
            summary: dict[str, Any] = {}
            for controller in ("rl", "fixed"):
                group = [row for row in rows if row.get("controller") == controller]
                if not group:
                    return None
                summary[controller] = {
                    key: sum(float(row[key]) for row in group) / len(group)
                    for key in ("average_wait", "average_queue", "vehicles_served", "switches", "reward")
                }
            summary["seeds"] = min(len([row for row in rows if row.get("controller") == c]) for c in ("rl", "fixed"))
            summary["density"] = rows[0].get("density", "medium")
            return summary
        except (OSError, ValueError, KeyError):
            return None

    def _make_buttons(self) -> list[Button]:
        x, width, gap = 674, 586, 9
        small_font = self.fonts["button"]
        buttons: list[Button] = []
        row_y = 466
        widths = (134, 132, 82)
        px = x
        for label, button_width, key, style in (
            ("▶  START", widths[0], "start", "primary"),
            ("Ⅱ  PAUSE", widths[1], "pause", "outline"),
            ("RESET", widths[2], "reset", "outline"),
        ):
            buttons.append(Button(label, pygame.Rect(px, row_y, button_width, 32), key, style))
            px += button_width + gap
        control_width = (width - 2 * gap) // 3
        for index, (label, key) in enumerate((("TRAIN AGENT", "train"), ("RUN LEARNED POLICY", "policy"), ("FIXED-TIME BASELINE", "baseline"))):
            buttons.append(Button(label, pygame.Rect(x + index * (control_width + gap), 507, control_width, 31), key, "outline"))
        density_y = 551
        density_width = 85
        for index, name in enumerate(("low", "medium", "high")):
            buttons.append(Button(name.upper(), pygame.Rect(x + index * (density_width + 7), density_y, density_width, 27), f"density:{name}", "outline"))
        buttons.append(Button("SPEED −", pygame.Rect(x + 3 * (density_width + 7) + 5, density_y, 81, 27), "slower", "outline"))
        buttons.append(Button("SPEED +", pygame.Rect(x + 3 * (density_width + 7) + 91, density_y, 81, 27), "faster", "outline"))
        buttons.append(Button("EPISODES −", pygame.Rect(x, 591, 87, 27), "episodes:-", "outline"))
        buttons.append(Button("EPISODES +", pygame.Rect(x + 94, 591, 87, 27), "episodes:+", "outline"))
        # A small status chip shares the remainder of the episode-control row.
        return buttons

    def _say(self, message: str, seconds: float = 5.0) -> None:
        self.message = message
        self.message_until = time.monotonic() + seconds

    def _set_mode(self, mode: str) -> None:
        self.mode = mode
        self.paused = False

    def _new_sim(self) -> IntersectionSimulator:
        return IntersectionSimulator(density=self.density, seed=int(time.time_ns() % 2**31))

    def start_demo(self) -> None:
        self.training = None
        self.comparison = None
        self.sim = self._new_sim()
        self.live_reward = 0.0
        self.last_action = 0
        self.decision_timer = 0.0
        self.is_running = True
        self._set_mode("INTERACTIVE DEMO")
        self._say("Live demonstration running. The environment controls all safe phase changes.")

    def start_training(self) -> None:
        self.comparison = None
        training = TrainingSession(
            self.agent,
            episodes=self.episode_count,
            density=self.density,
            seed=60213,
        )
        self.training = training
        self.sim = training.sim
        self.is_running = True
        self._set_mode("TRAINING")
        self.reward_history = []
        write_episode_history([])
        self._say(f"Training {self.episode_count} episodes incrementally; the Pygame event loop remains responsive.")

    def start_policy(self, baseline: bool = False) -> None:
        self.training = None
        self.sim = self._new_sim()
        self.live_reward = 0.0
        self.decision_timer = 0.0
        self.last_action = 0
        self.is_running = True
        self._set_mode("FIXED-TIME BASELINE" if baseline else "RL EVALUATION")
        self.comparison = MatchedComparison(self.agent, density=self.density, seeds=5)
        if baseline:
            self._say("Fixed-time controller running; paired 5-seed comparison runs incrementally alongside it.")
        elif self.agent.updates == 0:
            self._say("The Q-table has no recorded updates. Greedy evaluation will run, but train first for a learned policy.")
        else:
            self._say("Greedy learned-policy evaluation is active. The comparison also measures fixed timing on matched seeds.")

    def reset(self) -> None:
        self.is_running = False
        self.training = None
        self.comparison = None
        self.sim = self._new_sim()
        self.live_reward = 0.0
        self.accumulator = 0.0
        self.decision_timer = 0.0
        self._set_mode("READY")
        self._say("Simulation reset. The local Q-table and previous measured results are kept.")

    def handle_button(self, key: str) -> None:
        if key == "start":
            self.start_demo()
        elif key == "pause":
            if self.is_running:
                self.is_running = False
                self.paused = True
                self._say("Paused. Resume with START or begin another experiment.")
            elif self.paused:
                self.is_running = True
                self.paused = False
                self._say("Experiment resumed.")
        elif key == "reset":
            self.reset()
        elif key == "train":
            self.start_training()
        elif key == "policy":
            self.start_policy(False)
        elif key == "baseline":
            self.start_policy(True)
        elif key.startswith("density:"):
            self.density = key.split(":", 1)[1]
            self.sim.set_density(self.density)
            if self.training and self.is_running:
                self.training.set_density(self.density)
            self._say(f"Traffic density set to {self.density}.")
        elif key == "slower":
            self.speed_index = max(0, self.speed_index - 1)
        elif key == "faster":
            self.speed_index = min(len(SPEED_LEVELS) - 1, self.speed_index + 1)
        elif key == "episodes:-":
            self.episode_count = max(10, self.episode_count - 10)
        elif key == "episodes:+":
            self.episode_count = min(1000, self.episode_count + 10)

    def _run_live_step(self) -> None:
        if self.mode == "RL EVALUATION":
            action = self.agent.select_action(self.sim.state, explore=False)
            if action != self.last_action or self.decision_timer == 0:
                self.sim.request_action(action)
                self.last_action = action
        elif self.mode == "FIXED-TIME BASELINE":
            action = 0 if int(self.sim.time // 12) % 2 == 0 else 2
            if action != self.last_action or self.decision_timer == 0:
                self.sim.request_action(action)
                self.last_action = action
        outcome = self.sim.step(DT)
        self.live_reward += reward_for_transition(outcome)
        self.sim.cumulative_reward = self.live_reward
        if self.mode in ("RL EVALUATION", "FIXED-TIME BASELINE"):
            self.decision_timer += DT
            if self.decision_timer >= 1.0:
                self.decision_timer = 0.0

    def _advance(self, frame_seconds: float) -> None:
        if not self.is_running:
            return
        if self.mode == "TRAINING" and self.training:
            finished = self.training.advance_batch(1 + self.speed_index)
            self.sim = self.training.sim
            for entry in finished:
                self.reward_history.append(entry)
                save_agent(self.agent)
                write_episode_history(self.reward_history)
            if self.training.complete:
                self.is_running = False
                self.sim = self.training.sim
                self._say(f"Training complete: {self.training.completed_episodes} episodes, {self.agent.updates:,} Q updates. Table saved in models/q_table.npy.", 12.0)
        else:
            multiplier = SPEED_LEVELS[self.speed_index]
            self.accumulator += min(frame_seconds, 0.1) * multiplier
            safety = 0
            while self.accumulator >= DT and safety < 24:
                self._run_live_step()
                self.accumulator -= DT
                safety += 1
            if self.comparison and not self.comparison.complete:
                self.comparison.advance_batch(3)
                if self.comparison.complete:
                    self.last_comparison = self.comparison.summary()
                    write_comparison(self.comparison)
                    self._say("Five matched-seed episodes per controller are measured. Results are shown in the table and results/comparison.csv.", 12.0)

    def _draw_text(self, text: str, x: int, y: int, color: tuple[int, int, int] = INK, key: str = "body", align: str = "topleft") -> pygame.Rect:
        image = self.fonts[key].render(display_text(str(text)), True, color)
        rect = image.get_rect()
        setattr(rect, align, (x, y))
        self.screen.blit(image, rect)
        return rect

    def _draw_panel(self, rect: pygame.Rect, fill: tuple[int, int, int] = PANEL, radius: int = 6, border: tuple[int, int, int] = LINE) -> None:
        pygame.draw.rect(self.screen, fill, rect, border_radius=radius)
        pygame.draw.rect(self.screen, border, rect, width=1, border_radius=radius)

    def _draw_intersection(self) -> None:
        board = pygame.Rect(25, 87, 610, 610)
        pygame.draw.rect(self.screen, (43, 55, 48), board, border_radius=5)
        # Technical grid in the surrounding ground.
        for x in range(board.left + 12, board.right - 8, 27):
            for y in range(board.top + 12, board.bottom - 8, 27):
                pygame.draw.circle(self.screen, (62, 75, 66), (x, y), 1)
        cx, cy, half = board.centerx, board.centery, 88
        pygame.draw.rect(self.screen, ROAD_DEEP, (cx - half - 10, board.top, 2 * (half + 10), board.height))
        pygame.draw.rect(self.screen, ROAD_DEEP, (board.left, cy - half - 10, board.width, 2 * (half + 10)))
        pygame.draw.rect(self.screen, ROAD, (cx - half, board.top, 2 * half, board.height))
        pygame.draw.rect(self.screen, ROAD, (board.left, cy - half, board.width, 2 * half))
        pygame.draw.rect(self.screen, (64, 79, 69), (cx - half, cy - half, 2 * half, 2 * half))
        # Road edges are solid; lane centre marks are dashed and never run through the box.
        for offset in (-half, half):
            pygame.draw.line(self.screen, (135, 148, 136), (cx + offset, board.top), (cx + offset, cy - half - 10), 1)
            pygame.draw.line(self.screen, (135, 148, 136), (cx + offset, cy + half + 10), (cx + offset, board.bottom), 1)
            pygame.draw.line(self.screen, (135, 148, 136), (board.left, cy + offset), (cx - half - 10, cy + offset), 1)
            pygame.draw.line(self.screen, (135, 148, 136), (cx + half + 10, cy + offset), (board.right, cy + offset), 1)
        for top, bottom in ((board.top, cy - half - 10), (cy + half + 10, board.bottom)):
            y = top
            while y < bottom:
                pygame.draw.line(self.screen, ROAD_LINE, (cx, y), (cx, min(y + 12, bottom)), 1)
                y += 23
        for left, right in ((board.left, cx - half - 10), (cx + half + 10, board.right)):
            x = left
            while x < right:
                pygame.draw.line(self.screen, ROAD_LINE, (x, cy), (min(x + 12, right), cy), 1)
                x += 23

        extent = board.width * 0.619
        stop_n = cy - extent + 2 * extent * QUEUE_STOP
        stop_s = cy + extent - 2 * extent * QUEUE_STOP
        stop_w = cx - extent + 2 * extent * QUEUE_STOP
        stop_e = cx + extent - 2 * extent * QUEUE_STOP
        pygame.draw.line(self.screen, (245, 246, 235), (cx - half, int(stop_n)), (cx, int(stop_n)), 4)
        pygame.draw.line(self.screen, (245, 246, 235), (cx, int(stop_s)), (cx + half, int(stop_s)), 4)
        pygame.draw.line(self.screen, (245, 246, 235), (int(stop_w), cy), (int(stop_w), cy + half), 4)
        pygame.draw.line(self.screen, (245, 246, 235), (int(stop_e), cy - half), (int(stop_e), cy), 4)

        signal_points = {
            "north": (cx - half - 16, int(stop_n) - 17),
            "south": (cx + half + 16, int(stop_s) + 17),
            "west": (int(stop_w) - 20, cy + half + 17),
            "east": (int(stop_e) + 20, cy - half - 17),
        }
        for direction, (sx, sy) in signal_points.items():
            pygame.draw.rect(self.screen, (22, 30, 25), (sx - 8, sy - 17, 16, 34), border_radius=6)
            active = GREEN if self.sim.signal.is_green(direction) else (AMBER if self.sim.signal.mode == "all-red" else RED)
            for index, color in enumerate((RED, AMBER, GREEN)):
                is_on = (self.sim.signal.mode == "all-red" and index == 1) or (self.sim.signal.mode == "green" and self.sim.signal.is_green(direction) and index == 2) or (self.sim.signal.mode == "green" and not self.sim.signal.is_green(direction) and index == 0)
                pygame.draw.circle(self.screen, color if is_on else (68, 80, 71), (sx, sy - 10 + index * 10), 3)
                if is_on and color == active:
                    pygame.draw.circle(self.screen, color, (sx, sy - 10 + index * 10), 5, 1)

        def path_coord(progress: float, reverse: bool = False) -> float:
            return (board.centerx + extent - 2 * extent * progress) if reverse else (board.centerx - extent + 2 * extent * progress)

        def path_y(progress: float, reverse: bool = False) -> float:
            return (board.centery + extent - 2 * extent * progress) if reverse else (board.centery - extent + 2 * extent * progress)

        for direction in APPROACHES:
            for car in self.sim.lanes[direction]:
                if direction == "north":
                    px, py, horizontal = cx - 38, path_y(car.progress), False
                elif direction == "south":
                    px, py, horizontal = cx + 38, path_y(car.progress, True), False
                elif direction == "west":
                    px, py, horizontal = path_coord(car.progress), cy + 38, True
                else:
                    px, py, horizontal = path_coord(car.progress, True), cy - 38, True
                if not (board.left - 30 <= px <= board.right + 30 and board.top - 30 <= py <= board.bottom + 30):
                    continue
                rect = pygame.Rect(0, 0, 27, 15) if horizontal else pygame.Rect(0, 0, 15, 27)
                rect.center = (round(px), round(py))
                color = CAR_COLORS[car.color_index % len(CAR_COLORS)]
                pygame.draw.rect(self.screen, color, rect, border_radius=5)
                windshield = pygame.Rect(0, 0, 8, 10) if horizontal else pygame.Rect(0, 0, 10, 8)
                windshield.center = rect.center
                pygame.draw.rect(self.screen, (74, 92, 81), windshield, border_radius=3)
                if horizontal:
                    pygame.draw.circle(self.screen, (238, 239, 222), (rect.left + 4, rect.centery), 2)
                else:
                    pygame.draw.circle(self.screen, (238, 239, 222), (rect.centerx, rect.top + 4), 2)

        phase_text = "ALL RED" if self.sim.signal.mode == "all-red" else self.sim.signal.phase
        badge = pygame.Rect(cx - 43, cy - 25, 86, 50)
        pygame.draw.rect(self.screen, (35, 47, 39), badge, border_radius=5)
        pygame.draw.rect(self.screen, (113, 132, 115), badge, width=1, border_radius=5)
        self._draw_text("ACTIVE PHASE", cx, cy - 9, (195, 208, 194), "tiny", "center")
        self._draw_text(phase_text, cx, cy + 9, AMBER if self.sim.signal.mode == "all-red" else LIME, "mono", "center")
        self._draw_text("N", cx, board.top + 12, (202, 211, 198), "tiny", "center")
        self._draw_text("S", cx, board.bottom - 12, (202, 211, 198), "tiny", "center")
        self._draw_text("W", board.left + 12, cy, (202, 211, 198), "tiny", "center")
        self._draw_text("E", board.right - 12, cy, (202, 211, 198), "tiny", "center")
        self._draw_text("TOP-DOWN SIMULATION · STRAIGHT-THROUGH MOVEMENTS", board.left + 1, board.bottom + 13, MUTED, "tiny")
        self._draw_text("SIGNAL SAFETY LAYER ACTIVE", board.right - 1, board.bottom + 13, MUTED, "tiny", "topright")

    def _draw_metric_card(self, rect: pygame.Rect, label: str, value: str, foot: str, accent: bool = False) -> None:
        self._draw_panel(rect, fill=(239, 245, 228) if accent else PANEL)
        self._draw_text(label, rect.x + 11, rect.y + 9, MUTED, "tiny")
        self._draw_text(value, rect.x + 11, rect.y + 26, (67, 101, 48) if accent else INK, "metric")
        self._draw_text(foot, rect.x + 11, rect.bottom - 12, FAINT, "tiny")

    def _draw_queues(self, rect: pygame.Rect) -> None:
        self._draw_panel(rect, fill=(247, 249, 242))
        self._draw_text("APPROACH QUEUES", rect.x + 12, rect.y + 10, MUTED, "tiny")
        self._draw_text("LOW 0–2  ·  MED 3–5  ·  HIGH 6+", rect.right - 12, rect.y + 10, FAINT, "tiny", "topright")
        labels = (("NORTH", "↓"), ("SOUTH", "↑"), ("EAST", "←"), ("WEST", "→"))
        queues = self.sim.queue_by_direction()
        for index, (name, arrow) in enumerate(labels):
            bx = rect.x + 12 + index * 141
            self._draw_text(arrow, bx, rect.y + 39, (120, 150, 84), "body")
            self._draw_text(name, bx + 18, rect.y + 41, MUTED, "tiny")
            self._draw_text(str(queues[name.lower()]), bx + 102, rect.y + 37, INK, "mono", "topright")
            track = pygame.Rect(bx + 18, rect.y + 57, 90, 4)
            pygame.draw.rect(self.screen, (225, 230, 220), track, border_radius=3)
            fill_width = int(track.width * min(1.0, queues[name.lower()] / 10))
            pygame.draw.rect(self.screen, LIME if queues[name.lower()] <= 2 else AMBER if queues[name.lower()] <= 5 else RED, (track.x, track.y, fill_width, track.height), border_radius=3)

    def _draw_controls(self) -> None:
        panel = pygame.Rect(662, 446, 594, 185)
        self._draw_panel(panel)
        self._draw_text("CONTROL DESK", panel.x + 13, panel.y + 11, MUTED, "tiny")
        status = f"{self.mode}{' · PAUSED' if self.paused else ''}"
        dot = GREEN if self.is_running else FAINT
        pygame.draw.circle(self.screen, dot, (panel.right - 112, panel.y + 16), 4)
        self._draw_text(status[:23], panel.right - 101, panel.y + 16, MUTED, "tiny", "midleft")

        for button in self.buttons:
            button.enabled = True
            button.selected = False
            if button.key == "pause":
                button.label = "▶  RESUME" if self.paused else "Ⅱ  PAUSE"
            if button.key.startswith("density:"):
                button.selected = self.density == button.key.split(":")[1]
            if button.key in ("train", "policy", "baseline") and self.is_running:
                button.enabled = False
            if button.key == "pause" and not (self.is_running or self.paused):
                button.enabled = False
            if button.key == "start" and self.is_running:
                button.enabled = False
            if button.key == "episodes:-":
                button.label = "EPISODES −"
            if button.key == "episodes:+":
                button.label = "EPISODES +"
            button.draw(self.screen, self.fonts["button"])
        self._draw_text(f"{self.episode_count} EPISODES", panel.x + 202, panel.y + 156, (79, 91, 76), "tiny")
        self._draw_text(f"{SPEED_LEVELS[self.speed_index]:g}×", panel.x + 530, panel.y + 155, (79, 91, 76), "tiny", "topright")
        self._draw_text(self.message[:92], panel.x + 12, panel.bottom - 10, MUTED, "tiny", "bottomleft")

    def _draw_comparison(self) -> None:
        rect = pygame.Rect(662, 640, 594, 84)
        self._draw_panel(rect, fill=(246, 248, 241))
        self._draw_text("MEASURED COMPARISON", rect.x + 12, rect.y + 9, MUTED, "tiny")
        if self.comparison and not self.comparison.complete:
            label = f"RUN {self.comparison.completed_runs + 1}/10 · {self.comparison.controller_name.upper()} · SEED {self.comparison.seed_index + 1}/5"
            self._draw_text(label, rect.right - 12, rect.y + 9, (102, 132, 75), "tiny", "topright")
        elif self.last_comparison:
            self._draw_text(f"{self.last_comparison['seeds']} MATCHED SEEDS · {self.last_comparison['density'].upper()}", rect.right - 12, rect.y + 9, (102, 132, 75), "tiny", "topright")
        else:
            self._draw_text("NO RUN YET", rect.right - 12, rect.y + 9, FAINT, "tiny", "topright")
        col_x = (rect.x + 12, rect.x + 174, rect.x + 300, rect.x + 427)
        self._draw_text("CONTROLLER", col_x[0], rect.y + 32, FAINT, "tiny")
        self._draw_text("WAIT · s", col_x[1], rect.y + 32, FAINT, "tiny")
        self._draw_text("QUEUE / ARM", col_x[2], rect.y + 32, FAINT, "tiny")
        self._draw_text("SERVED · SW", col_x[3], rect.y + 32, FAINT, "tiny")
        if self.last_comparison:
            for row, key, label in ((0, "rl", "TABULAR Q-LEARNING"), (1, "fixed", "FIXED TIME")):
                y = rect.y + 49 + row * 16
                item = self.last_comparison[key]
                color = (83, 120, 55) if key == "rl" else INK
                self._draw_text(label, col_x[0], y, color, "tiny")
                self._draw_text(f"{item['average_wait']:.1f}", col_x[1], y, INK, "tiny")
                self._draw_text(f"{item['average_queue']:.2f}", col_x[2], y, INK, "tiny")
                self._draw_text(f"{item['vehicles_served']:.1f} · {item['switches']:.1f}", col_x[3], y, INK, "tiny")
        else:
            self._draw_text("Run RL policy or fixed baseline to collect measured values.", rect.x + 12, rect.y + 58, FAINT, "tiny")

    def _draw_training_chart(self) -> None:
        rect = pygame.Rect(662, 733, 594, 89)
        self._draw_panel(rect, fill=(232, 235, 227))
        self._draw_text("TRAINING TRACE · CUMULATIVE REWARD BY EPISODE", rect.x + 12, rect.y + 9, MUTED, "tiny")
        episode = self.training.completed_episodes if self.training else len(self.reward_history)
        target = self.training.episode_target if self.training else self.episode_count
        self._draw_text(f"{episode} / {target}", rect.right - 12, rect.y + 9, MUTED, "tiny", "topright")
        plot = pygame.Rect(rect.x + 38, rect.y + 28, rect.width - 52, 45)
        for frac in (0.0, 0.5, 1.0):
            y = plot.y + int(frac * plot.height)
            pygame.draw.line(self.screen, (211, 218, 207), (plot.x, y), (plot.right, y), 1)
        records = self.training.history if self.training else self.reward_history
        values = [float(item["reward"]) for item in records[-90:]]
        if values:
            smooth = [sum(values[max(0, i - 5): i + 1]) / len(values[max(0, i - 5): i + 1]) for i in range(len(values))]
            low, high = min(0.0, min(smooth)), max(0.0, max(smooth))
            if high - low < 1:
                high += 1.0; low -= 1.0
            points = []
            for i, value in enumerate(smooth):
                px = plot.x + (plot.width - 1) * i / max(1, len(smooth) - 1)
                py = plot.bottom - (value - low) / (high - low) * plot.height
                points.append((round(px), round(py)))
            if len(points) == 1:
                pygame.draw.circle(self.screen, (112, 148, 70), points[0], 3)
            else:
                pygame.draw.lines(self.screen, (112, 148, 70), False, points, 2)
            self._draw_text("MEASURED · NO CONVERGENCE CLAIM", rect.x + 12, rect.bottom - 8, (123, 135, 117), "tiny", "bottomleft")
        else:
            self._draw_text("No episodes yet. Train the agent to record reward.", plot.x, plot.centery, (139, 150, 136), "tiny", "midleft")

    def draw(self) -> None:
        self.screen.fill(PAPER)
        pygame.draw.line(self.screen, LINE, (22, 60), (WIDTH - 22, 60), 1)
        # Wordmark inspired by the four-way intersection.
        icon_x, icon_y = 26, 15
        pygame.draw.rect(self.screen, (61, 74, 63), (icon_x + 10, icon_y, 8, 28), border_radius=2)
        pygame.draw.rect(self.screen, (61, 74, 63), (icon_x, icon_y + 10, 28, 8), border_radius=2)
        pygame.draw.circle(self.screen, PAPER, (icon_x + 14, icon_y + 14), 4)
        pygame.draw.circle(self.screen, LIME, (icon_x + 14, icon_y + 3), 2)
        self._draw_text("Adaptive Traffic", 65, 19, INK, "heading")
        self._draw_text("LAB", 169, 22, MUTED, "small")
        self._draw_text("REINFORCEMENT LEARNING  ·  EDUCATIONAL SIMULATION", WIDTH - 24, 28, MUTED, "tiny", "midright")

        self._draw_text("FIELD STUDY 01   /   TABULAR Q-LEARNING", 27, 76, MUTED, "tiny")
        self._draw_text("Live intersection", 27, 96, INK, "title")
        phase = self.sim.signal.label
        phase_color = AMBER if self.sim.signal.mode == "all-red" else GREEN
        pygame.draw.circle(self.screen, phase_color, (608, 106), 5)
        self._draw_text(phase, 598, 107, (66, 83, 69), "tiny", "midright")
        self._draw_intersection()
        self._draw_text(f"STATE {self.sim.state:03d} / 161    ACTIONS  N · S · E · W    |    {self.density.upper()} TRAFFIC", 28, 735, MUTED, "tiny")

        panel_x, panel_width = 662, 594
        self._draw_text("INSTRUMENT PANEL", panel_x, 82, MUTED, "tiny")
        self._draw_text("Live readout", panel_x, 97, INK, "title")
        phase_name = self.sim.signal.label
        phase_chip = pygame.Rect(panel_x + 407, 88, 187, 28)
        self._draw_panel(phase_chip, fill=(235, 243, 224) if self.sim.signal.mode == "green" else (247, 239, 218), border=(216, 226, 206))
        pygame.draw.circle(self.screen, phase_color, (phase_chip.x + 13, phase_chip.centery), 4)
        self._draw_text(phase_name, phase_chip.x + 26, phase_chip.centery, (65, 89, 54), "tiny", "midleft")

        metrics = self.sim.metrics()
        elapsed_text = f"{int(self.sim.time // 60):02d}:{int(self.sim.time % 60):02d}"
        values = [
            ("IN SYSTEM", f"{metrics['vehicles_in_system']:02d}", "vehicles on approach", False),
            ("SERVED", f"{metrics['vehicles_served']:02d}", "cleared intersection", True),
            ("AVG WAIT", f"{metrics['average_wait']:.1f}s", "per vehicle served", False),
            ("AVG QUEUE", f"{metrics['average_queue']:.1f}", "vehicles / approach", False),
            ("REWARD", f"{self.sim.cumulative_reward:.2f}", f"sim clock {elapsed_text}", False),
            ("EXPLORATION · ε", f"{self.agent.epsilon:.2f}", f"{self.agent.updates:,} Q updates", False),
        ]
        card_width, card_height, gap = 290, 60, 12
        for index, (label, value, foot, accent) in enumerate(values):
            x = panel_x + (index % 2) * (card_width + gap)
            y = 130 + (index // 2) * (card_height + 8)
            self._draw_metric_card(pygame.Rect(x, y, card_width, card_height), label, value, foot, accent)

        self._draw_queues(pygame.Rect(panel_x, 334, panel_width, 100))
        self._draw_controls()
        self._draw_comparison()
        self._draw_training_chart()

        self._draw_text("Tabular Q-Learning only  ·  not a real-world traffic controller", 23, HEIGHT - 8, MUTED, "tiny", "bottomleft")
        self._draw_text("WINDOWS · PYTHON 3 · PYGAME · NUMPY", WIDTH - 23, HEIGHT - 8, MUTED, "tiny", "bottomright")
        pygame.display.flip()

    async def run(self) -> None:
        running = True
        while running:
            frame_seconds = self.clock.tick(FPS) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for button in self.buttons:
                        if button.enabled and button.rect.collidepoint(event.pos):
                            self.handle_button(button.key)
                            break
            self._advance(frame_seconds)
            self.draw()
            await asyncio.sleep(0)  # yield to the browser each frame (required by Pygbag)
        pygame.quit()


async def main() -> None:
    app = TrafficLabApp()
    await app.run()


if __name__ == "__main__":
    asyncio.run(main())
