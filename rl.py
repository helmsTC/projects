

adjustcal.py
1,579 lines

py


api.py
639 lines

py


generate_paths.py
71 lines

py


launch.py
239 lines

py


pairwise.py
523 lines

py


render.py
115 lines

py

my system is designed to get seed plans from my launch.py/generate_paths function that can be used by the rest of the system as initial plans for an RL algo to use as a seed route that it will observe and then create a set of preferences for as an action that will optimize the initial plan. from there two plans [seed_route, rl] should flow through the system to evaluation where one is selected as the best plan and the resulting reward is handed back to the RL. Im not sure the data flow is exactly correct so please read through the code and see if it is set up correctly. i want to ensure that one plan in = one rl plan and that the system doesnt start spiraling out of control with extra plans. maybe we can have a n_optimizations to set some amount of desired rl plans for one seed, but default it to one. we need to somehow modify this to enable the agent to reset() each time we get a new seed route, as it needs to see the routeless image so it know its actions impact right? I have a render script that im currently using, but it will need to be adapted to handle that. I dont want seed routes generating unless an option has been turned on for that, we want to only work with routes that are made available by the planner


Diagnosed RL system flow, proposed reset and optimization controls.
28s



Looking at your system, I can see several issues with the data flow and RL agent management. Here's a comprehensive analysis and solution:

Key Issues Identified:
RL Agent Not Resetting Between Seed Routes: The agent needs to see a route-less image when starting a new seed route optimization
Uncontrolled Plan Generation: The system can spiral with excessive RL plans per seed
Incorrect Comparison Logic: System compares unrelated routes instead of seed vs its RL optimization
Parent-Child Tracking: No clear tracking of which RL plans belong to which seed routes
Here's the corrected implementation:

Strategized API modifications to control system optimization flow



Strategized render.py modifications for reset functionality.



Pondered system flow and optimization strategy for route planning.



System Architecture Summary and Key Changes
Document 



import itertools
import asyncio
import warnings
import os
import sys
import hashlib
import msgpack
from typing import Dict, Any, Optional, Tuple, List, Set, TypedDict
from collections import defaultdict
import logging
from functools import lru_cache
import shelve
import torch
torch.autograd.set_detect_anomaly(True)
from fastapi import HTTPException
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from scipy.optimize import linprog, OptimizeWarning, minimize
import numpy as np
os.environ['RAY_LOG_LEVEL'] = 'DEBUG'
from box._samplers import Pairwise
# planning assistant imports
from .path.planning_assistant.utils import render
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.sample_batch import SampleBatch
# plan evaluator imports
from .path.plan_evaluator.plan_evaluator import PlanEvaluator
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import uuid, queue, threading, concurrent.futures, time
import cv2
import functools
from torch.utils.tensorboard import SummaryWriter
import random
import math
from ray.rllib.policy.sample_batch import concat_samples
import pickle

# threading issues with ray initialize
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "1")

def _build_ppo():
    return (PPOConfig()
            .api_stack(enable_env_runner_and_connector_v2=False, enable_rl_module_and_learner=False)
            .environment(env=PathEnv, env_config={"render_enabled": False})
            .framework("torch")
            .training(lr=1e-4, vf_clip_param=100.0, grad_clip=40)
            .build())

class PathEnv(gym.Env):
    """Env returns previous obs/reward while a new render is prepared externally."""
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, env_config=None):
        super().__init__()
        cfg = env_config or {}
        self.render_enabled = cfg.get("render_enabled", False)
        self.observation_space = spaces.Box(0, 255, (84, 84, 3), np.uint8)
        self.action_space = spaces.Box(0.0, 1.0, (4,), np.float32)

        self._last_obs = np.zeros((84, 84, 3), np.uint8)
        self._last_reward = 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._last_obs[:] = np.zeros((84, 84, 3), np.uint8)
        self._last_reward = 0.0
        return self._last_obs.copy(), {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        obs, rew = self._last_obs.copy(), self._last_reward
        self._last_reward = 0.0
        info = {}
        return obs, rew, False, False, info

    def deliver(self, img: np.ndarray, reward: float):
        img = cv2.resize(img, (84, 84)) if img.shape[:2] != (84, 84) else img
        self._last_obs = img.astype(np.uint8)
        self._last_reward = float(reward)
        print(f"Training with reward {self._last_reward}")

    def render(self):
        if self.render_enabled:
            cv2.imshow("PathEnv", cv2.cvtColor(self._last_obs, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
        return self._last_obs.copy()

    def close(self):
        if self.render_enabled:
            cv2.destroyAllWindows()

ArrayLike = np.ndarray
Features = List[float]
JSON = Dict[str, Any]
NodeIDs = List[int]
Path = JSON

logger = logging.getLogger(__name__)
DEBUG = int(os.environ.get("box_DEBUG", "1"))
logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)
handler = logging.StreamHandler(stream=sys.stdout)
fmt = logging.Formatter("%(message)s")
handler.setFormatter(fmt)
logger.addHandler(handler)

class AdjustCAL(Pairwise):
    can_extend = True
    
    def __init__(self, name, *, designer: "Planner", n_paths: int = 2, eps: float = 1, 
                 prefs_init=None, n_optimizations_per_seed: int = 1):
        """
        Parameters
        ----------
        n_paths : int
            The number of paths to compute at each iteration.
        eps : float
            The maximum difference in the feature space to consider a query informative.
        n_optimizations_per_seed : int
            Number of RL optimizations to create per seed route (default: 1)
        """
        self.name = name
        self.designer = designer
        self.d = self.designer.n_features - 1
        self.n_paths = n_paths
        self.eps = eps
        self.prefs_init = prefs_init
        self.n_optimizations_per_seed = n_optimizations_per_seed
        
        # Core data structures
        self.prefs_ = None
        self.plans_: List[Dict[str, Any]] = []
        self.features_: List[ArrayLike] = []
        self.plan_prefs_: List[ArrayLike] = []
        self.asked_: Set[Tuple[int, int]] = set()
        self.idents_: List[str] = []
        self.update_count = 0
        
        # Tracking seed routes and their RL children
        self.seed_to_rl_map: Dict[str, List[str]] = defaultdict(list)
        self.current_seed_route: Optional[str] = None
        self.seed_route_queue: List[str] = []
        
        # Initialize Ray
        ray.init(num_cpus=4, include_dashboard=False)
        
        # Path specific objects
        self.renderer = render.Render()
        self.plan_evaluator = PlanEvaluator(renderer=self.renderer)
        self.current_action_id = None
        self.request_number = 0
        self.batches: List[List[int]] = []
        self.pair_cache: List[Tuple[int, int]] = []
        self.writer = SummaryWriter(log_dir='runs/ppo')
        self.replay: list[SampleBatch] = []
        self._step = 0
        self.reset_image = self.renderer.render_route(draw_route=False)
        
        self.pair_for_plan: Dict[str, Dict[str, Any]] = {}
        self.seed_route_available = asyncio.Condition()
        self.path_algo = _build_ppo()
        self.policy = self.path_algo.get_policy()
        self.path_env = PathEnv({'render_enabled': False})
        self.obs, _ = self.path_env.reset()
        
        # History tracking
        self.history_: Dict[str, List[Any]] = defaultdict(list)
        self.john_ellipsoids_ = [np.ones(self.d) / (self.d + 1)]

    async def initialize_seed_route(self):
        """Initialize with seed routes from prefs_init"""
        if self.prefs_init:
            initial_plans = await self.designer.design(self.prefs_init)
            for plan in initial_plans:
                plan["tag"] = "seed_route"
                plan["batch"] = 0
                if "ident" not in plan:
                    plan["ident"] = _md5(plan["pref"])
                
                # Add to seed route queue
                self.seed_route_queue.append(plan["ident"])
            
            # Update all instance variables
            self.plans_.extend(initial_plans)
            costs = [p["costs"] for p in initial_plans]
            plan_prefs = [p["pref"] for p in initial_plans]
            idents = [p["ident"] for p in initial_plans]
            
            self.features_.extend(costs)
            self.plan_prefs_.extend(plan_prefs)
            self.idents_.extend(idents)
            
            if not self.batches:
                self.batches.append([])
            
            self.batches[0].extend(list(range(len(initial_plans))))
            
            logger.info(f"Initial seed routes populated: {len(initial_plans)} plans")
            
            # Process first seed route
            if self.seed_route_queue:
                await self._process_next_seed_route()

    async def _process_next_seed_route(self):
        """Process the next seed route in queue"""
        if not self.seed_route_queue:
            logger.info("No more seed routes to process")
            return
        
        # Get next seed route
        self.current_seed_route = self.seed_route_queue.pop(0)
        logger.info(f"Processing seed route: {self.current_seed_route}")
        
        # Reset RL agent for new seed route
        await self._reset_rl_agent()
        
        # Show the seed route to the agent
        seed_plan = next(p for p in self.plans_ if p["ident"] == self.current_seed_route)
        seed_image = self.renderer.render_route(seed_plan)
        self.path_env.deliver(seed_image, 0.0)  # No reward for just showing
        
        # Generate RL optimizations for this seed
        await self._generate_rl_optimizations()

    async def _reset_rl_agent(self):
        """Reset RL agent between seed routes"""
        logger.info("Resetting RL agent for new seed route")
        
        # Show reset image (route-less)
        self.path_env.deliver(self.reset_image, 0.0)
        self.obs, _ = self.path_env.reset()
        
        # Clear any stored actions/logits
        self.last_action = None
        self.last_logits = None
        self.last_logp = None
        self.last_vf = None

    async def _generate_rl_optimizations(self):
        """Generate RL optimizations for current seed route"""
        if not self.current_seed_route:
            logger.warning("No current seed route to optimize")
            return
        
        existing_rl = len(self.seed_to_rl_map[self.current_seed_route])
        to_generate = self.n_optimizations_per_seed - existing_rl
        
        if to_generate <= 0:
            logger.info(f"Seed route {self.current_seed_route} already has {existing_rl} RL optimizations")
            return
        
        logger.info(f"Generating {to_generate} RL optimizations for seed {self.current_seed_route}")
        
        for _ in range(to_generate):
            # Get RL action
            action, _, info = self.path_algo.compute_single_action(
                observation=self.obs,
                explore=True,
                full_fetch=True
            )
            
            self.last_action = action
            self.last_logits = info["action_dist_inputs"]
            self.last_logp = info["action_logp"]
            self.last_vf = info["vf_preds"]
            
            # Create RL-optimized plan
            prefs = np.array([self.last_action]) * 100
            rl_plans = await self.designer.design(prefs)
            
            for plan in rl_plans:
                plan["tag"] = "rl"
                plan["parent"] = self.current_seed_route
                plan["batch"] = len(self.batches)
                if "ident" not in plan:
                    plan["ident"] = _md5(plan["pref"])
                
                # Track parent-child relationship
                self.seed_to_rl_map[self.current_seed_route].append(plan["ident"])
            
            # Update data structures
            self.plans_.extend(rl_plans)
            costs = [p["costs"] for p in rl_plans]
            plan_prefs = [p["pref"] for p in rl_plans]
            idents = [p["ident"] for p in rl_plans]
            
            self.features_.extend(costs)
            self.plan_prefs_.extend(plan_prefs)
            self.idents_.extend(idents)
            
            # Update batches
            if len(self.batches) <= plan["batch"]:
                self.batches.append([])
            
            new_indices = list(range(len(self.plans_) - len(rl_plans), len(self.plans_)))
            self.batches[plan["batch"]].extend(new_indices)
            
            logger.info(f"Generated RL optimization {plan['ident']} for seed {self.current_seed_route}")

    async def _get_task(self, trials=100) -> Tuple[Optional[Path], Optional[float]]:
        """Get a comparison task - prioritize seed vs its RL optimizations"""
        logger.info("Looking for comparison task")
        
        # First priority: compare current seed with its RL optimizations
        if self.current_seed_route and self.seed_to_rl_map[self.current_seed_route]:
            seed_idx = next(i for i, p in enumerate(self.plans_) 
                           if p["ident"] == self.current_seed_route)
            
            for rl_ident in self.seed_to_rl_map[self.current_seed_route]:
                rl_idx = next(i for i, p in enumerate(self.plans_) 
                             if p["ident"] == rl_ident)
                
                if (seed_idx, rl_idx) not in self.asked_ and (rl_idx, seed_idx) not in self.asked_:
                    logger.info(f"Creating comparison: seed {self.current_seed_route} vs RL {rl_ident}")
                    
                    q1, q2 = await asyncio.gather(self.present(seed_idx), self.present(rl_idx))
                    
                    request = {
                        "i": int(seed_idx),
                        "j": int(rl_idx),
                        "routes": [q1, q2],
                        "pair": [q1['ident'], q2['ident']],
                        "id": _md5((min(seed_idx, rl_idx), max(seed_idx, rl_idx))),
                        "comparison_type": "seed_vs_its_rl"
                    }
                    
                    return request, 1.0
        
        # If all comparisons done for current seed, move to next
        if self.seed_route_queue:
            await self._process_next_seed_route()
            return await self._get_task(trials)
        
        logger.info("No more comparisons available")
        return None, None

    async def adjust_fit(self, answers: List[JSON]):
        """Process answers and update RL agent"""
        logger.info(f"Processing {len(answers)} answers")
        
        for ans in answers:
            i, j = ans.get("i", -1), ans.get("j", -1)
            self.asked_.update({(i, j)})
            
            winner_idx = ans.get("winner")
            routes = ans.get("routes", [])
            
            if not routes or winner_idx is None:
                continue
            
            winner_route = routes[winner_idx]
            loser_route = routes[1 - winner_idx]
            
            winner_tag = winner_route.get('plan', {}).get('tag')
            loser_tag = loser_route.get('plan', {}).get('tag')
            
            # Calculate reward
            if winner_tag == "rl" and loser_tag == "seed_route":
                reward = 1.0  # RL beat seed - good!
            elif winner_tag == "seed_route" and loser_tag == "rl":
                reward = -0.5  # Seed beat RL - needs improvement
            else:
                reward = 0.0
            
            # Update RL agent
            if winner_tag == "rl" or loser_tag == "rl":
                img = self.renderer.render_route(winner_route)
                await self._update_rl_agent(img, reward)
            
            # Update history
            self.history_["pair_idx"].append((i, j))
            self.history_["winner"].append(winner_route)
            self.history_["reward"].append(reward)
            
            self.update_count += 1
            
            # Check if we need to move to next seed route
            if self.current_seed_route:
                all_compared = all(
                    (seed_idx, rl_idx) in self.asked_ or (rl_idx, seed_idx) in self.asked_
                    for seed_idx in [i for i, p in enumerate(self.plans_) 
                                     if p["ident"] == self.current_seed_route]
                    for rl_idx in [i for i, p in enumerate(self.plans_) 
                                   if p["ident"] in self.seed_to_rl_map[self.current_seed_route]]
                )
                
                if all_compared and self.seed_route_queue:
                    logger.info(f"All comparisons done for seed {self.current_seed_route}")
                    await self._process_next_seed_route()

    async def _update_rl_agent(self, img, reward):
        """Update the RL agent with reward"""
        try:
            if not hasattr(self, 'last_action') or self.last_action is None:
                return False
            
            self.path_env.deliver(img, reward)
            next_obs, rew, *_ = self.path_env.step(self.last_action)
            
            # Create training batch
            train_batch = SampleBatch({
                SampleBatch.OBS: [self.obs],
                SampleBatch.ACTIONS: [self.last_action],
                SampleBatch.ACTION_DIST_INPUTS: [self.last_logits],
                SampleBatch.ACTION_LOGP: [self.last_logp],
                SampleBatch.VF_PREDS: [self.last_vf],
                SampleBatch.REWARDS: [rew],
                SampleBatch.NEXT_OBS: [next_obs],
                SampleBatch.TERMINATEDS: [False],
                SampleBatch.TRUNCATEDS: [False],
            })
            
            self.replay.append(train_batch)
            
            # Train when enough samples
            if len(self.replay) >= 32:
                big_batch = concat_samples(self.replay)
                big_batch = self.policy.postprocess_trajectory(big_batch)
                self.policy.learn_on_batch(big_batch)
                self.replay.clear()
            
            self.obs = next_obs
            self._step += 1
            
            if self.writer:
                self.writer.add_scalar("reward/plan", rew, self._step)
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating RL agent: {e}")
            return False

    async def get_query(self, **kwargs):
        """Get next query for comparison"""
        if len(self.plans_) <= 1:
            return [], []
        
        task = await self._get_task(trials=30)
        if task is not None:
            return task
        
        return [], []

    async def present(self, i: int) -> JSON:
        """Present a plan for comparison"""
        q = dict(plan=self.plans_[i], costs=self.features_[i], pref=self.plan_prefs_[i])
        q["ident"] = self.plans_[i].get("ident", _md5(self.plan_prefs_[i]))
        _d, _ = await self.designer.present([q], None)
        d = _d[0]
        return {**self.designer.meta, **d}

def _md5(X: Any) -> str:
    if isinstance(X, np.ndarray):
        X = X.tolist()
    to_hash = msgpack.dumps(X)
    return hashlib.md5(to_hash).hexdigest()





________________________________________



# Key modifications needed in api.py

# In the init function, add configuration for n_optimizations_per_seed:
@app.post("/init", status_code=201)
async def init(request: fastapi.Request):
    """
    Initialize a sampler with controlled plan generation
    """
    global FUTURES
    config = await request.json()
    
    # Extract configuration for plan generation control
    for sampler_name, sampler_config in config["samplers"].items():
        # Set default n_optimizations_per_seed if not provided
        if "n_optimizations_per_seed" not in sampler_config:
            sampler_config["n_optimizations_per_seed"] = 1
        
        # Ensure seed route generation is controlled
        if "auto_generate_seeds" not in sampler_config:
            sampler_config["auto_generate_seeds"] = False
    
    # ... rest of init function

# Modify get_requests_loop to properly handle seed route initialization:
async def get_requests_loop(name: str) -> bool:
    aclient = await Client("localhost:6461", asynchronous=True)
    _start = time()

    if not hasattr(get_requests_loop, "_tb"):
        from torch.utils.tensorboard import SummaryWriter
        get_requests_loop._tb = SummaryWriter(log_dir=f"runs/{name}")
    writer = get_requests_loop._tb
    decisions = 0

    state = await _get_state(name)
    
    # Initialize seed routes if available
    initialize_future = aclient.submit(
        lambda s: asyncio.run(s.initialize_seed_route()) if hasattr(s, 'initialize_seed_route') else None,
        state,
        pure=False
    )
    await initialize_future
    
    request_future = aclient.submit(get_requests, state, pure=False)

    for k in itertools.count():
        try:
            requests, scores, stats = await request_future
        except CancelledError as e:
            logger.warning("get_requests cancelled for %s: %s", name, e)
            await asyncio.sleep(0.5)
            state = await _get_state(name)
            request_future = aclient.submit(get_requests, state, pure=False)
            continue
        
        # Validate request structure
        if not requests or not isinstance(requests[0], dict):
            logger.warning("[%s] Invalid request structure, retrying", name)
            await asyncio.sleep(0.5)
            state = await _get_state(name)
            request_future = aclient.submit(get_requests, state, pure=False)
            continue
        
        current_routes = requests[0].get("routes", [])
        
        # Check if this is a valid comparison
        comparison_type = requests[0].get("comparison_type", "unknown")
        if comparison_type == "seed_vs_its_rl":
            logger.info(f"[{name}] Processing seed vs RL comparison")
        
        # Handle auto evaluation if configured
        evalr = AUTO_EVAL.get(name, lambda _: None)
        if hasattr(evalr, "evaluate"):
            winner = evalr.evaluate(current_routes)
        elif callable(evalr):
            winner = evalr(current_routes)
        else:
            winner = None
        
        # Post to frontend or process automatically
        if winner is not None:
            # Auto-process the answer
            answer = [{
                "i": requests[0]["i"],
                "j": requests[0]["j"],
                "sampler": name,
                "ident": current_routes[winner]["ident"],
                "pair": [r["ident"] for r in current_routes],
                "winner": winner,
                "rank": [1 if i == winner else 2 for i in range(2)],
                "routes": current_routes,
            }]
            
            try:
                state = await _get_state(name)
                fut = aclient.submit(
                    update_model,
                    state,
                    answer,
                    pure=False,
                    key=f"{name}-auto-{k}"
                )
                await fut
            except (KeyError, CancelledError) as e:
                logger.warning(f"[{name}] auto-update failed at iter {k}: {e}")
        else:
            # Post to frontend for human evaluation
            await utils.post(name, (requests, scores, stats), delete=False)
            
            # Wait for human response
            response = await conn.blpop(f"answers-{name}", timeout=60)
            
            if response is not None:
                response_feedback = msgpack.loads(response[1])
                winner_frontend = list(response_feedback.values())[0]
                
                answer = [{
                    "i": requests[0]["i"],
                    "j": requests[0]["j"],
                    "sampler": name,
                    "ident": current_routes[winner_frontend]["ident"],
                    "pair": [r["ident"] for r in current_routes],
                    "winner": winner_frontend,
                    "rank": [1 if i == winner_frontend else 2 for i in range(2)],
                    "routes": current_routes,
                }]
                
                try:
                    state = await _get_state(name)
                    fut = aclient.submit(
                        update_model,
                        state,
                        answer,
                        pure=False,
                        key=f"{name}-human-{k}"
                    )
                    await fut
                except (KeyError, CancelledError) as e:
                    logger.warning(f"[{name}] human-update failed at iter {k}: {e}")
        
        # Check if we should stop or continue
        if await utils.should_stop(name):
            await asyncio.sleep(1)
            break
        
        # Check if all seed routes have been processed
        all_processed_future = aclient.submit(
            lambda s: all(len(s.seed_to_rl_map[seed]) >= s.n_optimizations_per_seed 
                         for seed in s.seed_route_queue) if hasattr(s, 'seed_to_rl_map') else True,
            state,
            pure=False
        )
        all_processed = await all_processed_future
        
        if all_processed:
            logger.info(f"[{name}] All seed routes processed with required optimizations")
            # Optionally break or wait for new seed routes
            if not config.get("samplers", {}).get(name, {}).get("auto_generate_seeds", False):
                logger.info(f"[{name}] No auto seed generation - stopping")
                break
        
        try:
            state = await _get_state(name)
        except KeyError as e:
            logger.exception("[%s] failed to reload state: %s", name, e)
            await asyncio.sleep(0.5)
        
        request_future = aclient.submit(get_requests, state, pure=False)

    logger.info("[%s] stopping get_requests_loop; marking stopped flag", name)
    await conn.set(f"stopped-{name}-requests", b"1")
    return True

# Add new endpoint to check plan generation status
@app.get("/stats/{sampler}/plans")
async def get_plan_stats(sampler: str):
    """Get statistics about plan generation"""
    aclient = await Client("localhost:6461", asynchronous=True)
    
    try:
        state = await _get_state(sampler)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"sampler '{sampler}' not running")
    
    async def _get_stats(s):
        if not hasattr(s, 'seed_to_rl_map'):
            return {"error": "Sampler does not support plan tracking"}
        
        stats = {
            "total_plans": len(s.plans_),
            "seed_routes": len([p for p in s.plans_ if p.get("tag") == "seed_route"]),
            "rl_routes": len([p for p in s.plans_ if p.get("tag") == "rl"]),
            "current_seed": s.current_seed_route,
            "seed_queue": len(s.seed_route_queue),
            "seed_to_rl_map": {k: len(v) for k, v in s.seed_to_rl_map.items()},
            "n_optimizations_per_seed": s.n_optimizations_per_seed,
            "comparisons_made": len(s.asked_)
        }
        return stats
    
    return await aclient.submit(_get_stats, state, pure=False)

# Add endpoint to manually add seed routes
@app.post("/{sampler}/add_seed")
async def add_seed_route(sampler: str, prefs: List[float]):
    """Manually add a new seed route to the queue"""
    aclient = await Client("localhost:6461", asynchronous=True)
    
    try:
        state = await _get_state(sampler)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"sampler '{sampler}' not running")
    
    async def _add_seed(s, prefs):
        if not hasattr(s, 'seed_route_queue'):
            return {"error": "Sampler does not support seed route management"}
        
        # Design the plan
        plans = await s.designer.design([prefs])
        for plan in plans:
            plan["tag"] = "seed_route"
            plan["batch"] = len(s.batches)
            if "ident" not in plan:
                plan["ident"] = _md5(plan["pref"])
            
            # Add to queue
            s.seed_route_queue.append(plan["ident"])
            
            # Update data structures
            s.plans_.append(plan)
            s.features_.append(plan["costs"])
            s.plan_prefs_.append(plan["pref"])
            s.idents_.append(plan["ident"])
            
            # Update batches
            if len(s.batches) <= plan["batch"]:
                s.batches.append([])
            s.batches[plan["batch"]].append(len(s.plans_) - 1)
        
        return {"success": True, "added": len(plans), "queue_length": len(s.seed_route_queue)}
    
    future = aclient.submit(_add_seed, state, prefs, pure=False)
    result = await future
    
    # Update state
    async with FUTURES_LOCK[sampler]:
        FUTURES[sampler] = state
    
    return result



___________________________________________






import json
import cv2
import numpy as np
import math

class Render:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.save_num = 0
        
        # Cache zone data
        self._zone_data = None
        self._load_zones()
        
        # Pre-render reset image
        self._reset_image = None
        self._create_reset_image()

    def _load_zones(self):
        """Load zone data once"""
        try:
            with open('zones.json', 'r') as f:
                self._zone_data = json.load(f)['data']
        except FileNotFoundError:
            print("Warning: zones.json not found, using empty zone data")
            self._zone_data = []

    def _create_reset_image(self):
        """Create a reset image showing only zones without any route"""
        # Create blank image
        img = np.full((1000, 1000, 3), 0, dtype=np.uint8)
        
        if not self._zone_data:
            self._reset_image = img
            return
        
        # Use a default view area for reset image
        default_origin = (23.0, 120.0)  # Center of typical area
        default_scale = 111_320.0
        
        # Convert zone coordinates
        zone_lats = [zone['lat'] for zone in self._zone_data]
        zone_lons = [zone['lon'] for zone in self._zone_data]
        zones = list(zip(zone_lats, zone_lons))
        
        # Convert to XY coordinates
        xy_zones = [self.latlon_to_xy(p, origin=default_origin) for p in zones]
        
        # Normalize to image coordinates
        zones_norm = self.normalize(xy_zones)
        
        # Draw zones
        for i, zone in enumerate(zones_norm):
            zone_class = self._zone_data[i]['class']
            if zone_class == 'A':
                color = (255, 0, 0)  # Red
            elif zone_class == 'B':
                color = (0, 255, 0)  # Green
            elif zone_class == 'C':
                color = (0, 0, 255)  # Blue
            else:
                color = (128, 128, 128)  # Gray for unknown
            
            radius = int(float(self._zone_data[i].get('radius', 5)) / 10)
            cv2.circle(img, zone, radius, color, -1)
        
        # Add text to indicate this is reset state
        cv2.putText(img, "RESET STATE - NO ROUTE", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        self._reset_image = img

    def get_reset_image(self):
        """Get the pre-rendered reset image"""
        if self._reset_image is None:
            self._create_reset_image()
        return self._reset_image.copy()

    def bearing_deg(self, p1, p2):
        lat1, lon1 = map(math.radians, p1)
        lat2, lon2 = map(math.radians, p2)
        dlon = lon2 - lon1
        x = math.sin(dlon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        brng = math.degrees(math.atan2(x, y))
        return (brng + 360) % 360

    def latlon_to_xy(self, p, origin, scale=111_320.0):
        lat0, lon0 = origin
        lat, lon = p
        x = (float(lon) - lon0) * math.cos(math.radians(lat0)) * scale
        y = (float(lat) - lat0) * scale
        return x, -y

    def rotate_xy(self, points, angle_deg):
        a = math.radians(-angle_deg)
        rot = np.array([[math.cos(a), -math.sin(a)],
                        [math.sin(a), math.cos(a)]])
        return [tuple(rot @ np.array(p)) for p in points]

    def normalize(self, points, img_size=1000, margin=60):
        if not points:
            return []
        
        xs, ys = zip(*points)
        min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)

        span_x = max_x - min_x
        span_y = max_y - min_y
        span = max(span_x, span_y)
        k = (img_size - 2 * margin) / span if span > 0 else 1.0

        # Center the content
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        
        normalized = []
        for x, y in points:
            xn = int((x - center_x) * k + img_size / 2)
            yn = int((y - center_y) * k + img_size / 2)
            normalized.append((xn, yn))
        return normalized
    
    def render_route(self, route=None, draw_route=True):
        """
        Render a route with zones
        
        Parameters
        ----------
        route : dict or None
            Route data containing plan with lat/lon/alt. If None, returns reset image.
        draw_route : bool
            Whether to draw the route line
        """
        # Return reset image if no route provided
        if route is None or not route:
            return self.get_reset_image()
        
        # Extract route data
        if isinstance(route, dict) and "plan" in route:
            lat = route["plan"].get("lat", [])
            lon = route["plan"].get("lon", [])
        else:
            # Handle case where route might be malformed
            return self.get_reset_image()
        
        # Validate route data
        if not lat or not lon or len(lat) < 2:
            print("Warning: Invalid route data, returning reset image")
            return self.get_reset_image()
        
        # Convert route to coordinate pairs
        route_points = list(zip(lat, lon))
        
        # Get zones
        if self._zone_data:
            zone_lats = [zone['lat'] for zone in self._zone_data]
            zone_lons = [zone['lon'] for zone in self._zone_data]
            zones = list(zip(zone_lats, zone_lons))
        else:
            zones = []
        
        # Calculate bearing for route alignment
        brng = self.bearing_deg(route_points[0], route_points[-1])
        
        # Convert to XY coordinates
        xy_route = [self.latlon_to_xy(p, origin=route_points[0]) for p in route_points]
        xy_zones = [self.latlon_to_xy(p, origin=route_points[0]) for p in zones]
        
        # Rotate based on bearing
        xy_route_rot = self.rotate_xy(xy_route, brng)
        xy_zones_rot = self.rotate_xy(xy_zones, brng)
        
        # Normalize all points together
        xy_all = xy_route_rot + xy_zones_rot
        all_norm = self.normalize(xy_all)
        
        route_norm = all_norm[:len(route_points)]
        zones_norm = all_norm[len(route_points):]
        
        # Create image
        img = np.full((1000, 1000, 3), 0, dtype=np.uint8)
        
        # Draw zones first (background)
        for i, zone in enumerate(zones_norm):
            if i < len(self._zone_data):
                zone_class = self._zone_data[i].get('class', 'unknown')
                if zone_class == 'A':
                    color = (255, 0, 0)  # Red
                elif zone_class == 'B':
                    color = (0, 255, 0)  # Green
                elif zone_class == 'C':
                    color = (0, 0, 255)  # Blue
                else:
                    color = (128, 128, 128)  # Gray
                
                radius = int(float(self._zone_data[i].get('radius', 5)) / 10)
                cv2.circle(img, zone, radius, color, -1)
        
        # Draw route if requested
        if draw_route and len(route_norm) > 1:
            # Draw route line
            for a, b in zip(route_norm[:-1], route_norm[1:]):
                cv2.line(img, a, b, (255, 255, 255), 2, lineType=cv2.LINE_AA)
            
            # Highlight start and end points
            cv2.circle(img, route_norm[0], 8, (0, 255, 0), -1)  # Green start
            cv2.circle(img, route_norm[-1], 8, (255, 0, 0), -1)  # Red end
        
        return img
    
    def render_comparison(self, route1, route2):
        """Render two routes side by side for comparison"""
        img1 = self.render_route(route1)
        img2 = self.render_route(route2)
        
        # Create side-by-side image
        h, w = img1.shape[:2]
        comparison = np.zeros((h, w * 2 + 20, 3), dtype=np.uint8)
        
        # Place images
        comparison[:, :w] = img1
        comparison[:, w + 20:] = img2
        
        # Add labels
        cv2.putText(comparison, "Route 1", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "Route 2", (w + 30, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return comparison
    
    def display_route(self, img):
        """Display route in a window"""
        cv2.imshow('Route', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
