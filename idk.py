# Exact Line-by-Line Changes

## File: `adjustcal.py`

### Change 1: Update RandomPref.__init__ signature
```python
# OLD:
def __init__(self, name, *, designer: "Planner", n_paths: int = 2, eps: float = 1, prefs_init = None):

# NEW:
def __init__(self, name, *, designer: "Planner", n_paths: int = 2, eps: float = 1, 
             prefs_init = None, n_optimizations_per_seed: int = 1):
```

### Change 2: Add tracking variables at end of RandomPref.__init__
```python
# After this existing line:
self.john_ellipsoids_ = [np.ones(self.d) / (self.d + 1)]

# ADD THESE LINES:
# ADD THESE LINES for tracking seed routes and their RL children:
self.seed_to_rl_map = {}  # Track which RL routes belong to which seed
self.current_seed_being_optimized = None  # Track which seed we're currently optimizing
self.max_optimizations_per_seed = n_optimizations_per_seed  # Store the limit
```

### Change 3: In _get_paths, update the seed optimization check
```python
# OLD (around line 320-330):
has_rl_child = any(
    p.get("tag") == "rl" and p.get("parent") == seed_ident 
    for p in self.plans_
)

if not has_rl_child:
    seed_routes_needing_optimization.append((i, plan))

# NEW:
# Count existing RL children for this seed
rl_children_count = sum(1 for p in self.plans_ 
                      if p.get("tag") == "rl" and p.get("parent") == seed_ident)

if rl_children_count < self.max_optimizations_per_seed:
    seed_routes_needing_optimization.append((i, plan))
    logger.info(f"Seed route {seed_ident} has {rl_children_count} RL children, needs more")
else:
    logger.info(f"Seed route {seed_ident} already has {rl_children_count} RL optimizations")
```

### Change 4: In _get_paths, add seed tracking when creating RL
```python
# After this line (when seed_routes_needing_optimization has items):
batch_id = len(self.batches)

# ADD THIS LINE:
# Track which seed we're optimizing
self.current_seed_being_optimized = parent_plan["ident"]
```

### Change 5: In _get_paths, add parent tracking when creating plans
```python
# OLD (around line 360):
if parent_plan:
    p["parent"] = parent_plan["ident"]

# NEW:
if parent_plan:
    p["parent"] = parent_plan["ident"]
    # Track in our mapping
    if parent_plan["ident"] not in self.seed_to_rl_map:
        self.seed_to_rl_map[parent_plan["ident"]] = []
    self.seed_to_rl_map[parent_plan["ident"]].append(p["ident"])
```

### Change 6: In _get_task, update comparison logic
```python
# OLD (the part that adds to valid_pairs):
if (tag_i == "seed_route" and tag_j == "rl") or (tag_i == "rl" and tag_j == "seed_route"):
    if tag_i == "seed_route" and plan_j.get("parent") == plan_i.get("ident"):
        valid_pairs.append((i, j, "seed_vs_its_rl"))
    elif tag_j == "seed_route" and plan_i.get("parent") == plan_j.get("ident"):
        valid_pairs.append((i, j, "rl_vs_its_seed"))
    # For now, let's also allow any seed vs any RL comparison for more data
    elif (tag_i == "seed_route" and tag_j == "rl") or (tag_i == "rl" and tag_j == "seed_route"):
        valid_pairs.append((i, j, "seed_vs_rl_general"))

# NEW (remove the general comparison part):
if (tag_i == "seed_route" and tag_j == "rl") or (tag_i == "rl" and tag_j == "seed_route"):
    # Check if they have parent-child relationship
    if tag_i == "seed_route" and plan_j.get("parent") == plan_i.get("ident"):
        valid_pairs.append((i, j, "seed_vs_its_rl"))
    elif tag_j == "seed_route" and plan_i.get("parent") == plan_j.get("ident"):
        valid_pairs.append((i, j, "rl_vs_its_seed"))
```

### Change 7: In AdjustCAL.get_prefs, add reset logic at beginning
```python
# After these lines:
self.request_number += 1
if len(self.plans_) == 0 and self.prefs_init:
    return np.asarray([self.prefs_init[0]])

# ADD THESE LINES:
# Check if we're switching to a different seed route
if hasattr(self, 'last_optimized_seed') and self.current_seed_being_optimized != self.last_optimized_seed:
    logger.info(f"Switching from seed {self.last_optimized_seed} to {self.current_seed_being_optimized}")
    # Reset the RL environment when switching seeds
    if self.path_env is not None:
        self.path_env.deliver(self.reset_image, 0.0)
        self.obs, _ = self.path_env.reset()
        logger.info("Reset RL agent for new seed route")

self.last_optimized_seed = self.current_seed_being_optimized
```

## File: `pairwise.py`

### Change 1: Update AdjustCAL.__init__ signature
```python
# OLD:
def __init__(self, name: str = "", *, designer: "Planner" = None, n_paths: int = 2, 
             eps=1, prefs_init: Optional[List[float]]=None):

# NEW:
def __init__(self, name: str = "", *, designer: "Planner" = None, n_paths: int = 2, 
             eps=1, prefs_init: Optional[List[float]]=None, 
             n_optimizations_per_seed: int = 1):
```

### Change 2: Update super().__init__ call
```python
# OLD:
super().__init__(name=name, designer=designer, n_paths=n_paths, eps=eps, prefs_init=prefs_init)

# NEW:
super().__init__(name=name, designer=designer, n_paths=n_paths, eps=eps, 
                 prefs_init=prefs_init, 
                 n_optimizations_per_seed=n_optimizations_per_seed)
```

That's it! No other changes needed.
