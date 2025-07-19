async def _get_paths(self, n_search: int, max_optimizations_per_seed=1):
    """Simplified _get_paths that waits when all seeds are optimized"""
    logger.info(f"Starting _get_paths with n_search={n_search}")

    first_batch = len(self.plans_) == 0
    
    if first_batch:
        parent_plan = None
        tag = "seed_route"
        batch_id = 0
        n_to_make = 1
        logger.info("Generating initial seed route")
        
        if not self.batches:
            self.batches.append([])
    else:
        seed_routes_needing_optimization = []
        
        for i, plan in enumerate(self.plans_):
            if plan.get("tag") == "seed_route":
                seed_ident = plan["ident"]
                
                # Count existing RL children for this seed
                rl_children_count = sum(1 for p in self.plans_ 
                                      if p.get("tag") == "rl" and p.get("parent") == seed_ident)
                
                if rl_children_count < self.max_optimizations_per_seed:
                    seed_routes_needing_optimization.append((i, plan))
                    logger.info(f"Seed route {seed_ident} has {rl_children_count} RL children, needs more")
                else:
                    logger.info(f"Seed route {seed_ident} already has {rl_children_count} RL optimizations")
        
        if seed_routes_needing_optimization:
            seed_idx, parent_plan = seed_routes_needing_optimization[0]
            tag = "rl"
            batch_id = len(self.batches)  
            n_to_make = 1  
            
            # Track which seed we're optimizing
            self.current_seed_being_optimized = parent_plan["ident"]
            
            self.batches.append([])
            
            logger.info(f"Generating 1 RL-optimized route from seed route {parent_plan['ident']} in batch {batch_id}")
        else:
            # ALL SEEDS ARE FULLY OPTIMIZED - DO NOT GENERATE NEW SEED ROUTES
            logger.info(f"All seed routes have been fully optimized. Waiting for new seed routes to be added.")
            logger.info(f"Current state: {len([p for p in self.plans_ if p.get('tag') == 'seed_route'])} seed routes, "
                       f"{len([p for p in self.plans_ if p.get('tag') == 'rl'])} RL routes")
            
            # Return empty results - no new plans to generate
            return [], [], [], []

    prefs = self.get_prefs(n_to_make)
    plans = await self.designer.design(prefs)

    for p in plans:
        p["batch"] = batch_id
        p["tag"] = tag
        if "ident" not in p:
            p["ident"] = _md5(p["pref"])
        if parent_plan:
            p["parent"] = parent_plan["ident"]
            # Track in our mapping
            if parent_plan["ident"] not in self.seed_to_rl_map:
                self.seed_to_rl_map[parent_plan["ident"]] = []
            self.seed_to_rl_map[parent_plan["ident"]].append(p["ident"])

    new_indices = list(range(len(self.plans_), len(self.plans_) + len(plans)))
    
    while len(self.batches) <= batch_id:
        self.batches.append([])
    
    self.batches[batch_id].extend(new_indices)
    
    self.plans_.extend(plans)
    
    costs = [p["costs"] for p in plans]
    plan_prefs = [p["pref"] for p in plans]
    idents = [p["ident"] for p in plans]
    
    self.features_.extend(costs)
    self.plan_prefs_.extend(plan_prefs)
    self.idents_.extend(idents)
    
    assert len(self.plans_) == len(self.features_) == len(self.plan_prefs_) == len(self.idents_), \
        f"List length mismatch: plans={len(self.plans_)}, features={len(self.features_)}, " \
        f"prefs={len(self.plan_prefs_)}, idents={len(self.idents_)}"

    logger.info(f"Generated {len(plans)} plans with tag '{tag}' in batch {batch_id}")
    logger.info(f"Total plans: {len(self.plans_)}, Total batches: {len(self.batches)}")
    
    seed_count = len([p for p in self.plans_ if p.get("tag") == "seed_route"])
    rl_count = len([p for p in self.plans_ if p.get("tag") == "rl"])
    logger.info(f"Current state: {seed_count} seed routes, {rl_count} RL routes")

    return plans, costs, plan_prefs, idents
    
    
    
    async def extend_(self, n_paths: Optional[int]=None):
    n_paths = n_paths or self.n_paths
    more_data = await self._get_paths(n_paths, max_optimizations_per_seed=self.max_optimizations_per_seed)
    
    # Check if _get_paths returned empty data (all seeds optimized)
    if not more_data or not more_data[0]:  # more_data[0] is the plans list
        logger.info("No new plans generated - all seeds are fully optimized")
        return self  # Return self unchanged
    
    return await self.extend(more_data)
    
    
    # Add this method to RandomPref class:
async def add_seed_route(self, prefs: List[float]):
    """Manually add a new seed route with given preferences"""
    logger.info(f"Manually adding seed route with prefs: {prefs}")
    
    # Design the plan
    plans = await self.designer.design([prefs])
    
    batch_id = len(self.batches)
    if not self.batches:
        self.batches.append([])
    else:
        self.batches.append([])
    
    for plan in plans:
        plan["tag"] = "seed_route"
        plan["batch"] = batch_id
        if "ident" not in plan:
            plan["ident"] = _md5(plan["pref"])
    
    # Update all tracking structures
    new_indices = list(range(len(self.plans_), len(self.plans_) + len(plans)))
    self.batches[batch_id].extend(new_indices)
    
    self.plans_.extend(plans)
    costs = [p["costs"] for p in plans]
    plan_prefs = [p["pref"] for p in plans]
    idents = [p["ident"] for p in plans]
    
    self.features_.extend(costs)
    self.plan_prefs_.extend(plan_prefs)
    self.idents_.extend(idents)
    
    logger.info(f"Added {len(plans)} new seed routes")
    logger.info(f"Total routes: {len(self.plans_)} (seeds: {len([p for p in self.plans_ if p.get('tag') == 'seed_route'])})")
    
    # Trigger generation of RL routes for the new seeds
    await self._get_paths(1, max_optimizations_per_seed=self.max_optimizations_per_seed)
    
    return plans
    
    