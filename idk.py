# In adjustcal.py, add a helper method for initializing RL components:

class RandomPref(Pairwise):
    # ... existing code ...
    
    def _ensure_rl_components(self):
        """Ensure RL components are initialized (used after deserialization)"""
        if self.path_algo is None:
            logger.info("Initializing RL components...")
            self.path_algo = _build_ppo()
            
        if self.policy is None:
            self.policy = self.path_algo.get_policy()
            
        if self.path_env is None:
            self.path_env = PathEnv({'render_enabled': False})
            
        if self.obs is None:
            self.obs, _ = self.path_env.reset()
            logger.info(f"Initialized obs with shape: {self.obs.shape}")
    
    async def _reset_rl_agent(self):
        """Reset RL agent between seed routes"""
        logger.info("Resetting RL agent for new seed route")
        
        # Ensure RL components exist
        self._ensure_rl_components()
        
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
        
        # Ensure RL components exist
        self._ensure_rl_components()
        
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
            
            # ... rest of the method remains the same ...

    def _rl_components_ready(self):
        """Check if RL components are initialized and ready"""
        required_components = ['path_env', 'path_algo', 'policy', 'obs']
        
        for component in required_components:
            if not hasattr(self, component) or getattr(self, component) is None:
                logger.debug(f"RL component not ready: {component}")
                return False
        
        logger.debug("All RL components are ready")
        return True

    async def _simple_rl_update(self, img, reward):
        """Simple RL update without complex batch handling"""
        try:
            # Ensure RL components exist
            self._ensure_rl_components()
            
            if not hasattr(self, 'last_action'):
                logger.warning("No last_action available, skipping update")
                return False
            
            logger.info(f"Updating RL with reward: {reward}")
            
            self.path_env.deliver(img, reward)
            
            next_obs, rew, *_ = self.path_env.step(self.last_action)
            
            # ... rest of the method remains the same ...

# Also update AdjustCAL.get_prefs:

class AdjustCAL(RandomPref):
    def get_prefs(self, n: int) -> ArrayLike:
        self.request_number += 1
        if len(self.plans_) == 0 and self.prefs_init:
            return np.asarray([self.prefs_init[0]])
        
        # Ensure RL components exist (handles post-deserialization)
        self._ensure_rl_components()
        
        # ... rest of the method remains the same ...
        
        try:
            action, _, info = self.path_algo.compute_single_action(
                observation=self.obs,
                explore=True, 
                full_fetch=True
            )
        except Exception as e:
            logger.error(f"Error in compute_single_action: {e}")
            raise
        
        # ... rest of the method ...
