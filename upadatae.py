# In api.py, update get_requests_loop to store current request:

async def get_requests_loop(name: str) -> bool:
    # ... existing code ...
    
    for k in itertools.count():
        try:
            requests, scores, stats = await request_future
        except CancelledError as e:
            # ... existing error handling ...
            
        current_routes = (
            requests[0].get("routes")
            if requests and isinstance(requests[0], dict)
            else None
        )
        
        # ADD THIS: Store current request for active page
        if current_routes and requests and len(requests) > 0:
            # Generate route images
            try:
                from render import Render
                renderer = Render()
                
                for route in current_routes:
                    if 'plan' in route and 'lat' in route['plan']:
                        # Generate route image
                        img = renderer.render_route(route)
                        # Convert to base64
                        _, buffer = cv2.imencode('.png', img)
                        route['route_image'] = base64.b64encode(buffer).decode('utf-8')
            except Exception as e:
                logger.warning(f"Could not generate route images: {e}")
            
            # Store the request with images
            await conn.set(
                f"sampler-{name}-current-request",
                msgpack.dumps(requests[0]),
                ex=60  # Expire after 60 seconds
            )
        
        # ... rest of existing code ...

# Also add this simpler endpoint that just gets the latest request:
@app.get("/{sampler}/current-evaluation")
async def get_current_evaluation(sampler: str):
    """Get the current evaluation being processed"""
    # First try to get from Redis
    request_data = await conn.get(f"sampler-{sampler}-current-request")
    
    if request_data:
        try:
            request = msgpack.loads(request_data)
            return {
                "request": request,
                "auto_decision": None,  # You can enhance this later
                "timestamp": time()
            }
        except:
            pass
    
    # Fallback: get latest request from queue
    key = f"sampler-{sampler}-requests"
    blob = await conn.lindex(key, 0)  # Get first without removing
    
    if blob:
        try:
            request = msgpack.loads(blob)
            return {
                "request": request,
                "auto_decision": None,
                "timestamp": time()
            }
        except:
            pass
    
    return {"request": None, "auto_decision": None}




// Replace the updateCurrentComparison function in active.html with this:

function updateCurrentComparison(data) {
  if (!data || !data.routes || data.routes.length < 2) {
    document.getElementById('loading').style.display = 'block';
    document.getElementById('comparison-content').style.display = 'none';
    return;
  }

  document.getElementById('loading').style.display = 'none';
  document.getElementById('comparison-content').style.display = 'block';

  // Update route images
  var route1 = data.routes[0];
  var route2 = data.routes[1];

  // Try multiple image sources
  // 1. First try route_image (generated by backend)
  if (route1.route_image) {
    document.getElementById('route-1-img').src = 'data:image/png;base64,' + route1.route_image;
  }
  // 2. Then try file_contents
  else if (route1.file_contents) {
    document.getElementById('route-1-img').src = 'data:image/png;base64,' + route1.file_contents;
  }
  // 3. Generate a placeholder
  else {
    document.getElementById('route-1-img').src = '/static/images/no-image.png';
  }

  // Same for route 2
  if (route2.route_image) {
    document.getElementById('route-2-img').src = 'data:image/png;base64,' + route2.route_image;
  }
  else if (route2.file_contents) {
    document.getElementById('route-2-img').src = 'data:image/png;base64,' + route2.file_contents;
  }
  else {
    document.getElementById('route-2-img').src = '/static/images/no-image.png';
  }

  // Update route info - handle nested plan structure
  var tag1 = route1.plan?.tag || route1.tag || 'unknown';
  var tag2 = route2.plan?.tag || route2.tag || 'unknown';
  
  document.getElementById('route-1-tag').textContent = tag1;
  document.getElementById('route-1-id').textContent = (route1.ident || '').substring(0, 8);
  document.getElementById('route-2-tag').textContent = tag2;
  document.getElementById('route-2-id').textContent = (route2.ident || '').substring(0, 8);

  // Add badge styling for tags
  var tag1Element = document.getElementById('route-1-tag');
  var tag2Element = document.getElementById('route-2-tag');
  
  tag1Element.className = tag1 === 'seed_route' ? 'badge bg-primary' : 'badge bg-info';
  tag2Element.className = tag2 === 'seed_route' ? 'badge bg-primary' : 'badge bg-info';

  // Update comparison info
  document.getElementById('comparison-type').textContent = data.comparison_type || 'seed_vs_rl';
  document.getElementById('request-score').textContent = (data.score || 0).toFixed(3);

  currentRequest = data;
}






# Add these to the top of your render.py file:

import json
import cv2
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import base64

class Render:
    # ... your existing __init__ and other methods ...
    
    def generate_performance_plot(self, routes_data):
        """Generate a performance comparison plot for routes"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract data for plotting - handle both list and dict formats
        if isinstance(routes_data, list):
            seed_routes = [r for r in routes_data if r.get('plan', {}).get('tag') == 'seed_route']
            rl_routes = [r for r in routes_data if r.get('plan', {}).get('tag') == 'rl']
        else:
            # Handle if routes_data is a different format
            seed_routes = []
            rl_routes = []
        
        # Plot bars for average rewards
        seed_rewards = []
        rl_rewards = []
        
        for r in seed_routes[:5]:  # Limit to first 5
            reward = r.get('reward') or r.get('avg_reward') or 0
            seed_rewards.append(float(reward))
            
        for r in rl_routes[:5]:  # Limit to first 5
            reward = r.get('reward') or r.get('avg_reward') or 0
            rl_rewards.append(float(reward))
        
        # Create bar positions
        if seed_rewards or rl_rewards:
            x = range(max(len(seed_rewards), len(rl_rewards)))
            width = 0.35
            
            if seed_rewards:
                ax.bar([i - width/2 for i in range(len(seed_rewards))], 
                       seed_rewards, width, label='Seed Routes', color='blue', alpha=0.7)
            if rl_rewards:
                ax.bar([i + width/2 for i in range(len(rl_rewards))], 
                       rl_rewards, width, label='RL Routes', color='green', alpha=0.7)
        
        ax.set_xlabel('Route Index')
        ax.set_ylabel('Average Reward')
        ax.set_title('Route Performance Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        plot_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        
        return plot_base64
    
    def generate_win_rate_plot(self, routes_data):
        """Generate a win rate plot for routes"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate win rates
        route_labels = []
        win_rates = []
        colors = []
        
        if isinstance(routes_data, list):
            for i, route in enumerate(routes_data[:20]):  # Limit to top 20
                tag = route.get('plan', {}).get('tag', 'unknown')
                wins = route.get('winner') or route.get('n_wins') or 0
                total = route.get('n_shown', 1)  # Avoid division by zero
                
                if total > 0:
                    win_rate = (wins / total) * 100
                    route_labels.append(f"{tag[0].upper()}{i}")
                    win_rates.append(win_rate)
                    colors.append('blue' if tag == 'seed_route' else 'green')
        
        # Create bar chart if we have data
        if win_rates:
            bars = ax.bar(route_labels, win_rates, color=colors, alpha=0.7)
            
            # Add value labels on bars
            for bar, rate in zip(bars, win_rates):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{rate:.1f}%', ha='center', va='bottom')
        
        ax.set_xlabel('Route ID')
        ax.set_ylabel('Win Rate (%)')
        ax.set_title('Route Win Rates')
        ax.set_ylim(0, 110)
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add legend
        blue_patch = patches.Patch(color='blue', label='Seed Routes', alpha=0.7)
        green_patch = patches.Patch(color='green', label='RL Routes', alpha=0.7)
        ax.legend(handles=[blue_patch, green_patch])
        
        # Save to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        plot_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        
        return plot_base64
