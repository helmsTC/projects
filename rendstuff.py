# Add these methods to your Render class in render.py:

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg
import io
import base64

class Render:
    # ... existing __init__ and other methods ...
    
    def generate_performance_plot(self, routes_data):
        """Generate a performance comparison plot for routes"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract data for plotting
        seed_routes = [r for r in routes_data if r.get('tag') == 'seed_route']
        rl_routes = [r for r in routes_data if r.get('tag') == 'rl']
        
        # Plot bars for average rewards
        seed_rewards = [r.get('avg_reward', 0) for r in seed_routes]
        rl_rewards = [r.get('avg_reward', 0) for r in rl_routes]
        
        x = range(len(seed_routes))
        width = 0.35
        
        if seed_rewards:
            ax.bar([i - width/2 for i in x], seed_rewards, width, label='Seed Routes', color='blue', alpha=0.7)
        if rl_rewards:
            ax.bar([i + width/2 for i in x[:len(rl_rewards)]], rl_rewards, width, label='RL Routes', color='green', alpha=0.7)
        
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
        
        for i, route in enumerate(routes_data[:20]):  # Limit to top 20
            tag = route.get('tag', 'unknown')
            wins = route.get('n_wins', 0)
            total = route.get('n_shown', 0)
            
            if total > 0:
                win_rate = wins / total * 100
                route_labels.append(f"{tag[0].upper()}{i}")
                win_rates.append(win_rate)
                colors.append('blue' if tag == 'seed_route' else 'green')
        
        # Create bar chart
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
    
    def save_plots_to_files(self, routes_data, output_dir='static/plots'):
        """Save plots as files and return their paths"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate performance plot
        perf_plot = self.generate_performance_plot(routes_data)
        perf_path = os.path.join(output_dir, 'performance_plot.png')
        with open(perf_path, 'wb') as f:
            f.write(base64.b64decode(perf_plot))
        
        # Generate win rate plot
        win_plot = self.generate_win_rate_plot(routes_data)
        win_path = os.path.join(output_dir, 'win_rate_plot.png')
        with open(win_path, 'wb') as f:
            f.write(base64.b64decode(win_plot))
        
        return {
            'performance_plot': '/' + perf_path,
            'win_rate_plot': '/' + win_path
        }






        # Add these endpoints to api.py:

@app.get("/{sampler}/current-evaluation")
async def get_current_evaluation(sampler: str):
    """Get the current evaluation being processed"""
    # Check if there's a current request in Redis
    request_key = f"sampler-{sampler}-current-request"
    request_data = await conn.get(request_key)
    
    if request_data:
        request = msgpack.loads(request_data)
        
        # Check if auto-evaluator made a decision
        auto_decision = None
        if sampler in AUTO_EVAL:
            eval_func = AUTO_EVAL[sampler]
            if hasattr(eval_func, 'last_decision'):
                auto_decision = eval_func.last_decision
        
        return {
            "request": request,
            "auto_decision": auto_decision,
            "timestamp": time()
        }
    
    return {"request": None, "auto_decision": None}

# Modify the existing get_item endpoint to ensure it properly handles the data:
@app.get("/{sampler}/item/{ident}")
async def get_item(sampler: str, ident: str, remove: str = None, key: str = None) -> JSON:
    aclient = await Client("localhost:6461", asynchronous=True)
    try:
        state = await _get_state(sampler)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"sampler '{sampler}' not a running sampler")

    # Helper function to get item data
    async def _get_item_data(s, ident):
        # Find the item in the sampler's data
        for i, plan_ident in enumerate(s.idents_):
            if plan_ident == ident:
                # Build complete item data
                item = {
                    "ident": ident,
                    "plan": s.plans_[i],
                    "costs": s.features_[i],
                    "pref": s.plan_prefs_[i],
                    "score": 0.0  # Default score
                }
                
                # Add comparison stats if available
                if hasattr(s, '_comparison_stats'):
                    stats = s._comparison_stats(i)
                    item.update(stats)
                
                # Add file data if available
                if "file_name" in s.plans_[i]:
                    item["file_name"] = s.plans_[i]["file_name"]
                if "file_contents" in s.plans_[i]:
                    item["file_contents"] = s.plans_[i]["file_contents"]
                
                return item
        
        return None
    
    data = await aclient.submit(_get_item_data, state, ident, pure=False)
    
    if data is None:
        raise HTTPException(status_code=404, detail=f"Item with ident '{ident}' not found")
    
    # Handle remove parameter
    if remove and "," in remove:
        for field in remove.split(","):
            data.pop(field.strip(), None)
    elif remove:
        data.pop(remove, None)
    
    # Handle key parameter - return just one field
    if key:
        if key == "file" and "file_contents" in data:
            # Return file as download
            import base64
            file_data = base64.b64decode(data["file_contents"])
            file_name = data.get("file_name", f"{ident}.png")
            
            from fastapi.responses import Response
            return Response(
                content=file_data,
                media_type="application/octet-stream",
                headers={
                    "Content-Disposition": f"attachment; filename={file_name}"
                }
            )
        elif key in data:
            return {key: data[key]}
        else:
            raise HTTPException(status_code=404, detail=f"Key '{key}' not found in item data")
    
    return data

# Update the get_requests_loop to store current request for active page:
# Find this section in get_requests_loop and add the store operation:
"""
# After this line:
current_routes = (
    requests[0].get("routes")
    if requests and isinstance(requests[0], dict)
    else None
)

# ADD THIS:
# Store current request for active page
if current_routes and requests:
    await conn.set(
        f"sampler-{name}-current-request",
        msgpack.dumps(requests[0]),
        ex=60  # Expire after 60 seconds
    )
"""
