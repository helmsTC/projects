# Add these imports at the top of api.py
from fastapi.responses import HTMLResponse
from pathlib import Path
import base64
import io

# Add these endpoints to api.py

@app.get("/api/{sampler}/current-evaluation")
async def get_current_evaluation(sampler: str):
    """Get the current evaluation being processed"""
    # Check if there's a current request in Redis
    current_req_key = f"current-evaluation-{sampler}"
    current_data = await conn.get(current_req_key)
    
    if current_data:
        return msgpack.loads(current_data)
    
    # Fallback: get the latest request from the queue
    key = f"sampler-{sampler}-requests"
    latest = await conn.lindex(key, 0)
    
    if latest:
        data = msgpack.loads(latest)
        # Check if auto-evaluator made a decision
        auto_decision = AUTO_EVAL.get(sampler)
        if auto_decision and hasattr(auto_decision, 'last_decision'):
            data['auto_decision'] = auto_decision.last_decision
        return {"request": data, "auto_decision": None}
    
    return {"request": None, "auto_decision": None}

@app.get("/api/{sampler}/evaluation-stats")
async def get_evaluation_stats(sampler: str):
    """Get statistics about evaluations"""
    # Get counts from Redis
    total_key = f"evaluation-total-{sampler}"
    auto_key = f"evaluation-auto-{sampler}"
    human_key = f"evaluation-human-{sampler}"
    
    total = int(await conn.get(total_key) or 0)
    auto = int(await conn.get(auto_key) or 0)
    human = int(await conn.get(human_key) or 0)
    
    return {
        "total": total,
        "auto": auto,
        "human": human,
        "auto_percentage": (auto / total * 100) if total > 0 else 0
    }

# Update the get_requests_loop function to store current evaluation
# Add this after getting requests (around line 175 in get_requests_loop):
        
        # Store current evaluation for frontend
        current_eval = {
            "request": requests[0] if requests else None,
            "scores": scores,
            "timestamp": time()
        }
        await conn.set(f"current-evaluation-{name}", msgpack.dumps(current_eval), ex=60)
        
        # Update evaluation statistics
        await conn.incr(f"evaluation-total-{name}")
        if winner is not None:
            await conn.incr(f"evaluation-auto-{name}")
        else:
            await conn.incr(f"evaluation-human-{name}")





# Add these to _private.py after the imports
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

# Set up Jinja2 templates (add after app initialization)
templates = Jinja2Templates(directory=str(box_DIR / "templates"))

# Add these routes to _private.py

@app.get("/active/{sampler}", response_class=HTMLResponse, tags=["web interface"])
async def active_page(request: Request, sampler: str, username: str = Depends(_get_username)):
    """Serve the active evaluation page"""
    await _ensure_sampler_present(sampler)
    
    # Read the active.html template
    template_path = box_DIR / "templates" / "active.html"
    if not template_path.exists():
        # If template doesn't exist, use the content from the uploaded file
        template_content = """[Insert active.html content here]"""
    else:
        template_content = template_path.read_text()
    
    # Replace template variables
    template_content = template_content.replace("{{sampler}}", sampler)
    template_content = template_content.replace("{{ sampler }}", sampler)
    
    return HTMLResponse(content=template_content)

@app.get("/rankings/{sampler}", response_class=HTMLResponse, tags=["web interface"]) 
async def rankings_page(request: Request, sampler: str, username: str = Depends(_get_username)):
    """Serve the rankings page with plots"""
    await _ensure_sampler_present(sampler)
    
    # Get rankings data
    async with httpx.AsyncClient() as httpclient:
        r = await httpclient.get(f"http://backendpython:6463/rankings/{sampler}", timeout=TIMEOUT)
        rankings_data = r.json()
    
    # Get cost names from first item
    cost_names = []
    if rankings_data and 'costs' in rankings_data[0]:
        cost_names = [f"Cost {i+1}" for i in range(len(rankings_data[0]['costs']))]
    
    # Generate plots (simplified for now)
    plot1_path = None
    plot2_path = None
    
    # Read template
    template_path = box_DIR / "templates" / "rankings_routes.html"
    if template_path.exists():
        template_content = template_path.read_text()
    else:
        template_content = """[Insert rankings_routes.html content here]"""
    
    # Simple template rendering (replace with Jinja2 if available)
    template_content = template_content.replace("{{sampler}}", sampler)
    template_content = template_content.replace("{{ sampler }}", sampler)
    
    # Render rankings data
    rankings_html = ""
    for idx, item in enumerate(rankings_data):
        # Build the HTML for each ranking item
        # This is simplified - you'd want to use proper templating
        pass
    
    return HTMLResponse(content=template_content)







// Replace the fetchCurrentEvaluation function in active.html with this:

function fetchCurrentEvaluation() {
  // Get current evaluation from API
  $.ajax({
    url: '/api/' + sampler + '/current-evaluation',
    method: 'GET',
    success: function(data) {
      if (data && data.request) {
        updateCurrentComparison(data.request);
        
        if (data.auto_decision !== null && data.auto_decision !== undefined) {
          document.getElementById('auto-eval-result').style.display = 'block';
          document.getElementById('auto-choice').textContent = 'Route ' + (data.auto_decision + 1);
        } else {
          document.getElementById('auto-eval-result').style.display = 'none';
        }
      } else {
        // No current evaluation, try to get from requests queue
        $.get('/api/' + sampler + '/reqs?n=1', function(reqData) {
          if (reqData && reqData.data && reqData.data[0]) {
            updateCurrentComparison(reqData.data[0]);
            document.getElementById('auto-eval-result').style.display = 'none';
          }
        });
      }
    },
    error: function() {
      // Fallback to checking requests endpoint
      $.get('/api/' + sampler + '/reqs?n=1', function(data) {
        if (data && data.data && data.data[0]) {
          updateCurrentComparison(data.data[0]);
        }
      });
    }
  });

  // Update stats
  $.get('/api/' + sampler + '/evaluation-stats', function(data) {
    document.getElementById('total-comparisons').textContent = data.total || 0;
    document.getElementById('auto-evaluated').textContent = data.auto || 0;
    document.getElementById('human-evaluated').textContent = data.human || 0;
    
    // Update auto-eval percentage if needed
    var autoPercent = document.getElementById('auto-percentage');
    if (autoPercent) {
      autoPercent.textContent = (data.auto_percentage || 0).toFixed(1) + '%';
    }
  });
}

// Fix the updateCurrentComparison function to handle the data properly
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

    // Handle route images - check multiple possible fields
    function setRouteImage(imgElement, route) {
        if (route.route_image) {
            imgElement.src = 'data:image/png;base64,' + route.route_image;
        } else if (route.file_contents) {
            imgElement.src = 'data:image/png;base64,' + route.file_contents;
        } else if (route.plan && route.plan.file_contents) {
            imgElement.src = 'data:image/png;base64,' + route.plan.file_contents;
        } else {
            // If no image available, you could generate a placeholder or leave blank
            imgElement.src = '/static/images/no-route.png';
        }
    }

    setRouteImage(document.getElementById('route-1-img'), route1);
    setRouteImage(document.getElementById('route-2-img'), route2);

    // Update route info
    var tag1 = (route1.plan && route1.plan.tag) || route1.tag || 'unknown';
    var tag2 = (route2.plan && route2.plan.tag) || route2.tag || 'unknown';
    
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





<!-- Add this section after the "Estimated Preferences" details block in rankings_routes.html -->

<details>
  <summary>Evaluation Metrics</summary>
  <div id="evaluation-metrics" style="padding: 20px; background: #f8f9fa; border-radius: 5px; margin: 10px 0;">
    <h4>Performance Metrics</h4>
    <div class="row">
      <div class="col-md-3">
        <div class="metric-card" style="background: white; padding: 15px; border-radius: 5px; text-align: center;">
          <h5>Total Evaluations</h5>
          <p class="metric-value" style="font-size: 2em; font-weight: bold; color: #007bff;">
            <span id="metric-total-evals">0</span>
          </p>
        </div>
      </div>
      <div class="col-md-3">
        <div class="metric-card" style="background: white; padding: 15px; border-radius: 5px; text-align: center;">
          <h5>Auto-Evaluated</h5>
          <p class="metric-value" style="font-size: 2em; font-weight: bold; color: #28a745;">
            <span id="metric-auto-evals">0</span>
            <small style="font-size: 0.5em;">(<span id="metric-auto-percent">0</span>%)</small>
          </p>
        </div>
      </div>
      <div class="col-md-3">
        <div class="metric-card" style="background: white; padding: 15px; border-radius: 5px; text-align: center;">
          <h5>Human-Evaluated</h5>
          <p class="metric-value" style="font-size: 2em; font-weight: bold; color: #ffc107;">
            <span id="metric-human-evals">0</span>
          </p>
        </div>
      </div>
      <div class="col-md-3">
        <div class="metric-card" style="background: white; padding: 15px; border-radius: 5px; text-align: center;">
          <h5>Win Rate (Top 5)</h5>
          <p class="metric-value" style="font-size: 2em; font-weight: bold; color: #17a2b8;">
            <span id="metric-win-rate">0</span>%
          </p>
        </div>
      </div>
    </div>
    
    <div class="row mt-3">
      <div class="col-md-6">
        <h5>Route Distribution</h5>
        <div class="progress" style="height: 30px;">
          <div id="seed-progress" class="progress-bar bg-primary" role="progressbar" style="width: 50%">
            Seed Routes: <span id="seed-count">0</span>
          </div>
          <div id="rl-progress" class="progress-bar bg-success" role="progressbar" style="width: 50%">
            RL Routes: <span id="rl-count">0</span>
          </div>
        </div>
      </div>
      <div class="col-md-6">
        <h5>Average Reward Trend</h5>
        <canvas id="reward-trend-chart" width="400" height="200"></canvas>
      </div>
    </div>
  </div>
</details>

<script>
// Add this JavaScript to fetch and display metrics
function fetchEvaluationMetrics() {
  // Get evaluation stats
  $.get('/api/' + sampler + '/evaluation-stats', function(data) {
    $('#metric-total-evals').text(data.total || 0);
    $('#metric-auto-evals').text(data.auto || 0);
    $('#metric-human-evals').text(data.human || 0);
    $('#metric-auto-percent').text((data.auto_percentage || 0).toFixed(1));
  });
  
  // Calculate metrics from rankings data
  var seedCount = 0;
  var rlCount = 0;
  var totalWins = 0;
  var totalShown = 0;
  
  $('#table tbody tr').each(function(index) {
    var tag = $(this).find('td:eq(3) span').text();
    if (tag === 'Seed') seedCount++;
    else if (tag === 'RL') rlCount++;
    
    if (index < 5) { // Top 5 win rate
      var wins = parseInt($(this).find('td:eq(4) span').text()) || 0;
      totalWins += wins;
      totalShown += 1; // Simplified - you'd need actual shown count
    }
  });
  
  // Update route distribution
  var total = seedCount + rlCount;
  if (total > 0) {
    $('#seed-progress').css('width', (seedCount / total * 100) + '%');
    $('#rl-progress').css('width', (rlCount / total * 100) + '%');
  }
  $('#seed-count').text(seedCount);
  $('#rl-count').text(rlCount);
  
  // Calculate win rate
  var winRate = totalShown > 0 ? (totalWins / totalShown * 100) : 0;
  $('#metric-win-rate').text(winRate.toFixed(1));
  
  // Draw reward trend chart (simplified)
  drawRewardTrend();
}

function drawRewardTrend() {
  var canvas = document.getElementById('reward-trend-chart');
  if (!canvas) return;
  
  var ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  // Simple line chart - in production you'd use Chart.js or similar
  ctx.strokeStyle = '#007bff';
  ctx.lineWidth = 2;
  ctx.beginPath();
  
  // Get rewards from table
  var rewards = [];
  $('#table tbody tr').each(function(index) {
    if (index < 20) { // Limit to first 20
      var reward = parseFloat($(this).find('td:eq(5)').text()) || 0;
      rewards.push(reward);
    }
  });
  
  if (rewards.length > 0) {
    var maxReward = Math.max(...rewards);
    var minReward = Math.min(...rewards);
    var range = maxReward - minReward || 1;
    
    rewards.forEach(function(reward, index) {
      var x = (index / (rewards.length - 1)) * (canvas.width - 20) + 10;
      var y = canvas.height - ((reward - minReward) / range * (canvas.height - 20) + 10);
      
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    
    ctx.stroke();
  }
}

// Call metrics fetch when page loads
$(document).ready(function() {
  fetchEvaluationMetrics();
  
  // Refresh metrics every 5 seconds
  setInterval(fetchEvaluationMetrics, 5000);
});
</script>
