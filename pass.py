# In api.py, add this import at the top with other imports
from render import Render
import base64
import cv2

# Then update the get_requests_loop function
# Find the section after getting current_routes (around line 175-180)
# Replace/update with this code:

async def get_requests_loop(name: str) -> bool:
    aclient = await Client("localhost:6461", asynchronous=True)
    _start = time()

    if not hasattr(get_requests_loop, "_tb"):
        from torch.utils.tensorboard import SummaryWriter
        get_requests_loop._tb = SummaryWriter(log_dir=f"runs/{name}")
    writer = get_requests_loop._tb
    decisions = 0

    # Initialize renderer once
    renderer = Render()

    state = await _get_state(name)
    request_future = aclient.submit(get_requests, state, pure=False)

    seed_plan_processed = False
    for k in itertools.count():
        try:
            requests, scores, stats = await request_future
        except CancelledError as e:
            logger.warning("get_requests cancelled for %s: %s", name, e)
            await asyncio.sleep(0.5)
            state          = await _get_state(name)
            request_future = aclient.submit(get_requests, state, pure=False)
            continue
            
        current_routes = (
            requests[0].get("routes")
            if requests and isinstance(requests[0], dict)
            else None
        )
        valid_pair = (
            current_routes
            and len(current_routes) == 2
            and all(isinstance(r, dict) and "ident" in r for r in current_routes)
        )
        if not valid_pair:
            logger.warning("[%s] malformed routes payload, retrying: %r", name, current_routes)
            await asyncio.sleep(0.5)
            state          = await _get_state(name)
            request_future = aclient.submit(get_requests, state, pure=False)
            continue

        # Generate route images using Render
        try:
            # Render images for both routes
            route1_img = renderer.render_route(current_routes[0])
            route2_img = renderer.render_route(current_routes[1])
            
            # Convert images to base64
            _, buffer1 = cv2.imencode('.png', route1_img)
            route1_base64 = base64.b64encode(buffer1).decode('utf-8')
            
            _, buffer2 = cv2.imencode('.png', route2_img)
            route2_base64 = base64.b64encode(buffer2).decode('utf-8')
            
            # Add the base64 images to the routes data
            current_routes[0]['route_image'] = route1_base64
            current_routes[1]['route_image'] = route2_base64
            
            logger.info("[%s] Generated route images for comparison", name)
        except Exception as e:
            logger.warning("[%s] Failed to generate route images: %s", name, e)
            # Continue without images if rendering fails
            pass

        # Store current evaluation for frontend display with images
        current_eval_data = {
            "request": requests[0] if requests else None,
            "routes": current_routes,  # This now includes the route_image field
            "timestamp": time(),
            "auto_decision": None,
            "comparison_type": "seed_vs_rl" if any(r.get('plan', {}).get('tag') == 'seed_route' for r in current_routes) else "rl_vs_rl",
            "score": scores[0] if scores else 0
        }
        
        # Store in Redis for the Active page to retrieve
        await conn.set(
            f"current-evaluation-{name}", 
            msgpack.dumps(current_eval_data),
            ex=60  # Expire after 60 seconds
        )

        evalr = AUTO_EVAL.get(name, lambda _:None)
        if hasattr(evalr, "evaluate"):
            winner = evalr.evaluate(current_routes)
        elif callable(evalr):
            winner = evalr(current_routes)
        else:
            winner = None
            
        await utils.post(name, (requests, scores, stats), delete=False)

        logger.info("[%s] Evaluator chose: %r", name, winner)
        decisions += 1
        setattr(get_requests_loop, "_decisions", decisions)
        
        if writer := getattr(get_requests_loop, "_tb", None):
            writer.add_scalar("evaluator/auto_vs_human", 1.0 if winner is not None else 0.0, decisions)
            
        if winner is not None:
            # Update the stored evaluation with auto decision
            current_eval_data["auto_decision"] = winner
            await conn.set(
                f"current-evaluation-{name}", 
                msgpack.dumps(current_eval_data),
                ex=60
            )
            
            # Increment auto-evaluation counter
            await conn.incr(f"evaluation-auto-{name}")
            
            answer = [{
                "i": requests[0]["i"],
                "j": requests[0]["i"],
                "sampler": name,
                "ident":  current_routes[winner]["ident"],
                "pair":   [r["ident"] for r in current_routes],
                "winner": winner,
                "rank":   [1 if i == winner else 2 for i in range(2)],
                "routes": current_routes,
            }]
            
            try:
                state = await _get_state(name)
                fut   = aclient.submit(
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
            # Human evaluation needed
            await conn.incr(f"evaluation-human-{name}")
            
            logger.info(f"[{name}] LOW confidence → posting to UI for human feedback")
            await utils.post(name, (requests, scores, stats), delete=False)
            response = await conn.blpop(f"answers-{name}", timeout=60)

            if response is not None:
                response_feedback = msgpack.loads(response[1])
                winner_frontend = list(response_feedback.values())[0]

                answer = [{
                    "i": requests[0]["i"],
                    "j": requests[0]["i"],
                    "sampler": name,
                    "ident":  current_routes[winner_frontend]["ident"],
                    "pair":   [r["ident"] for r in current_routes],
                    "winner": winner_frontend,
                    "rank":   [1 if i == winner_frontend else 2 for i in range(2)],
                    "routes": current_routes,
                }]
                
                try:
                    state = await _get_state(name)
                    fut   = aclient.submit(
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
                logger.warning("Received no human feedback from frontend in time")
                
            await asyncio.sleep(0.5)

        if k % 10 == 0:
            uptime = time() - _start
            await conn.set(f"sampler-{name}-uptime", uptime)
        if await utils.should_stop(name):
            await asyncio.sleep(1)
            break

        if not seed_plan_processed and current_routes[0].get("tag") == "seed_route":
            seed_plan_processed = True
        if seed_plan_processed and current_routes[0].get("tag") != "seed_route":
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






// Update the fetchCurrentEvaluation function in active.html
// This should replace the existing fetchCurrentEvaluation function

function fetchCurrentEvaluation() {
  // Get current evaluation from API
  $.ajax({
    url: '/api/' + sampler + '/current-evaluation',
    method: 'GET',
    success: function(data) {
      console.log('Current evaluation data:', data); // Debug log
      
      if (data) {
        // Handle both cases: data.request or data directly containing routes
        var evalData = data.request || data;
        
        if (evalData && evalData.routes && evalData.routes.length >= 2) {
          updateCurrentComparison({
            routes: evalData.routes,
            comparison_type: data.comparison_type || evalData.comparison_type || 'unknown',
            score: data.score || evalData.score || 0
          });
          
          if (data.auto_decision !== null && data.auto_decision !== undefined) {
            document.getElementById('auto-eval-result').style.display = 'block';
            document.getElementById('auto-choice').textContent = 'Route ' + (data.auto_decision + 1);
          } else {
            document.getElementById('auto-eval-result').style.display = 'none';
          }
        } else {
          // No current evaluation, show loading
          document.getElementById('loading').style.display = 'block';
          document.getElementById('comparison-content').style.display = 'none';
        }
      }
    },
    error: function(xhr, status, error) {
      console.error('Error fetching current evaluation:', error);
      document.getElementById('loading').style.display = 'block';
      document.getElementById('comparison-content').style.display = 'none';
    }
  });

  // Update stats
  $.get('/api/' + sampler + '/evaluation-stats', function(data) {
    document.getElementById('total-comparisons').textContent = data.total || 0;
    document.getElementById('auto-evaluated').textContent = data.auto || 0;
    document.getElementById('human-evaluated').textContent = data.human || 0;
  }).fail(function() {
    console.error('Failed to fetch evaluation stats');
  });
}

// Update the updateCurrentComparison function to handle the rendered images
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

    // The route_image field now contains the rendered image from api.py
    if (route1.route_image) {
        document.getElementById('route-1-img').src = 'data:image/png;base64,' + route1.route_image;
    } else {
        console.warn('No route_image for route 1');
        document.getElementById('route-1-img').src = '/static/images/no-route.png';
    }

    if (route2.route_image) {
        document.getElementById('route-2-img').src = 'data:image/png;base64,' + route2.route_image;
    } else {
        console.warn('No route_image for route 2');
        document.getElementById('route-2-img').src = '/static/images/no-route.png';
    }

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
    document.getElementById('comparison-type').textContent = data.comparison_type || 'unknown';
    document.getElementById('request-score').textContent = (data.score || 0).toFixed(3);

    currentRequest = data;
}

// Initialize and start auto-refresh
var currentRequest = null;
var refreshInterval;

$(document).ready(function() {
    // Initial fetch
    fetchCurrentEvaluation();
    
    // Check auto-evaluator status
    checkAutoEvaluator();
    
    // Set up auto-refresh every 2 seconds
    refreshInterval = setInterval(function() {
        fetchCurrentEvaluation();
    }, 2000);
    
    // Check auto-evaluator every 5 seconds
    setInterval(checkAutoEvaluator, 5000);
});

function checkAutoEvaluator() {
    $.get('/api/meta/' + sampler, function(data) {
        var hasAutoEval = data.auto_evaluator_enabled || false;
        var statusBadge = document.getElementById('auto-eval-status');
        if (hasAutoEval) {
            statusBadge.textContent = 'Enabled';
            statusBadge.className = 'badge bg-success';
        } else {
            statusBadge.textContent = 'Disabled';
            statusBadge.className = 'badge bg-secondary';
        }
    }).fail(function() {
        document.getElementById('auto-eval-status').textContent = 'Unknown';
        document.getElementById('auto-eval-status').className = 'badge bg-warning';
    });
}






# Update this endpoint in _private.py

@app.get("/api/{sampler}/current-evaluation", tags=["experiment state"])
async def get_current_evaluation(sampler: str, username: str = Depends(_get_username)):
    """Get the current evaluation being processed with rendered route images"""
    await _ensure_sampler_present(sampler)
    
    # Get current evaluation from Redis (stored by api.py)
    current_req_key = f"current-evaluation-{sampler}"
    current_data = await rj.get(current_req_key)
    
    if current_data:
        try:
            data = msgpack.loads(current_data)
            # The data should already include route_image fields from api.py
            return data
        except Exception as e:
            logger.warning(f"Failed to load current evaluation data: {e}")
    
    # If no current evaluation in Redis, return empty
    return {
        "request": None,
        "routes": None,
        "auto_decision": None,
        "comparison_type": None,
        "score": None
    }





