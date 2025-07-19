# Add this route to _gui.py after the other routes

@app.get("/active/{sampler}", tags=["GUI"])
async def _active(request: Request, sampler: str, username: str = Depends(_get_username)):
    """Serve the active evaluation page"""
    samplers = await get_samplers(pattern=sampler)
    if sampler not in samplers:
        return PlainTextResponse(f"sampler={sampler} not in samplers={samplers}")
    
    return templates.TemplateResponse("active.html", {"request": request, "sampler": sampler})



# Replace the existing _rankings function in _gui.py with this updated version

@app.get("/rankings/{sampler}", tags=["GUI"])
async def _rankings(request: Request, sampler: str, domain: str = "all", sort: str = "score", username: str = Depends(_get_username)):
    ranks = await api.get_rankings(sampler=sampler, domain=domain)

    # sorted by score by default
    sort_key = sort.replace("reverse", "")

    # low to high; reverse by default
    ranks = list(sorted(ranks, key = lambda x: x.get(sort_key, 0)))  # low to high
    if "reverse" not in sort:
        ranks = ranks[::-1]  # now high to low

    metacols = list({c for item in ranks for c in item.get("meta", {})})

    config = await api._get_config(sampler, username=username)
    objects = config["html"]["objects"]
    ranking_templates = {"images": "rankings_images.html", "routes": "rankings_routes.html"}
    kwargs = dict()
    
    if objects == "routes":
        cost_names = ranks[0]["plan"]["cost_names"] if len(ranks) else [""]

        model = await api.get_model(sampler=sampler, request=request, username=username)
        df = pd.DataFrame(dict(prefs=model["prefs"], names=cost_names))
        df["prefs"] /= 100

        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X('prefs:Q').axis(format='%').title("Importance"),
            y=alt.Y('names:O', sort=None).title("Costs"),
        ).properties(width=200)
        with io.StringIO() as f:
            chart.save(f, format="svg")
            chart_img = f.getvalue()
        
        # Generate placeholder plots for now
        # Later these can be replaced with actual plot generation using render.py
        plot1_path = None  # Placeholder for performance plot
        plot2_path = None  # Placeholder for win rate plot
        
        # Try to generate actual plots if render.py is available
        try:
            from render import Render
            renderer = Render()
            
            # Generate performance plot
            # This would need actual implementation based on your data
            # plot1_base64 = renderer.generate_performance_plot(ranks)
            # plot1_path = f"data:image/png;base64,{plot1_base64}"
            
            # Generate win rate plot  
            # plot2_base64 = renderer.generate_win_rate_plot(ranks)
            # plot2_path = f"data:image/png;base64,{plot2_base64}"
        except Exception as e:
            logger.warning(f"Could not generate plots: {e}")
        
        kwargs.update(
            cost_names=cost_names, 
            pref_chart=chart_img,
            plot1_path=plot1_path,
            plot2_path=plot2_path
        )

    return templates.TemplateResponse(
        ranking_templates[objects],
        {"request": request, "sampler": sampler, "rankings": ranks, "meta_cols": metacols, **kwargs}
    )




# Add these endpoints to _private.py (they're API endpoints, not HTML routes)

@app.get("/api/{sampler}/current-evaluation", tags=["experiment state"])
async def get_current_evaluation(sampler: str, username: str = Depends(_get_username)):
    """Get the current evaluation being processed"""
    await _ensure_sampler_present(sampler)
    
    # Check if there's a current request in Redis
    current_req_key = f"current-evaluation-{sampler}"
    current_data = await rj.get(current_req_key)
    
    if current_data:
        try:
            data = msgpack.loads(current_data)
            return data
        except:
            pass
    
    # Fallback: get the latest request
    key = f"reqs-{sampler}"
    latest_id = await rj.zrange(f"{key}-scores", -1, -1)
    
    if latest_id:
        req_data = await rj.hget(key, latest_id[0])
        if req_data:
            request = msgpack.loads(req_data)
            return {"request": request, "auto_decision": None}
    
    return {"request": None, "auto_decision": None}

@app.get("/api/{sampler}/evaluation-stats", tags=["experiment state"])
async def get_evaluation_stats(sampler: str, username: str = Depends(_get_username)):
    """Get statistics about evaluations"""
    await _ensure_sampler_present(sampler)
    
    # Get response data to calculate stats
    responses = await get_responses(sampler)
    total = len(responses)
    
    # Try to get auto/human counts from stored data
    auto_key = f"evaluation-auto-{sampler}"
    human_key = f"evaluation-human-{sampler}"
    
    auto = int(await rj.get(auto_key) or 0)
    human = int(await rj.get(human_key) or 0)
    
    # If no stored counts, estimate from responses
    if auto == 0 and human == 0 and total > 0:
        # You would need to determine this from response metadata
        # For now, just return estimates
        human = total
        auto = 0
    
    return {
        "total": total,
        "auto": auto,
        "human": human,
        "auto_percentage": (auto / total * 100) if total > 0 else 0
    }




# In api.py, update the get_requests_loop function
# Add this code after getting requests (around line 170-180 where you process requests)

        # After this line: requests, scores, stats = await request_future
        # Add:
        
        # Store current evaluation for frontend display
        if requests and len(requests) > 0:
            current_eval_data = {
                "request": requests[0],
                "timestamp": time(),
                "auto_decision": None
            }
            
            # Store in Redis for the Active page to retrieve
            await conn.set(
                f"current-evaluation-{name}", 
                msgpack.dumps(current_eval_data),
                ex=60  # Expire after 60 seconds
            )
        
        # After the evaluator makes a decision, update it:
        if winner is not None:
            current_eval_data["auto_decision"] = winner
            await conn.set(
                f"current-evaluation-{name}", 
                msgpack.dumps(current_eval_data),
                ex=60
            )
            
            # Increment auto-evaluation counter
            await conn.incr(f"evaluation-auto-{name}")
        else:
            # Increment human-evaluation counter when human feedback is needed
            await conn.incr(f"evaluation-human-{name}")
