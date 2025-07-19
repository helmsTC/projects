@app.get("/api/{sampler}/current-evaluation", tags=["experiment state"])
async def get_current_evaluation(
    sampler: str,
    username: str = Depends(_get_username),
) -> JSON:
    """Get the current evaluation state for the active view"""
    await _ensure_sampler_present(sampler)
    
    # Get the latest queryfrom Redis
    query_key = f"sampler-{sampler}-queries"
    query_data = await rj.lpop(query_key)
    
    if query_data:
        query= msgpack.loads(query_data)
        # Put it back since we're just peeking
        await rj.lpush(query_key, query_data)
        
        # Check if there's an auto-evaluation decision
        auto_decision = await rj.get(f"auto-eval-{sampler}-decision")
        
        return {
            "request": request,
            "auto_decision": int(auto_decision) if auto_decision else None
        }
    
    return {"request": None, "auto_decision": None}

@app.get("/api/{sampler}/queries", tags=["experiment state"])
async def get_queries(
    sampler: str,
    n: int = 1,
    username: str = Depends(_get_username),
) -> JSON:
    """Get pending queries for a sampler"""
    await _ensure_sampler_present(sampler)
    
    key = f"sampler-{sampler}-queries"
    queries = []
    
    for _ in range(n):
        blob = await rj.lpop(key)
        if blob is None:
            break
        queries.append(msgpack.loads(blob))
    
    if not queries:
        return {"data": []}
    
    return {"data": queries}



@app.get("/rankings/{sampler}")
async def rankings_page(sampler: str):
    # Get rankings data
    rankings = await get_rankings(sampler)
    
    # Generate plots
    renderer = Render()
    plot1_base64 = renderer.generate_performance_plot(rankings)
    plot2_base64 = renderer.generate_win_rate_plot(rankings)
    
    # Save plots or convert to data URLs
    plot1_path = f"data:image/png;base64,{plot1_base64}"
    plot2_path = f"data:image/png;base64,{plot2_base64}"
    
    return templates.TemplateResponse("rankings_routes.html", {
        "query": query,
        "sampler": sampler,
        "rankings": rankings,
        "plot1_path": plot1_path,
        "plot2_path": plot2_path,
        # ... other context variab
