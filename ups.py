@app.get("/api/{sampler}/requests", tags=["experiment state"])
async def get_sampler_requests(
    sampler: str,
    n: int = 1,
    username: str = Depends(_get_username),
) -> JSON:
    """Get pending requests for a sampler - proxy to backend"""
    await _ensure_sampler_present(sampler)
    
    # Get from the backend API
    async with httpx.AsyncClient() as httpclient:
        try:
            r = await httpclient.get(
                f"http://backendpython:6463/{sampler}/requests?n={n}",
                timeout=30,
            )
            if r.status_code == 200:
                return r.json()
        except:
            pass
    
    return {"data": []}

@app.get("/api/{sampler}/current-evaluation", tags=["experiment state"])
async def get_current_evaluation(
    sampler: str,
    username: str = Depends(_get_username),
) -> JSON:
    """Get current evaluation state for active view"""
    # Try to get the latest request from Redis
    request_key = f"sampler-{sampler}-requests"
    request_data = await rj.lindex(request_key, 0)  # Peek at first item
    
    if request_data:
        try:
            request = msgpack.loads(request_data)
            return {
                "request": request,
                "auto_decision": None  # You can implement auto-decision tracking if needed
            }
        except:
            pass
    
    return {"request": None, "auto_decision": None}




$.get('/api/' + sampler + '/stats/training', function(data) {  // Add '/training' to match your API
    if (data) {
        document.getElementById('total-comparisons').textContent = data.n_answers || 0;
    }
});
