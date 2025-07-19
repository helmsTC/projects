<!-- In index.html, update the menu to include the Active tab -->
<!-- Find this section in index.html: -->
<div class="menubar" id="menu">
  <ul class="menu">
      <li class="menuItem"><a class="menuLink active" id="indexMenuLink">Home</a></li>
      <li class="menuItem"><a class="menuLink" id="docsLink" href="/docs/" target="_blank" rel="noopener noreferrer">Documentation</a></li>
      <li class="menuItem"><a class="menuLink" id="classMenuLink">Classroom</a></li>
      <li class="menuItem"><a class="menuLink" id="rankingsMenuLink">Report Card</a></li>
      <li class="menuItem"><a class="menuLink" id="dashMenuLink">Office</a></li>
      <li class="menuItem"><a class="menuLink" id="activeMenuLink">Active</a></li> <!-- ADD THIS LINE -->
      <li class="menuImg"><div class="menuImg-container"><img src="/static/images/AFRL_logo.png"></div></li>
      <li class="menuImg"><div class="menuImg-container"><img src="/static/images/AUKUS_logo.png"></div></li>
      <li class="menuImg"><div class="menuImg-container top-pad"><img src="/static/images/DSTL_logo.png"></div></li>
  </ul>
</div>

<!-- Also update the JavaScript section to set the Active link href: -->
<script>
  const url = new URL(window.location.href);
  const sampler = url.searchParams.get('sampler');
  var urlParams = new URLSearchParams(window.location.search);

  if (sampler == null) {
    var tabs = ["indexMenuLink", "docsLink", "classMenuLink", "rankingsMenuLink", "dashMenuLink", "activeMenuLink"]; // ADD activeMenuLink
    for (var i=0; i<tabs.length; i++) {
      var m = document.getElementById(tabs[i]);
      if (i >= 2){
        m.style.display = "none";
      }
    }
  }
  document.getElementById("dashMenuLink").setAttribute("href", "/dashboard/" + sampler);
  document.getElementById("classMenuLink").setAttribute("href", "/classroom/" + sampler);
  document.getElementById("rankingsMenuLink").setAttribute("href", "/rankings/" + sampler);
  document.getElementById("activeMenuLink").setAttribute("href", "/active/" + sampler); // ADD THIS LINE
</script>



# Add this to _gui.py after the other routes (like after _dashboard)

@app.get("/active/{sampler}", tags=["GUI"])
async def _active(request: Request, sampler: str, username: str = Depends(_get_username)):
    """Serve the active evaluation page"""
    # Verify sampler exists
    samplers = await get_samplers(pattern=sampler)
    if not samplers or sampler not in samplers:
        raise HTTPException(status_code=404, detail=f"Sampler {sampler} not found")
    
    return templates.TemplateResponse("active.html", {"request": request, "sampler": sampler})





// In index.html, update the refresh_experiments function to include Active link:

function refresh_experiments() {
    $.get("/api/meta", function (data) {
      data = data["samplers"];
      console.log(data);
      if (Array(data).length) {
        rows = ["<thead><tr><th>Experiment name</th>" +
                "<th>Classroom</th>" +
                "<th>Report Card</th>" +
                "<th>Office</th>" +
                "<th>Active</th></tr></thead>"];  // ADD Active header

        href = function(url) {
            return "<a href='" + url + "'>" + url + "</a>";
        }
        for (var name in data){
          var datum = data[name];
          if (name == "meta"){
              continue;
          }
          const row = "<tr><td>" + datum["name"] + "</td>" +
                     "<td>" + href(datum["classroom_path"]) + "</td>" +
                     "<td>" + href(datum["results_path"]) + "</td>" +
                     "<td>" + href(datum["dashboard_path"]) + "</td>" +
                     "<td>" + href("/active/" + datum["name"]) + "</td></tr>";  // ADD Active link
          rows.push(row);
        }
        $("#expnav").html(rows.join(""));
      }
    });
}





# Run this script to save the active.html template
from pathlib import Path

# Get the templates directory path
templates_dir = Path(__file__).parent / "templates"
templates_dir.mkdir(exist_ok=True)

# The active.html content (from your uploaded file)
active_html_content = '''<!DOCTYPE html>
<html lang="en-US">
<head>
  <link rel="stylesheet" href="/static/bootstrap.min.css">
  <link href="/static/font-awesome/stylesheets/font-awesome.min.css" rel="stylesheet">
  <link href="/static/style.css" rel="stylesheet">
  <script type="text/javascript" src="/static/javascripts/jquery/jquery-3.5.1.js"></script>
  <script type="text/javascript" src="/static/javascripts/bootstrap/bootstrap.min.js"></script>
  <style>
    .evaluation-container {
      margin: 20px auto;
      max-width: 1200px;
    }
    .route-image {
      max-width: 400px;
      height: auto;
      border: 2px solid #ddd;
      padding: 10px;
      margin: 10px;
    }
    .route-comparison {
      display: flex;
      justify-content: center;
      align-items: flex-start;
      margin: 20px 0;
      flex-wrap: wrap;
    }
    .route-item {
      text-align: center;
      margin: 10px;
    }
    .route-label {
      font-weight: bold;
      margin-top: 10px;
    }
    .status-info {
      background: #f8f9fa;
      padding: 15px;
      border-radius: 5px;
      margin: 20px 0;
    }
    .auto-evaluation {
      background: #e7f3ff;
      padding: 10px;
      border-radius: 5px;
      margin: 10px 0;
    }
    .refresh-info {
      text-align: center;
      color: #666;
      margin: 10px 0;
    }
  </style>
</head>
<body>
  <div class="menubar">
    <ul class="menu">
      <li class="menuItem"><a class="menuLink" id="indexMenuLink">Home</a></li>
      <li class="menuItem"><a class="menuLink" id="docsLink" href="/docs/" target="_blank" rel="noopener noreferrer">Documentation</a></li>
      <li class="menuItem"><a class="menuLink" id="classMenuLink">Classroom</a></li>
      <li class="menuItem"><a class="menuLink" id="rankingsMenuLink">Report Card</a></li>
      <li class="menuItem"><a class="menuLink" id="dashMenuLink">Office</a></li>
      <li class="menuItem"><a class="menuLink active" id="activeMenuLink">Active</a></li>
      <li class="menuImg"><div class="menuImg-container"><img src="/static/images/AFRL_logo.png"></div></li>
      <li class="menuImg"><div class="menuImg-container"><img src="/static/images/AUKUS_logo.png"></div></li>
      <li class="menuImg"><div class="menuImg-container top-pad"><img src="/static/images/DSTL_logo.png"></div></li>
    </ul>
  </div>
  <br><br>
  <div class="title" id="title">
    Active Evaluations
  </div>

  <div class="evaluation-container">
    <div class="status-info">
      <h4>Current Status</h4>
      <p>Sampler: <strong id="sampler-name">{{ sampler }}</strong></p>
      <p>Auto-Evaluator: <span id="auto-eval-status" class="badge">Checking...</span></p>
      <p>Total Comparisons: <span id="total-comparisons">0</span></p>
      <p>Auto-Evaluated: <span id="auto-evaluated">0</span></p>
      <p>Human-Evaluated: <span id="human-evaluated">0</span></p>
    </div>

    <div class="refresh-info">
      <small>Auto-refreshing every 2 seconds</small>
    </div>

    <div id="current-evaluation">
      <h3>Current Comparison</h3>
      <div id="loading" style="text-align: center;">
        <p>Waiting for next comparison...</p>
      </div>
      
      <div id="comparison-content" style="display: none;">
        <div class="auto-evaluation" id="auto-eval-result" style="display: none;">
          <strong>Auto-Evaluator Decision:</strong> <span id="auto-choice"></span>
        </div>
        
        <div class="route-comparison">
          <div class="route-item">
            <img id="route-1-img" src="" alt="Route 1" class="route-image">
            <div class="route-label">
              Route 1
              <div id="route-1-info">
                <small>
                  <span id="route-1-tag"></span><br>
                  ID: <span id="route-1-id"></span>
                </small>
              </div>
            </div>
          </div>
          
          <div class="route-item">
            <img id="route-2-img" src="" alt="Route 2" class="route-image">
            <div class="route-label">
              Route 2
              <div id="route-2-info">
                <small>
                  <span id="route-2-tag"></span><br>
                  ID: <span id="route-2-id"></span>
                </small>
              </div>
            </div>
          </div>
        </div>
        
        <div style="text-align: center; margin-top: 20px;">
          <p>Comparison Type: <span id="comparison-type" class="badge bg-info"></span></p>
          <p>Request Score: <span id="request-score"></span></p>
        </div>
      </div>
    </div>

    <hr>

    <div id="recent-history">
      <h3>Recent Evaluations</h3>
      <table class="table table-striped">
        <thead>
          <tr>
            <th>Time</th>
            <th>Type</th>
            <th>Route 1</th>
            <th>Route 2</th>
            <th>Winner</th>
            <th>Evaluator</th>
          </tr>
        </thead>
        <tbody id="history-tbody">
        </tbody>
      </table>
    </div>
  </div>

  <script>
    var sampler = "{{sampler}}";
    document.getElementById("indexMenuLink").setAttribute("href", "/?sampler=" + sampler);
    document.getElementById("classMenuLink").setAttribute("href", "/classroom/" + sampler);
    document.getElementById("rankingsMenuLink").setAttribute("href", "/rankings/" + sampler);
    document.getElementById("dashMenuLink").setAttribute("href", "/dashboard/" + sampler);

    // Fixed JavaScript functions go here...
  </script>
</body>
</html>'''

# Save the file
active_template_path = templates_dir / "active.html"
active_template_path.write_text(active_html_content)
print(f"Saved active.html to {active_template_path}")








