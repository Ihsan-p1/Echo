import urllib.request
import re

url = "https://universe.roboflow.com/search?q=desk"
try:
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    html = urllib.request.urlopen(req).read().decode('utf-8')
    
    # regex to find hrefs like "/workspace-name/project-name"
    matches = re.findall(r'href="/([^/]+)/([^/]+)"', html)
    
    # filter out generic links
    excluded = ['search', 'login', 'signup', 'contact', 'about', 'blog', 'pricing', 'docs', 'explore']
    valid_projects = [m for m in matches if m[0] not in excluded and m[1] not in excluded]
    
    print("Found potential projects:")
    unique_matches = list(set(valid_projects))
    count = 0
    for m in unique_matches:
        if count >= 10: break
        print(f"Workspace: {m[0]}, Project: {m[1]}")
        count += 1
except Exception as e:
    print("Error:", e)
