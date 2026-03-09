import urllib.request
import json

url = "https://api.github.com/repos/kysrgit/Balik_Projesi/pulls?state=open"
try:
    with urllib.request.urlopen(url) as response:
        data = json.loads(response.read().decode())
        for pr in data:
            print(f"PR #{pr['number']}: {pr['title']} [Branch: {pr['head']['ref']}]")
except Exception as e:
    print(f"Error: {e}")
