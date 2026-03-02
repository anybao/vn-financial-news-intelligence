import requests

url = "https://restv2.fireant.vn/posts?type=1&offset=0&limit=10"
try:
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    print(response.status_code)
    print(response.json()[:2])
except Exception as e:
    print(e)
