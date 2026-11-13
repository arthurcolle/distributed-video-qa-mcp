import base64, os, sys, time, httpx
src, out_prefix, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
prompt = open(sys.argv[4]).read()
t = time.time()
with open(src, "rb") as f:
    r = httpx.post(
        "https://api.openai.com/v1/images/edits",
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        data={"model": "gpt-image-2", "prompt": prompt, "size": "3584x1792", "quality": "high", "n": str(n)},
        files={"image[]": ("art.png", f, "image/png")},
        timeout=600,
    )
print("status", r.status_code, f"{time.time() - t:.0f}s")
body = r.json()
if r.status_code != 200:
    print(body); sys.exit(1)
for i, item in enumerate(body["data"], 1):
    path = f"{out_prefix}_{i}.png"
    open(path, "wb").write(base64.b64decode(item["b64_json"]))
    print("saved", path)
print("usage", body.get("usage"))
