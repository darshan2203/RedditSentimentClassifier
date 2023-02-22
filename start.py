import subprocess

subprocess.run("uvicorn modules.app:app --host 0.0.0.0 --port 8000", shell=True)
