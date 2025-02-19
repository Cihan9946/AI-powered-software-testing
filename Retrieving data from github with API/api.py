import os
import subprocess
import requests

# Token'i doğrudan burada tanımla (⚠️ Güvenlik riski!)
token = "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"

# API başlıkları
headers = {'Authorization': f'Bearer {token}'}

# Konu başlığı (örnek: Machine Learning projeleri)
topic = "machine-learning"

# GitHub API URL'si
url = f"https://api.github.com/search/repositories?q=topic:{topic}&sort=stars&order=desc&per_page=3"

# API isteği gönder
response = requests.get(url, headers=headers)

# Yanıtı kontrol et
if response.status_code == 200:
    repositories = response.json()["items"]
    
    # Klasörü oluştur (varsa hata vermez)
    os.makedirs("ml_projects", exist_ok=True)

    for repo in repositories:
        repo_name = repo['name']
        repo_url = repo['clone_url']
        save_path = os.path.join("ml_projects", repo_name)
        
        # Eğer repo daha önce klonlanmamışsa indir
        if not os.path.exists(save_path):
            print(f"Klonlanıyor: {repo_name} -> {repo_url}")
            subprocess.run(["git", "clone", repo_url, save_path])
        else:
            print(f"Zaten mevcut: {repo_name}")

else:
    print(f"Error: {response.status_code}, {response.text}")
