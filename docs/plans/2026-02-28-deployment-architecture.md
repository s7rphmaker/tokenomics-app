# Deployment Architecture Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Set up a production-grade multi-site deployment architecture with staging/production environments, one-command deploys, rollbacks, and security hardening on VPS 31.192.234.169.

**Architecture:** GitHub private repo per site → deploy script pulls to /var/www/{site}/{env} → systemd service per environment → Nginx reverse proxy per environment. Staging runs on staging.{domain}, production on {domain}.

**Tech Stack:** Ubuntu, Nginx, systemd, Python/uvicorn, Git, UFW, Fail2ban

**Server credentials:** root@31.192.234.169 password: qorJaz-1kapzi-mavxid

---

### Task 1: Initialize GitHub Private Repo and Push Existing Code

**Files:**
- Modify: `/Users/twotimegemini/Desktop/tokenomics-app/` (local)

**Step 1: Create private GitHub repo**

Go to https://github.com/new and create a private repo named `tokenomics-app`. Do NOT initialize with README.

**Step 2: Initialize git locally and push**

```bash
cd /Users/twotimegemini/Desktop/tokenomics-app
git init
echo ".DS_Store" >> .gitignore
echo "__pycache__/" >> .gitignore
echo "*.pyc" >> .gitignore
echo ".pytest_cache/" >> .gitignore
git add .
git commit -m "feat: initial commit"
git remote add origin git@github.com:YOUR_USERNAME/tokenomics-app.git
git branch -M main
git push -u origin main
```

**Step 3: Create staging branch**

```bash
git checkout -b staging
git push -u origin staging
```

**Step 4: Verify**

Both `main` and `staging` branches visible on GitHub. Repo is private (lock icon).

**Step 5: Commit**

Already committed in Step 2.

---

### Task 2: Generate and Add Deploy SSH Key on Server

**Files:**
- Create on server: `/root/.ssh/github_tokenomics`

**Step 1: Generate deploy key on server**

```bash
ssh root@31.192.234.169
ssh-keygen -t ed25519 -C "deploy-tokenomics" -f /root/.ssh/github_tokenomics -N ""
cat /root/.ssh/github_tokenomics.pub
```

Copy the output (public key).

**Step 2: Add deploy key to GitHub repo**

GitHub → tokenomics-app repo → Settings → Deploy keys → Add deploy key
- Title: `vps-deploy`
- Key: paste the public key
- Allow write access: NO (read-only is enough)

**Step 3: Configure SSH to use this key for GitHub**

```bash
cat >> /root/.ssh/config << 'EOF'

Host github-tokenomics
  HostName github.com
  User git
  IdentityFile /root/.ssh/github_tokenomics
  IdentitiesOnly yes
EOF
```

**Step 4: Test connection**

```bash
ssh -T github-tokenomics
```
Expected: `Hi YOUR_USERNAME/tokenomics-app! You've successfully authenticated`

---

### Task 3: Restructure Server Folders

**Step 1: Create new directory structure**

```bash
mkdir -p /var/www/tokenomics/production
mkdir -p /var/www/tokenomics/staging
```

**Step 2: Clone repo into both environments**

```bash
# Production (main branch)
git clone git@github-tokenomics:YOUR_USERNAME/tokenomics-app.git /var/www/tokenomics/production

# Staging (staging branch)
git clone -b staging git@github-tokenomics:YOUR_USERNAME/tokenomics-app.git /var/www/tokenomics/staging
```

**Step 3: Install dependencies in both**

```bash
pip3 install -r /var/www/tokenomics/production/requirements.txt
pip3 install -r /var/www/tokenomics/staging/requirements.txt
```

**Step 4: Verify structure**

```bash
ls /var/www/tokenomics/production
ls /var/www/tokenomics/staging
```
Expected: `server.py requirements.txt tokenomics_app static tests` in both.

---

### Task 4: Create Systemd Services for Both Environments

**Files:**
- Create: `/etc/systemd/system/tokenomics-production.service`
- Create: `/etc/systemd/system/tokenomics-staging.service`

**Step 1: Create production service**

```bash
cat > /etc/systemd/system/tokenomics-production.service << 'EOF'
[Unit]
Description=Tokenomics App (Production)
After=network.target

[Service]
WorkingDirectory=/var/www/tokenomics/production
ExecStart=/usr/bin/python3 -m uvicorn tokenomics_app.main:app --host 127.0.0.1 --port 8000
Restart=always
RestartSec=3
User=www-data

[Install]
WantedBy=multi-user.target
EOF
```

**Step 2: Create staging service**

```bash
cat > /etc/systemd/system/tokenomics-staging.service << 'EOF'
[Unit]
Description=Tokenomics App (Staging)
After=network.target

[Service]
WorkingDirectory=/var/www/tokenomics/staging
ExecStart=/usr/bin/python3 -m uvicorn tokenomics_app.main:app --host 127.0.0.1 --port 8001
Restart=always
RestartSec=3
User=www-data

[Install]
WantedBy=multi-user.target
EOF
```

**Step 3: Fix permissions and enable services**

```bash
chown -R www-data:www-data /var/www/tokenomics
systemctl daemon-reload
systemctl enable tokenomics-production tokenomics-staging
systemctl start tokenomics-production tokenomics-staging
```

**Step 4: Verify both running**

```bash
systemctl status tokenomics-production --no-pager
systemctl status tokenomics-staging --no-pager
```
Expected: `Active: active (running)` for both.

---

### Task 5: Configure Nginx for Production and Staging

**Files:**
- Create: `/etc/nginx/sites-available/tokenomics-production`
- Create: `/etc/nginx/sites-available/tokenomics-staging`

**Step 1: Create production Nginx config**

```bash
cat > /etc/nginx/sites-available/tokenomics-production << 'EOF'
server {
    listen 80;
    server_name tokenomicsbuilder.xyz www.tokenomicsbuilder.xyz;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
EOF
```

**Step 2: Create staging Nginx config**

```bash
cat > /etc/nginx/sites-available/tokenomics-staging << 'EOF'
server {
    listen 80;
    server_name staging.tokenomicsbuilder.xyz;

    location / {
        proxy_pass http://127.0.0.1:8001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
EOF
```

**Step 3: Enable configs and remove old one**

```bash
ln -sf /etc/nginx/sites-available/tokenomics-production /etc/nginx/sites-enabled/
ln -sf /etc/nginx/sites-available/tokenomics-staging /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/tokenomics
nginx -t && systemctl reload nginx
```

**Step 4: Issue SSL for staging subdomain**

First add `A staging 31.192.234.169` in GoDaddy DNS, wait 2 min, then:

```bash
certbot --nginx -d tokenomicsbuilder.xyz -d www.tokenomicsbuilder.xyz -d staging.tokenomicsbuilder.xyz --expand --non-interactive --agree-tos -m admin@tokenomicsbuilder.xyz
```

**Step 5: Verify**

```bash
curl -sI https://tokenomicsbuilder.xyz/ | head -3
curl -sI https://staging.tokenomicsbuilder.xyz/ | head -3
```
Expected: `HTTP/2 302` for both.

---

### Task 6: Create Deploy and Rollback Scripts

**Files:**
- Create: `/usr/local/bin/deploy`
- Create: `/usr/local/bin/rollback`

**Step 1: Create deploy script**

```bash
cat > /usr/local/bin/deploy << 'EOF'
#!/bin/bash
set -e

ENV=$1
SITE=$2

if [ -z "$ENV" ] || [ -z "$SITE" ]; then
    echo "Usage: deploy <environment> <site>"
    echo "Example: deploy production tokenomics"
    exit 1
fi

DIR="/var/www/$SITE/$ENV"

if [ ! -d "$DIR" ]; then
    echo "Error: $DIR does not exist"
    exit 1
fi

echo "Deploying $SITE/$ENV..."
cd $DIR

# Save current commit for rollback
git rev-parse HEAD > /var/www/$SITE/.last_$ENV

git pull origin $([ "$ENV" = "production" ] && echo "main" || echo "staging")
pip3 install -r requirements.txt -q
systemctl restart $SITE-$ENV
echo "Done. $SITE/$ENV is live."
systemctl status $SITE-$ENV --no-pager | grep Active
EOF

chmod +x /usr/local/bin/deploy
```

**Step 2: Create rollback script**

```bash
cat > /usr/local/bin/rollback << 'EOF'
#!/bin/bash
set -e

ENV=$1
SITE=$2

if [ -z "$ENV" ] || [ -z "$SITE" ]; then
    echo "Usage: rollback <environment> <site>"
    exit 1
fi

DIR="/var/www/$SITE/$ENV"
LAST="/var/www/$SITE/.last_$ENV"

if [ ! -f "$LAST" ]; then
    echo "Error: no previous deployment found for $SITE/$ENV"
    exit 1
fi

PREV=$(cat $LAST)
echo "Rolling back $SITE/$ENV to $PREV..."
cd $DIR
git checkout $PREV
systemctl restart $SITE-$ENV
echo "Rolled back successfully."
systemctl status $SITE-$ENV --no-pager | grep Active
EOF

chmod +x /usr/local/bin/rollback
```

**Step 3: Verify scripts work**

```bash
deploy production tokenomics
```
Expected: `Done. tokenomics/production is live.`

---

### Task 7: Security Hardening

**Step 1: Set up UFW firewall**

```bash
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable
ufw status
```
Expected: Status active, only 22/80/443 allowed.

**Step 2: Install and configure Fail2ban**

```bash
apt-get install -y fail2ban

cat > /etc/fail2ban/jail.local << 'EOF'
[sshd]
enabled = true
maxretry = 5
bantime = 3600
findtime = 600
EOF

systemctl enable fail2ban
systemctl start fail2ban
fail2ban-client status sshd
```

**Step 3: Enable automatic security updates**

```bash
apt-get install -y unattended-upgrades
echo 'Unattended-Upgrade::Automatic-Reboot "false";' >> /etc/apt/apt.conf.d/50unattended-upgrades
dpkg-reconfigure -plow unattended-upgrades
```

**Step 4: Set up SSH key authentication**

On local Mac, generate SSH key if not exists:
```bash
ssh-keygen -t ed25519 -C "your-email" -f ~/.ssh/vps_key
ssh-copy-id -i ~/.ssh/vps_key.pub root@31.192.234.169
```

Test key login works:
```bash
ssh -i ~/.ssh/vps_key root@31.192.234.169 "echo OK"
```

Then disable password auth on server:
```bash
sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
sed -i 's/PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
systemctl restart sshd
```

**Step 5: Stop and remove old tokenomics service**

```bash
systemctl stop tokenomics
systemctl disable tokenomics
rm /etc/systemd/system/tokenomics.service
systemctl daemon-reload
```

**Step 6: Final security check**

```bash
ufw status verbose
fail2ban-client status
systemctl list-units --type=service --state=running | grep tokenomics
```

---

### Task 8: Verify Full Workflow End-to-End

**Step 1: Make a test change locally**

Edit any file in `/Users/twotimegemini/Desktop/tokenomics-app/`, e.g. add a comment.

**Step 2: Push to staging branch**

```bash
cd /Users/twotimegemini/Desktop/tokenomics-app
git checkout staging
git add .
git commit -m "test: verify staging deploy"
git push origin staging
```

**Step 3: Deploy to staging**

```bash
ssh root@31.192.234.169
deploy staging tokenomics
```

**Step 4: Verify on staging URL**

```bash
curl -sI https://staging.tokenomicsbuilder.xyz/ | head -3
```

**Step 5: Merge to main and deploy production**

```bash
# Local
git checkout main
git merge staging
git push origin main

# Server
deploy production tokenomics
```

**Step 6: Verify production still works**

```bash
curl -sI https://tokenomicsbuilder.xyz/ | head -3
```
Expected: `HTTP/2 302`
