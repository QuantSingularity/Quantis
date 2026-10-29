#!/bin/bash
# Enhanced User Data Script for Financial-Grade EC2 Instances
# This script hardens the instance and sets up the application environment

set -euo pipefail

# Variables from Terraform
APP_NAME="${app_name}"
ENVIRONMENT="${environment}"
SSM_PARAMETER_PATH="${ssm_parameter_path}"
S3_BUCKET="${s3_bucket}"
KMS_KEY_ID="${kms_key_id}"

# Logging setup
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1
echo "Starting user data script execution at $(date)"

# Update system packages
yum update -y

# Install required packages
yum install -y \
    awscli \
    amazon-cloudwatch-agent \
    amazon-ssm-agent \
    htop \
    iotop \
    sysstat \
    fail2ban \
    aide \
    rkhunter \
    chkrootkit \
    lynis \
    docker \
    git \
    curl \
    wget \
    unzip \
    jq

# Security hardening
echo "Applying security hardening..."

# Disable unused services
systemctl disable bluetooth
systemctl disable cups
systemctl disable avahi-daemon

# Configure fail2ban
cat > /etc/fail2ban/jail.local << 'EOF'
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 3
backend = systemd

[sshd]
enabled = true
port = ssh
logpath = /var/log/secure
maxretry = 3
bantime = 3600
EOF

systemctl enable fail2ban
systemctl start fail2ban

# Configure SSH hardening
sed -i 's/#PermitRootLogin yes/PermitRootLogin no/' /etc/ssh/sshd_config
sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/' /etc/ssh/sshd_config
sed -i 's/#Protocol 2/Protocol 2/' /etc/ssh/sshd_config
sed -i 's/#MaxAuthTries 6/MaxAuthTries 3/' /etc/ssh/sshd_config
sed -i 's/#ClientAliveInterval 0/ClientAliveInterval 300/' /etc/ssh/sshd_config
sed -i 's/#ClientAliveCountMax 3/ClientAliveCountMax 2/' /etc/ssh/sshd_config

# Add additional SSH security
echo "AllowUsers ec2-user" >> /etc/ssh/sshd_config
echo "DenyUsers root" >> /etc/ssh/sshd_config

systemctl restart sshd

# Configure kernel parameters for security
cat > /etc/sysctl.d/99-security.conf << 'EOF'
# IP Spoofing protection
net.ipv4.conf.default.rp_filter = 1
net.ipv4.conf.all.rp_filter = 1

# Ignore ICMP redirects
net.ipv4.conf.all.accept_redirects = 0
net.ipv6.conf.all.accept_redirects = 0
net.ipv4.conf.default.accept_redirects = 0
net.ipv6.conf.default.accept_redirects = 0

# Ignore send redirects
net.ipv4.conf.all.send_redirects = 0
net.ipv4.conf.default.send_redirects = 0

# Disable source packet routing
net.ipv4.conf.all.accept_source_route = 0
net.ipv6.conf.all.accept_source_route = 0
net.ipv4.conf.default.accept_source_route = 0
net.ipv6.conf.default.accept_source_route = 0

# Log Martians
net.ipv4.conf.all.log_martians = 1
net.ipv4.conf.default.log_martians = 1

# Ignore ICMP ping requests
net.ipv4.icmp_echo_ignore_all = 1

# Ignore Directed pings
net.ipv4.icmp_echo_ignore_broadcasts = 1

# Disable IPv6 if not needed
net.ipv6.conf.all.disable_ipv6 = 1
net.ipv6.conf.default.disable_ipv6 = 1

# TCP SYN flood protection
net.ipv4.tcp_syncookies = 1
net.ipv4.tcp_max_syn_backlog = 2048
net.ipv4.tcp_synack_retries = 2
net.ipv4.tcp_syn_retries = 5
EOF

sysctl -p /etc/sysctl.d/99-security.conf

# Configure file system security
echo "Configuring file system security..."

# Set proper permissions on sensitive files
chmod 600 /etc/ssh/sshd_config
chmod 600 /etc/shadow
chmod 600 /etc/gshadow
chmod 644 /etc/passwd
chmod 644 /etc/group

# Configure umask for more restrictive default permissions
echo "umask 027" >> /etc/profile
echo "umask 027" >> /etc/bashrc

# Configure audit logging
yum install -y audit
systemctl enable auditd
systemctl start auditd

# Add audit rules for financial compliance
cat > /etc/audit/rules.d/financial-compliance.rules << 'EOF'
# Monitor file access
-w /etc/passwd -p wa -k identity
-w /etc/group -p wa -k identity
-w /etc/shadow -p wa -k identity
-w /etc/sudoers -p wa -k identity

# Monitor authentication events
-w /var/log/auth.log -p wa -k authentication
-w /var/log/secure -p wa -k authentication

# Monitor network configuration
-w /etc/network/ -p wa -k network
-w /etc/sysconfig/network-scripts/ -p wa -k network

# Monitor system calls
-a always,exit -F arch=b64 -S adjtimex -S settimeofday -k time-change
-a always,exit -F arch=b32 -S adjtimex -S settimeofday -S stime -k time-change
-a always,exit -F arch=b64 -S clock_settime -k time-change
-a always,exit -F arch=b32 -S clock_settime -k time-change

# Monitor file deletions
-a always,exit -F arch=b64 -S unlink -S unlinkat -S rename -S renameat -F auid>=1000 -F auid!=4294967295 -k delete
-a always,exit -F arch=b32 -S unlink -S unlinkat -S rename -S renameat -F auid>=1000 -F auid!=4294967295 -k delete
EOF

service auditd restart

# Install and configure CloudWatch agent
echo "Configuring CloudWatch agent..."

cat > /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json << EOF
{
    "agent": {
        "metrics_collection_interval": 60,
        "run_as_user": "cwagent"
    },
    "logs": {
        "logs_collected": {
            "files": {
                "collect_list": [
                    {
                        "file_path": "/var/log/messages",
                        "log_group_name": "/aws/ec2/$APP_NAME-$ENVIRONMENT",
                        "log_stream_name": "{instance_id}/messages"
                    },
                    {
                        "file_path": "/var/log/secure",
                        "log_group_name": "/aws/ec2/$APP_NAME-$ENVIRONMENT",
                        "log_stream_name": "{instance_id}/secure"
                    },
                    {
                        "file_path": "/var/log/audit/audit.log",
                        "log_group_name": "/aws/ec2/$APP_NAME-$ENVIRONMENT",
                        "log_stream_name": "{instance_id}/audit"
                    },
                    {
                        "file_path": "/var/log/application.log",
                        "log_group_name": "/aws/ec2/$APP_NAME-$ENVIRONMENT",
                        "log_stream_name": "{instance_id}/application"
                    }
                ]
            }
        }
    },
    "metrics": {
        "namespace": "CWAgent",
        "metrics_collected": {
            "cpu": {
                "measurement": [
                    "cpu_usage_idle",
                    "cpu_usage_iowait",
                    "cpu_usage_user",
                    "cpu_usage_system"
                ],
                "metrics_collection_interval": 60,
                "totalcpu": false
            },
            "disk": {
                "measurement": [
                    "used_percent"
                ],
                "metrics_collection_interval": 60,
                "resources": [
                    "*"
                ]
            },
            "diskio": {
                "measurement": [
                    "io_time"
                ],
                "metrics_collection_interval": 60,
                "resources": [
                    "*"
                ]
            },
            "mem": {
                "measurement": [
                    "mem_used_percent"
                ],
                "metrics_collection_interval": 60
            },
            "netstat": {
                "measurement": [
                    "tcp_established",
                    "tcp_time_wait"
                ],
                "metrics_collection_interval": 60
            },
            "swap": {
                "measurement": [
                    "swap_used_percent"
                ],
                "metrics_collection_interval": 60
            }
        }
    }
}
EOF

# Start CloudWatch agent
/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
    -a fetch-config \
    -m ec2 \
    -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json \
    -s

# Configure Docker
echo "Configuring Docker..."
systemctl enable docker
systemctl start docker

# Add ec2-user to docker group
usermod -a -G docker ec2-user

# Configure Docker daemon for security
mkdir -p /etc/docker
cat > /etc/docker/daemon.json << 'EOF'
{
    "log-driver": "json-file",
    "log-opts": {
        "max-size": "10m",
        "max-file": "3"
    },
    "live-restore": true,
    "userland-proxy": false,
    "no-new-privileges": true
}
EOF

systemctl restart docker

# Mount and format data volume
echo "Configuring data volume..."
if [ -b /dev/xvdf ]; then
    # Check if volume is already formatted
    if ! blkid /dev/xvdf; then
        mkfs.ext4 /dev/xvdf
    fi

    # Create mount point
    mkdir -p /opt/app-data

    # Add to fstab
    echo "/dev/xvdf /opt/app-data ext4 defaults,noatime 0 2" >> /etc/fstab

    # Mount the volume
    mount /opt/app-data

    # Set proper ownership and permissions
    chown ec2-user:ec2-user /opt/app-data
    chmod 750 /opt/app-data
fi

# Create application directories
mkdir -p /opt/app/{bin,config,logs,tmp}
chown -R ec2-user:ec2-user /opt/app
chmod -R 750 /opt/app

# Set up log rotation for application logs
cat > /etc/logrotate.d/application << 'EOF'
/opt/app/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 640 ec2-user ec2-user
    postrotate
        /bin/kill -HUP `cat /var/run/rsyslogd.pid 2> /dev/null` 2> /dev/null || true
    endscript
}
EOF

# Install application dependencies
echo "Installing application dependencies..."

# Install Node.js (for frontend builds)
curl -fsSL https://rpm.nodesource.com/setup_18.x | bash -
yum install -y nodejs

# Install Python 3.9 and pip
yum install -y python3 python3-pip

# Install application-specific packages
pip3 install --upgrade pip
pip3 install \
    flask \
    gunicorn \
    psycopg2-binary \
    redis \
    celery \
    boto3 \
    cryptography \
    requests \
    sqlalchemy \
    alembic

# Create application user
useradd -r -s /bin/false -d /opt/app appuser
usermod -a -G docker appuser

# Set up systemd service for the application
cat > /etc/systemd/system/quantis-app.service << 'EOF'
[Unit]
Description=Quantis Application
After=network.target docker.service
Requires=docker.service

[Service]
Type=forking
User=appuser
Group=appuser
WorkingDirectory=/opt/app
ExecStart=/opt/app/bin/start.sh
ExecStop=/opt/app/bin/stop.sh
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=quantis-app

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/app /opt/app-data

[Install]
WantedBy=multi-user.target
EOF

# Enable the service (but don't start it yet)
systemctl enable quantis-app

# Configure firewall
echo "Configuring firewall..."
yum install -y iptables-services
systemctl enable iptables

# Basic iptables rules
cat > /etc/sysconfig/iptables << 'EOF'
*filter
:INPUT DROP [0:0]
:FORWARD DROP [0:0]
:OUTPUT ACCEPT [0:0]

# Allow loopback
-A INPUT -i lo -j ACCEPT

# Allow established connections
-A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

# Allow SSH (port 22)
-A INPUT -p tcp --dport 22 -j ACCEPT

# Allow HTTP (port 80) and HTTPS (port 443)
-A INPUT -p tcp --dport 80 -j ACCEPT
-A INPUT -p tcp --dport 443 -j ACCEPT

# Allow application port (8000)
-A INPUT -p tcp --dport 8000 -j ACCEPT

# Allow health check port (8080)
-A INPUT -p tcp --dport 8080 -j ACCEPT

# Log dropped packets
-A INPUT -j LOG --log-prefix "iptables-dropped: "

COMMIT
EOF

systemctl start iptables

# Configure automatic security updates
echo "Configuring automatic security updates..."
yum install -y yum-cron

sed -i 's/update_cmd = default/update_cmd = security/' /etc/yum/yum-cron.conf
sed -i 's/apply_updates = no/apply_updates = yes/' /etc/yum/yum-cron.conf

systemctl enable yum-cron
systemctl start yum-cron

# Install and configure AIDE (Advanced Intrusion Detection Environment)
echo "Configuring AIDE..."
aide --init
mv /var/lib/aide/aide.db.new.gz /var/lib/aide/aide.db.gz

# Schedule daily AIDE checks
cat > /etc/cron.daily/aide-check << 'EOF'
#!/bin/bash
/usr/sbin/aide --check | /bin/mail -s "AIDE Report - $(hostname)" root
EOF

chmod +x /etc/cron.daily/aide-check

# Configure system monitoring
echo "Setting up system monitoring..."

# Install and configure htop
cat > /home/ec2-user/.htoprc << 'EOF'
fields=0 48 17 18 38 39 40 2 46 47 49 1
sort_key=46
sort_direction=1
hide_threads=0
hide_kernel_threads=1
hide_userland_threads=0
shadow_other_users=0
show_thread_names=0
show_program_path=1
highlight_base_name=0
highlight_megabytes=1
highlight_threads=1
tree_view=0
header_margin=1
detailed_cpu_time=0
cpu_count_from_zero=0
update_process_names=0
account_guest_in_cpu_meter=0
color_scheme=0
delay=15
left_meters=AllCPUs Memory Swap
left_meter_modes=1 1 1
right_meters=Tasks LoadAverage Uptime
right_meter_modes=2 2 2
EOF

chown ec2-user:ec2-user /home/ec2-user/.htoprc

# Set up performance monitoring
cat > /etc/cron.d/performance-monitoring << 'EOF'
# Performance monitoring every 5 minutes
*/5 * * * * root /usr/bin/iostat -x 1 1 >> /var/log/iostat.log
*/5 * * * * root /usr/bin/vmstat 1 1 >> /var/log/vmstat.log
*/5 * * * * root /usr/bin/free -m >> /var/log/memory.log
EOF

# Create startup script
cat > /opt/app/bin/start.sh << 'EOF'
#!/bin/bash
set -euo pipefail

# Source environment variables
source /opt/app/config/environment

# Start the application
cd /opt/app
exec gunicorn --bind 0.0.0.0:8000 --workers 4 --timeout 120 app:app
EOF

chmod +x /opt/app/bin/start.sh

# Create stop script
cat > /opt/app/bin/stop.sh << 'EOF'
#!/bin/bash
pkill -f gunicorn || true
EOF

chmod +x /opt/app/bin/stop.sh

# Create health check script
cat > /opt/app/bin/health-check.sh << 'EOF'
#!/bin/bash
curl -f http://localhost:8080/health || exit 1
EOF

chmod +x /opt/app/bin/health-check.sh

# Set up health check endpoint
cat > /opt/app/health.py << 'EOF'
from flask import Flask, jsonify
import psutil
import os

app = Flask(__name__)

@app.route('/health')
def health_check():
    try:
        # Check system resources
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        # Basic health checks
        if cpu_percent > 90:
            return jsonify({'status': 'unhealthy', 'reason': 'high_cpu'}), 503

        if memory.percent > 90:
            return jsonify({'status': 'unhealthy', 'reason': 'high_memory'}), 503

        if disk.percent > 90:
            return jsonify({'status': 'unhealthy', 'reason': 'high_disk'}), 503

        return jsonify({
            'status': 'healthy',
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'disk_percent': disk.percent
        })
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 503

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
EOF

# Create systemd service for health check
cat > /etc/systemd/system/quantis-health.service << 'EOF'
[Unit]
Description=Quantis Health Check Service
After=network.target

[Service]
Type=simple
User=appuser
Group=appuser
WorkingDirectory=/opt/app
ExecStart=/usr/bin/python3 /opt/app/health.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl enable quantis-health
systemctl start quantis-health

# Final security scan
echo "Running final security scan..."
lynis audit system --quick

# Signal completion
echo "User data script completed successfully at $(date)"

# Send completion notification to CloudWatch
aws logs put-log-events \
    --log-group-name "/aws/ec2/$APP_NAME-$ENVIRONMENT" \
    --log-stream-name "$(curl -s http://169.254.169.254/latest/meta-data/instance-id)/user-data" \
    --log-events timestamp=$(date +%s000),message="User data script completed successfully" \
    --region $(curl -s http://169.254.169.254/latest/meta-data/placement/region) || true

exit 0
