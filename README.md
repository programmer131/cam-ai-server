# cam-ai-server
a simple ai server, with same api as codeprojectAi/deepstack for use with agentDVR/Blueiris

first install ultralytics, usually done from single pip install command,

then run the server

point iSpyAgent to this server running at port 5000

now you have you IP cam setup running with yolo26 

create a service or start it with cron automatically on every system boot

# Reload systemd

sudo systemctl daemon-reload

# Enable and start

sudo systemctl enable yolo-server

sudo systemctl start yolo-server

# Check status

sudo systemctl status yolo-server
