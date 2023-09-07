# if error with qt when plt.show(), try belows.
# Tested on Ubuntu22.04 & Ubuntu20.04 WSL Distribution
# sudo apt update -q
# sudo apt install -y -q build-essential libgl1-mesa-dev

# sudo apt install -y -q libxkbcommon-x11-0
# sudo apt install -y -q libxcb-image0
# sudo apt install -y -q libxcb-keysyms1
# sudo apt install -y -q libxcb-render-util0
# sudo apt install -y -q libxcb-xinerama0
# sudo apt install -y -q libxcb-icccm4
# ==============================================
# export for Xming server display
export DISPLAY=:0.0
# if not connect the display, try belows.
# localhost:0.0
# 127.0.0.1:0.0
# $(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0
