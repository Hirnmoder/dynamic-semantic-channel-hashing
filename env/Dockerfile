FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

# Update and optionally install missing dependencies
RUN apt-get update && apt-get install -y \
   zsh git curl nano zip 7zip

COPY ./requirements*.txt .

# Install packages and development dependencies
RUN <<EOF
pip3 install -r requirements.txt
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
sed -i 's|ZSH_THEME="[^"]*"|ZSH_THEME="agnoster"|' ~/.zshrc
sed -i 's|^plugins=([^)]*)|plugins=(git z)|' ~/.zshrc
sed -i 's|⚡|🗲|' ~/.oh-my-zsh/themes/agnoster.zsh-theme
pip3 install -r requirements.dev.txt
adduser -u 5678 --disabled-password --gecos "" vscode
EOF

# Expose some ports if necessary
EXPOSE 8080

WORKDIR /app
# Run program
CMD ["jupyter", "server"]
