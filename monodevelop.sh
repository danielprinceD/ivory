#!/bin/bash
yes | sudo apt install gnupg 
yes | sudo apt install ca-certificates
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 3FA7E0328081BFF6A14DA29AA6A19B38D3D831EF
echo "deb https://download.mono-project.com/repo/ubuntu stable-focal main" | sudo tee /etc/apt/sources.list.d/mono-official-stable.list
yes | sudo apt-get install monodevelop
yes | sudo apt-get install mono-devel
yes | sudo apt-get install mono-complete
yes | sudo apt-get install mono-dbg
yes | sudo apt-get install ca-certificates-mono
echo -e "\n\n\n\n\n\nDei Parama Padida \n\n"
