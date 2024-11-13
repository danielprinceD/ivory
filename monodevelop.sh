#!/bin/bash
echo "\n\n Downloading GNUPG \n\n"
yes | sudo apt install gnupg 
echo "\n\n Downloading CA-CERTIFICATES \n\n"
yes | sudo apt install ca-certificates
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 3FA7E0328081BFF6A14DA29AA6A19B38D3D831EF
echo "deb https://download.mono-project.com/repo/ubuntu stable-focal main" | sudo tee /etc/apt/sources.list.d/mono-official-stable.list
echo "\n\n Updating apt \n\n"
yes | sudo apt-get update
echo "\n\n Downloading MONODEVELOP UTILS \n\n"
yes | sudo apt-get install monodevelop
yes | sudo apt-get install mono-devel
yes | sudo apt-get install mono-complete
yes | sudo apt-get install mono-dbg
yes | sudo apt-get install ca-certificates-mono
echo -e "\n\n\n\n\n\nDei Parama Padida \n\n"
