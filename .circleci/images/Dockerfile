FROM circleci/python:3.7

# name: Install Git LFS
WORKDIR usr/src/app
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
RUN sudo apt-get install git-lfs && git lfs install
COPY . .

#name: Install Shellcheck
WORKDIR /tmp/shellcheck
RUN curl -OL https://storage.googleapis.com/shellcheck/shellcheck-stable.linux.x86_64.tar.xz
RUN tar xf shellcheck-stable.linux.x86_64.tar.xz
RUN sudo mv shellcheck-stable/shellcheck /usr/local/bin
COPY . .
