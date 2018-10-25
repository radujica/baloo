#!/bin/bash

# rust
curl https://sh.rustup.rs -sSf | sh -s -- -y
export PATH="$PATH:$HOME/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin"

# llvm
llvm="clang+llvm-6.0.1-x86_64-linux-gnu-ubuntu-14.04"
wget http://releases.llvm.org/6.0.1/${llvm}".tar.xz"
tar xf ${llvm}".tar.xz"
export PATH="$TRAVIS_BUILD_DIR/${llvm}/bin:$PATH"
ln -s "$TRAVIS_BUILD_DIR/${llvm}/bin/clang++" "$TRAVIS_BUILD_DIR/${llvm}/bin/clang++-6.0"

# weld
git clone https://github.com/weld-project/weld.git
cd weld
cargo build --release

# make sure we return here
cd $TRAVIS_BUILD_DIR
