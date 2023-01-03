#!/bin/bash

# Manipulation
DIR="datasets/manipulation/fixed/"
RAW="manipulation_fixed"
mkdir -p $DIR
wget -O $RAW https://figshare.com/ndownloader/articles/21277911?private_link=c160601888f51f005344
unzip $RAW -d $DIR && rm $RAW

DIR="datasets/manipulation/varying/"
RAW="manipulation_varying"
mkdir -p $DIR
wget -O $RAW  https://figshare.com/ndownloader/articles/21277914?private_link=f24a7268d63765299b0c
unzip $RAW -d datasets/manipulation/varying/ && rm $RAW 


# Boat/ Water Monitoring Robot
DIR="datasets/boat/fixed/"
RAW="boat_fixed"
mkdir -p $DIR
wget -O $RAW https://figshare.com/ndownloader/articles/21280527?private_link=367dabc7b8aec49a4d5a
unzip $RAW -d $DIR && rm $RAW

DIR="datasets/boat/varying/"
RAW="boat_varying"
mkdir -p $DIR
wget -O $RAW https://figshare.com/ndownloader/articles/21280530?private_link=5db078c218152f1a7d94
unzip $RAW -d $DIR && rm $RAW


# Qcat-6
DIR="datasets/qcat_6/"
RAW="qcat_6"
mkdir -p $DIR
wget -O $RAW https://figshare.com/ndownloader/articles/21280533?private_link=3e337bf73212a776b688
unzip $RAW -d $DIR && rm $RAW


# Qcat
DIR="datasets/qcat/fixed/"
RAW="qcat"
mkdir -p $DIR
wget -O $RAW https://figshare.com/ndownloader/articles/21280548?private_link=26db47f3627a37ab1eab
unzip $RAW -d $DIR && rm $RAW


# PUTany
DIR="datasets/putany/"
RAW="putnay"
mkdir -p $DIR
wget -O $RAW https://figshare.com/ndownloader/articles/21280554?private_link=6ca2330ecfa5e447c5d3
unzip $RAW -d $DIR && rm $RAW