#!/bin/bash

## Check if an IP address is provided
if [ $# -eq 0 ]; then
    echo "Error: Please provide an IP address as an argument."
    echo "Usage: $0 <ip_address>"
    exit 1
fi

## Validate IP address format
if ! [[ $1 =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Error: Invalid IP address format. Please use xxx.xxx.xxx.xxx"
    exit 1
fi

## Create directory if it doesn't exist
mkdir -p ~/.stretch

## Write IP address to file
echo "$1" > ~/.stretch/robot_ip.txt

## Verify file creation and content
if [ -f ~/.stretch/robot_ip.txt ]; then
    echo "Success: IP address $1 has been written to ~/.stretch/robot_ip.txt"
else
    echo "Error: Failed to create or write to ~/.stretch/robot_ip.txt"
    exit 1
fi

