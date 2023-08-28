#!/bin/bash

python3 -m pip install -r requirements.txt

for i in 1894780431 1895933568 1896811198 1896812496 1896965671 1898459410; do
    json=${i}.json
    if [ ! -f "$json" ]; then
        chat_downloader -o $json https://www.twitch.tv/videos/${i} > /dev/null
    fi
done

./main.py -n *.json
