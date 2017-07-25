# OpenAPS
This repository holds the prediction algorithms for OpenAPS.

The OpenAPS data must be in a subdirectory called "data" with a subdirectory of
the ID number that contains the devicestatus.json file. For example:

        ./data/00897741/devicestatus.json
        ./data/01352464/devicestatus.json
        ./data/01884126/devicestatus.json
        ./data/14092221/devicestatus.json
        ./data/15634563/devicestatus.json
        ./data/17161370/devicestatus.json
        ./data/24587372/devicestatus.json
        ./data/40997757/devicestatus.json
        ./data/41663654/devicestatus.json
        ./data/45025419/devicestatus.json
        ./data/46966807/devicestatus.json
        ./data/68267781/devicestatus.json
        ./data/84984656/devicestatus.json
where . represents the current directory with the code

The code requires the following files:

        bgdataframe.py
        bgarray.py
        datamatrix.py
        mlalgorithm.py
        ClarkeErrorGrid.py

This code also requires the following libraries:

        pandas
        numpy
        gatspy
        sklearn

Trevor Tsue

2017-7-24
