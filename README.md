# Predicting BG using various algorithms 
This repository holds various prediction algorithms for BG, using data from the OpenAPS Data Commons (which comes from Nightscout), with the intention to compare these algorithms to existing in-use prediction algorithms in the open source diabetes community. 

The OpenAPS data must be in a subdirectory called "data" with a subdirectory of
the ID number that contains the devicestatus.json file. For example:

        ./data/12345678/devicestatus.json

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


Once all of these have been satsified, you can run the code with the following command:

        python main.py


