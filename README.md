## To run the script

- Create a 3.9 python virtual environment 
- Install packages with `pip install -r requirements.txt`
- Open a prompt and run your virtual environment
- Type `python studentAlcohol.py`

Before opening the visualisation and statistics notebooks, you need to run the code to generate the data including the communities. In addition, you will find in the Graph folder the graphs in *.gexf* format in order to visualise them with Gephi.

## Script launch options

- -h, --help: display all launch options
- -o, --optimize: run the script to optimise weights only
- -e, --epoch: choose the number of iteration when optimizing weights (by default equals to 10)
- -g, --graph: create a graph using the last saved weights
- -x, --xgb_weights: Use XGBoost Optimize weights (by default use random optimized weights)

Example:

- `python studentAlcohol.py -o -e 100`: we will look for the best weights by iterating 100 times
- `python studentAlcohol.py -h`: show help
- `python studentAlcohol.py -g`: create a graph using the last random saved weights
- `python studentAlcohol.py -g -x`: create a graph using the last xgboost optimized saved weights

## Miscellaneous informations

Scripting: python 3.9 <br>

Hardware: <br>
For scripting: CPU i7 6500U 2.5GHz, 16 Go RAM

Optimizing: 
- CPU i7 6500U 2.5GHz, 16 Go RAM *(20 secs / iteration when optimizing weights)* 
- CPU i7 7700 3.6GHz, 8 Go RAM, GPU GTX 1050 *(15 sec / iteration when optimizing weights)*
- CPU Ryzen 7 5800x 3.8GHz, 32 Go RAM, GPU RTX 3070 Ti *(10 secs / iteration when optimizing weights)* 

Software: windows 10