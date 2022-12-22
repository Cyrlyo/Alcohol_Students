## To run the script

- Create a python virtual environment 
- Install packages with `pip install -r requirements.txt`
- Open a prompt and run your virtual environment
- Type `python studentAlcohol.py`

## Script launch options:

- -h: display all launch options
- -o, --optimize: run the script to optimise weights only
- -e, --epoch: choose the number of iteration when optimizing weights (by default equals to 10)
- -g, --graph: create a graph using the last saved weights

Example:

- `python studentAlcohol.py -o -e 100`: we will look for the best weights by iterating 100 times
- `python studentAlcohol.py -h`: show help
- `python studentAlcohol.py -g`: create a graph using the last saved weights