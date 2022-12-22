## To run the script

- Create a 3.9 python virtual environment 
- Install packages with `pip install -r requirements.txt`
- Open a prompt and run your virtual environment
- Type `python studentAlcohol.py`

## Script launch options

- -h, --help: display all launch options
- -o, --optimize: run the script to optimise weights only
- -e, --epoch: choose the number of iteration when optimizing weights (by default equals to 10)
- -g, --graph: create a graph using the last saved weights

Example:

- `python studentAlcohol.py -o -e 100`: we will look for the best weights by iterating 100 times
- `python studentAlcohol.py -h`: show help
- `python studentAlcohol.py -g`: create a graph using the last saved weights

## Miscellaneous informations

Scripting: python 3.9 <br>
Hardware: 
- For scripting: CPU i7 6500U 2.5GHz, 16 Go RAM *(20 secs / iterations when optimizing weights)*
- Optimizing: Ryzen 7 5800x, 32 Go RAM *(10 secs / iterations when optimizing weights)*
Software: windows 10